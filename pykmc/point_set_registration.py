"""Manages Point Set Registration (shape matching) methods."""

import json
import ira_mod
import numpy as np
import os
import subprocess
import sys
import tempfile
from scipy.optimize import linear_sum_assignment
from .result import Result, ErrorInfo, PSROutput, Ok, Err, ErrorType
from .config import Config 
from .system import System 
import pandas as pd 
from .neighbors_list import NeighborsList


class PointSetRegistration:
    """Perform a point set registration between a reference event and an atomic environment of an atom based on the configuration parameters.

    Parameters
    ----------
    config : Config
        The configuration.
    system : System
        The atomic system.
    dfevent : pd.Series
        The reference event.
    neighbors_list : NeighborsList
        The NeighborsList of the System.
    central_atom_index : int
        Index of the central atom in the System for which we want to perfrom the point set registration.

    """

    def __init__(
        self, config: Config, system: System, dfevent: pd.Series, neighbors_list: NeighborsList, central_atom_index: int
    ) -> None:
        self.system = system
        self.config = config
        self.dfevent = dfevent
        self.neighbors_list = neighbors_list
        self.central_atom_index = central_atom_index
        self.psr_style = self.config.psr.style

    def match(self) -> Result[PSROutput, ErrorInfo]:
        """Run the point set registration based on the style defined in the configuration.

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            Results of the point set registration.

        Raises
        ------
        Exception
            If the style in not known.

        """
        match self.psr_style:
            case "ira":
                return self.ira(self.central_atom_index)
            case _:
                raise Exception("Point set registration style unknown")

    def ira(self, central_atom_index: int) -> Result[PSROutput, ErrorInfo]:
        """Use IRA to extract rotation, translation, permutation matrix to apply on generic event.

        Parameters
        ----------
        central_atom_index : int
           index of the system's central atom

        Returns
        -------
        Result[PSROutput, ErrorInfo]
            The results of the ira psr procedure.

        """
        # Event informations :
        coords2 = self.dfevent.at["initial_positions"]
        nat2 = len(coords2)

        # atom in the rcutevent around the central atom
        neighbor_list = self.neighbors_list.get_neighbors("rcut", central_atom_index)

        coords1 = self.system.positions[neighbor_list]


        system_types = _system_neighbor_types(self.system, neighbor_list, len(coords1))
        event_types = _event_types(self.dfevent, nat2)
        if system_types is not None and event_types is not None:
            typ1 = system_types
            typ2 = event_types
        else:
            typ1 = ["X"] * len(coords1)
            typ2 = ["X"] * nat2

        # unwrap if close to cell limits :
        alat = self.system.cell[0][0]
        for i in range(len(coords1)):
            if (
                np.linalg.norm(
                    coords1[i][0] - self.system.positions[central_atom_index][0]
                )
                > alat / 2
            ):
                coords1[i][0] = (
                    coords1[i][0]
                    + np.sign(
                        self.system.positions[central_atom_index][0] - coords1[i][0]
                    )
                    * alat
                )
            if (
                np.linalg.norm(
                    coords1[i][1] - self.system.positions[central_atom_index][1]
                )
                > alat / 2
            ):
                coords1[i][1] = (
                    coords1[i][1]
                    + np.sign(
                        self.system.positions[central_atom_index][1] - coords1[i][1]
                    )
                    * alat
                )
            if (
                np.linalg.norm(
                    coords1[i][2] - self.system.positions[central_atom_index][2]
                )
                > alat / 2
            ):
                coords1[i][2] = (
                    coords1[i][2]
                    + np.sign(
                        self.system.positions[central_atom_index][2] - coords1[i][2]
                    )
                    * alat
                )
        nat1 = len(coords1)
        kmax_factor = self.config.ira.kmax_factor

        result = simple_ira(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor)
        if result.is_ok():
            return result
        fallback = _translation_psr_fallback(
            coords1,
            coords2,
            typ1=system_types,
            typ2=event_types,
        )
        if fallback is not None:
            return Ok(fallback)
        return result


def check_match(
    result_match: Result[PSROutput, ErrorInfo], matching_score: float
) -> Result[PSROutput, ErrorInfo]:
    """Check if a result from the point set registration method is valid and gives a matching score lower than the matching score threshold defined in the configuration.

    Parameters
    ----------
    result_match : Result[PSROutput, ErrorInfo]
        Result of the PSR procedure.
    matching_score : float
        matching score threshold.

    Returns
    -------
    Result[PSROutput, ErrorInfo]
        Result of the check.

    """
    if not result_match.is_ok():
        return result_match  # ErrorInfo no match
    else:
        if result_match.ok_value().matching_score > matching_score:
            return Err(
                ErrorInfo(
                    type=ErrorType.PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD,
                    message="PSR found a match but matching score is above acceptance threshold",
                    details="Hausdorff distance = {}, acceptance threshold = {} ".format(
                        result_match.ok_value().matching_score, matching_score
                    ),
                    variables={"matching_score": result_match.ok_value().matching_score}
                )
            )
        
        else:
            return result_match  # Ok(PSROutput)


def _psr_no_match(message: str = "IRA did not find a match"):
    return Err(
        ErrorInfo(
            type=ErrorType.PSR_NO_MATCH_FOUND,
            message=message,
        )
    )


def _system_neighbor_types(system, neighbor_list, natoms: int) -> list[str] | None:
    types = getattr(system, "types", None)
    if types is None:
        return None
    values = np.asarray(types)[np.asarray(neighbor_list, dtype=int)]
    if len(values) != natoms:
        return None
    return [str(atom_type) for atom_type in values.tolist()]


def _event_types(dfevent: pd.Series, natoms: int) -> list[str] | None:
    if "types" not in dfevent:
        return None
    value = dfevent.get("types", None)
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    values = np.asarray(value)
    if values.shape[0] != natoms:
        return None
    return [str(atom_type) for atom_type in values.tolist()]


def _translation_psr_fallback(
    coords1,
    coords2,
    typ1: list[str] | None = None,
    typ2: list[str] | None = None,
) -> PSROutput | None:
    current = np.asarray(coords1, dtype=float)
    reference = np.asarray(coords2, dtype=float)
    if current.shape != reference.shape:
        return None
    if current.size == 0:
        return PSROutput(
            rotation_matrix=np.eye(3),
            translation_matrix=np.zeros(3),
            permutation_matrix=np.arange(0),
            matching_score=0.0,
        )
    translation = current.mean(axis=0) - reference.mean(axis=0)
    mapped = reference + translation
    distances = np.linalg.norm(current[:, None, :] - mapped[None, :, :], axis=2)
    if typ1 is not None and typ2 is not None:
        current_types = np.asarray(typ1, dtype=str)
        reference_types = np.asarray(typ2, dtype=str)
        if len(current_types) != len(current) or len(reference_types) != len(reference):
            return None
        allowed = current_types[:, None] == reference_types[None, :]
        if not np.all(allowed.any(axis=1)) or not np.all(allowed.any(axis=0)):
            return None
        costs = np.where(allowed, distances, 1.0e30)
    else:
        allowed = None
        costs = distances
    row_ind, col_ind = linear_sum_assignment(costs)
    if len(row_ind) != len(current):
        return None
    if allowed is not None and not np.all(allowed[row_ind, col_ind]):
        return None
    permutation = np.empty(len(current), dtype=int)
    permutation[row_ind] = col_ind
    residual = current - mapped[permutation]
    score = float(np.max(np.linalg.norm(residual, axis=1)))
    return PSROutput(
        rotation_matrix=np.eye(3),
        translation_matrix=translation,
        permutation_matrix=permutation,
        matching_score=score,
    )


def _ira_match_subprocess_available() -> bool:
    return os.environ.get("PYKMC_DISABLE_IRA_MATCH", "").lower() not in {
        "1",
        "true",
        "yes",
    }


def _run_ira_match_subprocess(
    nat1,
    typ1,
    coords1,
    nat2,
    typ2,
    coords2,
    kmax_factor,
):
    code = """
import json
import numpy as np
import sys
import ira_mod

data = np.load(sys.argv[1])
typ1 = json.loads(sys.argv[3])
typ2 = json.loads(sys.argv[4])
kmax_factor = float(sys.argv[5])
rmat, tr, perm, dh = ira_mod.IRA().match(
    int(data["nat1"]),
    typ1,
    data["coords1"],
    int(data["nat2"]),
    typ2,
    data["coords2"],
    kmax_factor,
)
np.savez(sys.argv[2], rmat=np.asarray(rmat), tr=np.asarray(tr), perm=np.asarray(perm), dh=float(dh))
"""
    timeout_s = float(os.environ.get("PYKMC_IRA_MATCH_TIMEOUT_S", "30"))
    with tempfile.TemporaryDirectory(prefix="pykmc-ira-") as tmpdir:
        input_path = os.path.join(tmpdir, "match-input.npz")
        output_path = os.path.join(tmpdir, "match-output.npz")
        np.savez(
            input_path,
            nat1=int(nat1),
            coords1=np.asarray(coords1, dtype=float),
            nat2=int(nat2),
            coords2=np.asarray(coords2, dtype=float),
        )
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    input_path,
                    output_path,
                    json.dumps(list(typ1)),
                    json.dumps(list(typ2)),
                    str(float(kmax_factor)),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0 or not os.path.exists(output_path):
            return None
        try:
            with np.load(output_path) as data:
                return (
                    np.asarray(data["rmat"]),
                    np.asarray(data["tr"]),
                    np.asarray(data["perm"]),
                    float(data["dh"]),
                )
        except Exception:
            return None


def simple_ira(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor) : 
    if not _ira_match_subprocess_available():
        return _psr_no_match("IRA matcher backend unavailable")
    matched = _run_ira_match_subprocess(
        nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor
    )
    if matched is None:
        return _psr_no_match()
    rmat, tr, perm, dh = matched
    return Ok(
        PSROutput(
            rotation_matrix=rmat,
            translation_matrix=tr,
            permutation_matrix=perm,
            matching_score=dh,
        )
    )
