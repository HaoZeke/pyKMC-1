"""Module for detecting unique symmetry of an atomic environment based on atomic displacements."""

import ira_mod
import numpy as np
import os
import subprocess
import sys
import tempfile


def _identity_symmetry(nat: int) -> tuple[np.ndarray, np.ndarray]:
    return np.array([np.eye(3)]), np.array([np.arange(nat)])


def _sofi_subprocess_available() -> bool:
    return os.environ.get("PYKMC_DISABLE_IRA_SOFI", "").lower() not in {
        "1",
        "true",
        "yes",
    }


def _compute_sofi_symmetries(
    initial_positions: np.ndarray,
    sym_thr: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    code = """
import numpy as np
import sys
import ira_mod

positions = np.load(sys.argv[1])
sym_thr = float(sys.argv[3])
nat = len(positions)
sym = ira_mod.SOFI().compute(nat, nat * [1], positions, sym_thr)
np.savez(sys.argv[2], matrix=np.asarray(sym.matrix), perm=np.asarray(sym.perm))
"""
    timeout_s = float(os.environ.get("PYKMC_IRA_SOFI_TIMEOUT_S", "30"))
    with tempfile.TemporaryDirectory(prefix="pykmc-sofi-") as tmpdir:
        input_path = os.path.join(tmpdir, "positions.npy")
        output_path = os.path.join(tmpdir, "symmetries.npz")
        np.save(input_path, np.asarray(initial_positions, dtype=float))
        try:
            result = subprocess.run(
                [sys.executable, "-c", code, input_path, output_path, str(sym_thr)],
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
                return np.asarray(data["matrix"]), np.asarray(data["perm"], dtype=int)
        except Exception:
            return None


def unique_symmetries(
    initial_positions: np.ndarray, final_positions: np.ndarray, sym_thr: float
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Identify the unique symmetry operations of an event based on atomic displacements.

    This function computes all the symmetry operations of the initial configuration using `ira_mod`,
    then filters out equivalent operations by comparing the associated atomic displacements after applying the symmetries.

    Parameters
    ----------
    initial_positions : np.ndarray
        Initial atomic positions (N, 3).
    final_positions : np.ndarray
        Final atomic positions (N, 3).
    sym_thr : float
        Symmetry tolerance threshold for the `ira_mod` symmetry detection.

    Returns
    -------
    sym_matrix : list[np.ndarray]
        Arrays of unique 3,3 symmetry rotation matrices, including the identity. Shape: (M, 3, 3).
    sym_perm : list[np.ndarray]
        Arrays of corresponding atom index permutations for each symmetry. Shape: (M, N),
        where M is the number of unique symmetries and N the number of atoms.

    """
    # Find all symmetries of initial_positions
    initial_positions = np.asarray(initial_positions, dtype=float)
    final_positions = np.asarray(final_positions, dtype=float)
    nat = len(initial_positions)
    if not _sofi_subprocess_available():
        return _identity_symmetry(nat)
    sym = _compute_sofi_symmetries(initial_positions, sym_thr)
    if sym is None:
        return _identity_symmetry(nat)
    sym_matrix_all, sym_perm_all = sym

    # Find unique symmetries
    # Displacment event matrix
    displacements = initial_positions - final_positions

    unique_displacements = [displacements]
    unique_sym_index = []

    for i in range(len(sym_matrix_all)):  # Loop over all symmetries
        is_duplicated = False
        # Apply symmetry to displacements event matrix
        new_displacements = displacements @ sym_matrix_all[i].T
        new_displacements = new_displacements[sym_perm_all[i]]

        for disp in unique_displacements:  # Check if alreay in unique_displacements
            if np.allclose(disp, new_displacements, atol=1e-2, rtol=0):
                is_duplicated = True
                break

        if not is_duplicated:  # if new unique symmetry
            unique_sym_index.append(i)  # add symmtry to unique
            unique_displacements.append(new_displacements)

    # unique symetries and add identity :
    sym_matrix = np.concatenate(
        [[np.eye(3)]] + [[sym_matrix_all[i]] for i in unique_sym_index], axis=0
    )
    # associated permutation :
    sym_perm = np.array([np.arange(nat)] + [sym_perm_all[i] for i in unique_sym_index])
    return sym_matrix, sym_perm
