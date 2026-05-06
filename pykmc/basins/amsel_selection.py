"""Amsel-backed basin selector with optional adaptive clocking.

Mirrors :class:`pykmc.basins.selection.FPTASelector` but delegates the
numerical work to `amsel <https://pypi.org/project/amsel>`_ -- a pure
Rust implementation of the corrected Ferasat 2020 two-step FPTA with
zero-copy DLPack tensor I/O and free-threaded Python wheels.

The API intentionally matches ``FPTASelector`` so that code using
:meth:`select_from_connectivity` can swap the class out with no other
changes. Behaviour is mathematically equivalent for well-conditioned
problems (condition number below ~1e11 in the current amsel v0.1
envelope); the Rust kernel is typically an order of magnitude faster
than the scipy path on superbasins of size 10 and above, and scales
linearly under free-threaded Python because the amsel kernel runs
without holding the GIL.

If ``amsel`` is not importable or rejects the problem (for example on
ill-conditioned matrices outside its f64 envelope) the selector
returns an ``ErrorInfo`` and the caller can fall back to
:class:`FPTASelector`.

References
----------
[1] Puchala, Falk, Garikipati, J. Chem. Phys. 132, 134104 (2010) --
    the original FPTA formulation.
[2] Ferasat et al., J. Chem. Phys. 153, 074109 (2020) -- the two-step
    correction amsel implements.
[3] `amsel <https://github.com/lode-org/amsel>`_ -- the Rust kernel.
"""
from __future__ import annotations

import numpy as np

from pykmc.result import (
    BasinSelectorOutput,
    Err,
    ErrorInfo,
    ErrorType,
    Ok,
    Result,
)

from .connectivity import StatesConnectivity


try:
    import amsel as _amsel

    _AMSEL_AVAILABLE = True
    _AMSEL_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised only when amsel missing
    _amsel = None
    _AMSEL_AVAILABLE = False
    _AMSEL_IMPORT_ERROR = exc


class AmselFPTASelector:
    """Basin selector with the Rust ``amsel`` kernel as its backend.

    Parameters
    ----------
    clock_mode
        ``"sampled"`` keeps the exact sampled FPTA clock, ``"mean"``
        forces the scalar MRM mean clock, and ``"adaptive"`` uses a
        sampled rank-1 reduced clock when diagnostics support it.
    rank_tol
        Tolerance passed to ``ReducedKineticsResult.one_rate_clock_is_plausible``.
    rng
        Random number provider with a ``random()`` method. Defaults to
        ``numpy.random`` to preserve the existing selector API.
    """

    def __init__(self, clock_mode: str = "adaptive", rank_tol: float = 1.0e-8, rng=None) -> None:
        if clock_mode not in {"sampled", "mean", "adaptive"}:
            raise ValueError("clock_mode must be one of 'sampled', 'mean', or 'adaptive'")
        self.last_t_exit: float | None = None
        self.last_weights: np.ndarray | None = None
        self.last_clock_mode: str | None = None
        self.last_reduced_kinetics = None
        self.last_diagnostics: dict[str, object] | None = None
        self.clock_mode = clock_mode
        self.rank_tol = rank_tol
        self.rng = np.random
        if rng is not None:
            self.rng = rng

    def select_from_connectivity(
        self, connectivity_table: StatesConnectivity
    ) -> Result[BasinSelectorOutput, ErrorInfo]:
        """Find an exit time and exit state for the given basin.

        Parameters
        ----------
        connectivity_table
            :class:`StatesConnectivity` describing the superbasin's
            transient-to-transient and transient-to-absorbing edges.

        Returns
        -------
        Result[BasinSelectorOutput, ErrorInfo]
            Ok(BasinSelectorOutput) on success; Err on amsel absence,
            amsel kernel rejection, or degenerate input.
        """
        if not _AMSEL_AVAILABLE:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message=(
                        "amsel is not installed "
                        f"({_AMSEL_IMPORT_ERROR!r}); fall back to FPTASelector "
                        "or install amsel>=0.1"
                    ),
                )
            )

        transient, absorbing, rates = self._extract_graph(connectivity_table)
        if not transient:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message="no transient states in connectivity table",
                )
            )
        if not absorbing:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message="no absorbing states in connectivity table",
                )
            )

        # Entry is state 0 by pyKMC convention (see FPTASelector).
        entry = transient[0]
        problem = _amsel.AmcProblem(
            transient=transient,
            absorbing=absorbing,
            rates=rates,
        )

        self.last_reduced_kinetics = None
        self.last_clock_mode = None

        try:
            if self.clock_mode == "sampled":
                fpta_res = problem.fpta(entry=entry, r=float(self.rng.random()))
                t_exit = float(fpta_res.t_exit)
                weights_arr = np.asarray(fpta_res.weights, dtype=np.float64)
                self.last_clock_mode = "sampled"
            else:
                rk = problem.reduced_kinetics(entry=entry)
                self.last_reduced_kinetics = rk
                diagnostics = self.diagnose_connectivity(
                    connectivity_table,
                    entry=entry,
                    include_outlets=False,
                )
                use_reduced_clock = self.clock_mode == "mean" or (
                    self._mean_clock_diagnostics_ok(diagnostics)
                    and rk.one_rate_clock_is_plausible(self.rank_tol)
                )
                if use_reduced_clock:
                    mrm_res = problem.mrm(entry=entry)
                    effective_rate = float(rk.effective_rate)
                    if self.clock_mode == "mean":
                        t_exit = float(mrm_res.tau_total)
                        weights_arr = (
                            np.asarray(mrm_res.rate_to_absorbing, dtype=np.float64) * t_exit
                        )
                        self.last_clock_mode = "mean"
                    elif np.isfinite(effective_rate) and effective_rate > 0.0:
                        r = float(self.rng.random())
                        t_exit = float(-np.log1p(-r) / effective_rate)
                        weights_arr = (
                            np.asarray(mrm_res.rate_to_absorbing, dtype=np.float64)
                            / effective_rate
                        )
                        self.last_clock_mode = "reduced-sampled"
                    else:
                        fpta_res = problem.fpta(entry=entry, r=float(self.rng.random()))
                        t_exit = float(fpta_res.t_exit)
                        weights_arr = np.asarray(fpta_res.weights, dtype=np.float64)
                        self.last_clock_mode = "sampled"
                else:
                    fpta_res = problem.fpta(entry=entry, r=float(self.rng.random()))
                    t_exit = float(fpta_res.t_exit)
                    weights_arr = np.asarray(fpta_res.weights, dtype=np.float64)
                    self.last_clock_mode = "sampled"
        except _amsel.AmselError as exc:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message=f"amsel kernel rejected problem: {exc}",
                )
            )

        weight_sum = float(np.sum(weights_arr))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message="amsel returned non-positive absorbing weights",
                )
            )
        weights_arr = weights_arr / weight_sum
        cumul = np.cumsum(weights_arr)
        r2 = float(self.rng.random())
        idx = int(np.searchsorted(cumul, r2))
        idx = min(idx, len(absorbing) - 1)
        exit_state = absorbing[idx]

        self.last_t_exit = float(t_exit)
        self.last_weights = weights_arr

        return Ok(BasinSelectorOutput(t_exit=float(t_exit), exit_state=int(exit_state)))

    def diagnose_connectivity(
        self,
        connectivity_table: StatesConnectivity,
        entry: int | None = None,
        include_outlets: bool = True,
    ) -> dict[str, object]:
        """Return AMSEL diagnostics for a PyKMC basin graph.

        Diagnostics are independent: a failed moment solve does not hide
        reduced-kinetics or NGT outlet information. Adaptive selection uses
        this report to avoid scalar mean clocks when the moment system is
        ill-conditioned, while still allowing sampled FPTA exits.
        """
        if not _AMSEL_AVAILABLE:
            report = {
                "ok": False,
                "error": f"amsel is not installed ({_AMSEL_IMPORT_ERROR!r})",
            }
            self.last_diagnostics = report
            return report

        transient, absorbing, rates = self._extract_graph(connectivity_table)
        if not transient or not absorbing:
            report = {
                "ok": False,
                "error": "connectivity table needs transient and absorbing states",
            }
            self.last_diagnostics = report
            return report

        entry = transient[0] if entry is None else int(entry)
        problem = _amsel.AmcProblem(transient=transient, absorbing=absorbing, rates=rates)
        report: dict[str, object] = {
            "ok": True,
            "entry": int(entry),
            "n_transient": len(transient),
            "n_absorbing": len(absorbing),
            "n_rates": len(rates),
        }
        try:
            mean, variance, cv, second_moment, residual_inf = _amsel.mrm_moments(
                transient=transient,
                absorbing=absorbing,
                rates=rates,
                entry=int(entry),
            )
            report["mrm_moments"] = {
                "ok": True,
                "mean": float(mean),
                "variance": float(variance),
                "cv": float(cv),
                "second_moment": float(second_moment),
                "residual_inf": float(residual_inf),
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics should be non-blocking.
            report["mrm_moments"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        try:
            reduced = problem.reduced_kinetics(entry=int(entry))
            report["reduced_kinetics"] = {
                "ok": True,
                "slow_subspace_rank": int(reduced.slow_subspace_rank),
                "effective_mode_count": float(reduced.effective_mode_count),
                "rank1_invalidity": float(reduced.rank1_invalidity),
                "initial_hazard": float(reduced.initial_hazard),
                "effective_rate": float(reduced.effective_rate),
                "tail_rate": float(reduced.tail_rate),
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics should be non-blocking.
            report["reduced_kinetics"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        if include_outlets:
            outlet_reports = []
            for outlet in absorbing:
                try:
                    ngt = problem.ngt(source=[int(entry)], target=[int(outlet)])
                    outlet_reports.append(
                        {
                            "ok": True,
                            "absorbing_state": int(outlet),
                            "rate": float(ngt.rate),
                            "mfpt": float(ngt.mfpt),
                            "committor": float(ngt.committor),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - report per-outlet failures.
                    outlet_reports.append(
                        {
                            "ok": False,
                            "absorbing_state": int(outlet),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            report["ngt_outlets"] = outlet_reports
            ok_outlets = [item for item in outlet_reports if item["ok"]]
            if len(ok_outlets) == len(outlet_reports):
                mfpts = [float(item["mfpt"]) for item in ok_outlets]
                mfpt_mean = float(np.mean(mfpts))
                if mfpt_mean > 0.0:
                    mfpt_rel_spread = float((max(mfpts) - min(mfpts)) / mfpt_mean)
                else:
                    mfpt_rel_spread = 0.0
                report["ngt_summary"] = {
                    "ok": True,
                    "committor_sum": float(
                        sum(float(item["committor"]) for item in ok_outlets)
                    ),
                    "rate_sum": float(sum(float(item["rate"]) for item in ok_outlets)),
                    "mfpt_mean": mfpt_mean,
                    "mfpt_rel_spread": mfpt_rel_spread,
                }
            else:
                report["ngt_summary"] = {
                    "ok": False,
                    "failed_outlets": len(outlet_reports) - len(ok_outlets),
                }

        self.last_diagnostics = report
        return report

    @staticmethod
    def _mean_clock_diagnostics_ok(diagnostics: dict[str, object]) -> bool:
        if not diagnostics.get("ok", False):
            return False
        moments = diagnostics.get("mrm_moments")
        reduced = diagnostics.get("reduced_kinetics")
        if not isinstance(moments, dict) or not isinstance(reduced, dict):
            return False
        return bool(moments.get("ok", False)) and bool(reduced.get("ok", False))

    @staticmethod
    def _extract_graph(
        connectivity_table: StatesConnectivity,
    ) -> tuple[list[int], list[int], list[tuple[int, int, float]]]:
        """Flatten the StatesConnectivity DataFrame into amsel input form.

        Transient states are the unique values in the ``state`` column;
        absorbing states are the values appearing in ``state_connexion``
        but not in ``state``. Each DataFrame row contributes one
        ``(from, to, rate)`` edge using ``k_forward`` as the rate. The
        ordering of ``transient`` and ``absorbing`` lists is stable so
        repeated calls produce deterministic amsel indexing.
        """
        df = connectivity_table.df
        transient_set = set(int(s) for s in df["state"].to_numpy())
        connexion_set = set(int(s) for s in df["state_connexion"].to_numpy())
        absorbing_set = connexion_set - transient_set
        transient = sorted(transient_set)
        absorbing = sorted(absorbing_set)
        rates: list[tuple[int, int, float]] = []
        for _, row in df.iterrows():
            rates.append(
                (
                    int(row["state"]),
                    int(row["state_connexion"]),
                    float(row["k_forward"]),
                )
            )
        return transient, absorbing, rates
