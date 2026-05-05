#!/usr/bin/env python
"""Compare PyKMC legacy basin selection with AMSEL-backed selection."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from pykmc.basins import AmselFPTASelector, FPTASelector, StatesConnectivity


class CycleRng:
    """Repeat a fixed list of probability draws."""

    def __init__(self, draws):
        values = [float(value) for value in draws]
        if not values:
            raise ValueError("draws must contain at least one value")
        self._draws = values
        self._index = 0

    def random(self):
        value = self._draws[self._index % len(self._draws)]
        self._index += 1
        return value


def single_exit_connectivity(rate: float = 2.0) -> StatesConnectivity:
    table = StatesConnectivity()
    table.df = pd.DataFrame(
        {
            "state": [0],
            "state_connexion": [1],
            "k_forward": [float(rate)],
        }
    )
    return table


def load_connectivity(path: str | Path) -> StatesConnectivity:
    table = StatesConnectivity()
    table.df = pd.read_pickle(path)
    return table


def legacy_fpta_selector(rng: CycleRng) -> FPTASelector:
    return FPTASelector(rng=rng)


def amsel_sampled_selector(rng: CycleRng) -> AmselFPTASelector:
    return AmselFPTASelector(clock_mode="sampled", rng=rng)


def amsel_adaptive_selector(rng: CycleRng) -> AmselFPTASelector:
    return AmselFPTASelector(clock_mode="adaptive", rng=rng)


def run_selector(
    *,
    case_name: str,
    selector_name: str,
    selector_factory: Callable[[CycleRng], object],
    table: StatesConnectivity,
    draws: list[float],
) -> dict[str, object]:
    selector = selector_factory(CycleRng(draws))
    start_ns = time.perf_counter_ns()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = selector.select_from_connectivity(table)
            elapsed_ns = time.perf_counter_ns() - start_ns
            if result.is_ok():
                value = result.ok_value()
                return {
                    "case": case_name,
                    "selector": selector_name,
                    "ok": True,
                    "elapsed_ns": elapsed_ns,
                    "t_exit": float(value.t_exit),
                    "exit_state": int(value.exit_state),
                    "warning_count": len(captured),
                    "warnings": [str(item.message) for item in captured],
                    "clock_mode": getattr(selector, "last_clock_mode", None),
                }
            err = result.err_value()
            return {
                "case": case_name,
                "selector": selector_name,
                "ok": False,
                "elapsed_ns": elapsed_ns,
                "error": str(getattr(err, "message", err)),
                "warning_count": len(captured),
                "warnings": [str(item.message) for item in captured],
                "clock_mode": getattr(selector, "last_clock_mode", None),
            }
        except Exception as exc:  # noqa: BLE001 - report comparison failures.
            elapsed_ns = time.perf_counter_ns() - start_ns
            return {
                "case": case_name,
                "selector": selector_name,
                "ok": False,
                "elapsed_ns": elapsed_ns,
                "error": f"{type(exc).__name__}: {exc}",
                "warning_count": len(captured),
                "warnings": [str(item.message) for item in captured],
                "clock_mode": getattr(selector, "last_clock_mode", None),
            }


def amsel_feature_report(table: StatesConnectivity, entry: int = 0) -> dict[str, object]:
    try:
        import amsel
    except ImportError as exc:
        return {"ok": False, "error": f"amsel import failed: {exc}"}

    transient, absorbing, rates = AmselFPTASelector._extract_graph(table)
    if not transient or not absorbing:
        return {
            "ok": False,
            "error": "connectivity table needs transient and absorbing states",
        }
    problem = amsel.AmcProblem(transient=transient, absorbing=absorbing, rates=rates)
    out: dict[str, object] = {
        "ok": True,
        "n_transient": len(transient),
        "n_absorbing": len(absorbing),
        "n_rates": len(rates),
    }
    try:
        mean, variance, cv, second_moment, residual_inf = amsel.mrm_moments(
            transient=transient,
            absorbing=absorbing,
            rates=rates,
            entry=int(entry),
        )
        out["mrm_moments"] = {
            "ok": True,
            "mean": float(mean),
            "variance": float(variance),
            "cv": float(cv),
            "second_moment": float(second_moment),
            "residual_inf": float(residual_inf),
        }
    except Exception as exc:  # noqa: BLE001 - report feature-level evidence.
        out["mrm_moments"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        reduced = problem.reduced_kinetics(entry=int(entry))
        out["reduced_kinetics"] = {
            "ok": True,
            "slow_subspace_rank": int(reduced.slow_subspace_rank),
            "effective_mode_count": float(reduced.effective_mode_count),
            "rank1_invalidity": float(reduced.rank1_invalidity),
            "initial_hazard": float(reduced.initial_hazard),
            "effective_rate": float(reduced.effective_rate),
            "tail_rate": float(reduced.tail_rate),
        }
    except Exception as exc:  # noqa: BLE001 - report feature-level evidence.
        out["reduced_kinetics"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

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
        except Exception as exc:  # noqa: BLE001 - report feature-level evidence.
            outlet_reports.append(
                {
                    "ok": False,
                    "absorbing_state": int(outlet),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    out["ngt_outlets"] = outlet_reports
    return out


def _parse_draws(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--connectivity",
        type=Path,
        default=Path("tests/data/basin_connectivity_Cu_fake.pickle"),
        help="Pickled StatesConnectivity/DataFrame fixture to replay.",
    )
    parser.add_argument("--case-name", default="pykmc-connectivity")
    parser.add_argument("--draws", default="0.5,0.0")
    parser.add_argument("--entry", type=int, default=0)
    args = parser.parse_args(argv)

    table = load_connectivity(args.connectivity)
    draws = _parse_draws(args.draws)
    reports = [
        run_selector(
            case_name=args.case_name,
            selector_name="legacy-fpta",
            selector_factory=legacy_fpta_selector,
            table=table,
            draws=draws,
        ),
        run_selector(
            case_name=args.case_name,
            selector_name="amsel-sampled",
            selector_factory=amsel_sampled_selector,
            table=table,
            draws=draws,
        ),
        run_selector(
            case_name=args.case_name,
            selector_name="amsel-adaptive",
            selector_factory=amsel_adaptive_selector,
            table=table,
            draws=draws,
        ),
    ]
    payload = {
        "selectors": reports,
        "amsel_features": amsel_feature_report(table, entry=args.entry),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
