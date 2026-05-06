#!/usr/bin/env python
"""Compare PyKMC legacy basin selection with AMSEL-backed selection."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pykmc.basins import AmselFPTASelector, FPTASelector, StatesConnectivity
from pykmc.basins.utils import solve_master_equation


class CycleRng:
    """Repeat a fixed list of probability draws."""

    def __init__(self, draws):
        values = [float(value) for value in draws]
        if not values:
            raise ValueError("draws must contain at least one value")
        self._draws = values
        self._index = 0
        self.consumed: list[float] = []

    def random(self):
        value = self._draws[self._index % len(self._draws)]
        self._index += 1
        self.consumed.append(value)
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


def amsel_mean_selector(rng: CycleRng) -> AmselFPTASelector:
    return AmselFPTASelector(clock_mode="mean", rng=rng)


def amsel_adaptive_selector(rng: CycleRng) -> AmselFPTASelector:
    return AmselFPTASelector(clock_mode="adaptive", rng=rng)


def selector_clock_report(
    *,
    selector_name: str,
    clock_mode: str | None,
    consumed_draws: list[float],
) -> dict[str, object]:
    if selector_name == "legacy-fpta" or clock_mode in {"sampled", "reduced-sampled"}:
        semantics = "sampled-quantile"
    elif selector_name == "amsel-mean" or clock_mode == "mean":
        semantics = "deterministic-mfpt"
    else:
        semantics = "unresolved"

    if semantics == "sampled-quantile":
        time_draw = consumed_draws[0] if consumed_draws else None
        outlet_draw = consumed_draws[1] if len(consumed_draws) > 1 else None
    elif semantics == "deterministic-mfpt":
        time_draw = None
        outlet_draw = consumed_draws[0] if consumed_draws else None
    else:
        time_draw = None
        outlet_draw = None

    return {
        "clock_semantics": semantics,
        "time_draw": time_draw,
        "outlet_draw": outlet_draw,
        "consumed_draws": consumed_draws,
    }


def run_selector(
    *,
    case_name: str,
    selector_name: str,
    selector_factory: Callable[[CycleRng], object],
    table: StatesConnectivity,
    draws: list[float],
) -> dict[str, object]:
    rng = CycleRng(draws)
    selector = selector_factory(rng)
    start_ns = time.perf_counter_ns()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            result = selector.select_from_connectivity(table)
            elapsed_ns = time.perf_counter_ns() - start_ns
            clock_mode = getattr(selector, "last_clock_mode", None)
            clock_report = selector_clock_report(
                selector_name=selector_name,
                clock_mode=clock_mode,
                consumed_draws=list(rng.consumed),
            )
            if result.is_ok():
                value = result.ok_value()
                master_report = master_equation_report(
                    table=table,
                    t_exit=float(value.t_exit),
                    time_draw=clock_report["time_draw"],
                    outlet_draw=clock_report["outlet_draw"],
                    exit_state=int(value.exit_state),
                )
                return {
                    "case": case_name,
                    "selector": selector_name,
                    "ok": True,
                    "elapsed_ns": elapsed_ns,
                    "t_exit": float(value.t_exit),
                    "exit_state": int(value.exit_state),
                    "warning_count": len(captured),
                    "warnings": [str(item.message) for item in captured],
                    "clock_mode": clock_mode,
                    **clock_report,
                    **master_report,
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
                "clock_mode": clock_mode,
                **clock_report,
            }
        except Exception as exc:  # noqa: BLE001 - report comparison failures.
            elapsed_ns = time.perf_counter_ns() - start_ns
            clock_mode = getattr(selector, "last_clock_mode", None)
            clock_report = selector_clock_report(
                selector_name=selector_name,
                clock_mode=clock_mode,
                consumed_draws=list(rng.consumed),
            )
            return {
                "case": case_name,
                "selector": selector_name,
                "ok": False,
                "elapsed_ns": elapsed_ns,
                "error": f"{type(exc).__name__}: {exc}",
                "warning_count": len(captured),
                "warnings": [str(item.message) for item in captured],
                "clock_mode": clock_mode,
                **clock_report,
            }


def _real_array(values) -> np.ndarray:
    arr = np.real_if_close(np.asarray(values), tol=1000)
    return np.asarray(arr, dtype=np.float64)


def master_equation_report(
    *,
    table: StatesConnectivity,
    t_exit: float,
    time_draw: float | None,
    outlet_draw: float | None,
    exit_state: int,
) -> dict[str, float | bool | int | None]:
    selector = FPTASelector()
    selector.build_absorbing_matrix_from_connectivity(table)
    n_transient = len(set(table.df["state"]))
    selector.build_reduced_matrix(n_transient)

    p0_full = np.zeros(len(selector.M_abs))
    p0_full[0] = 1.0
    p_full = _real_array(
        solve_master_equation(
            M=selector.M_abs,
            t=float(t_exit),
            p0=p0_full,
            spectral_decomposition=False,
        )
    )

    p0 = np.zeros(len(selector.M_abs_reduced))
    p0[0] = 1.0
    p_reduced = _real_array(
        solve_master_equation(
            M=selector.M_abs_reduced,
            t=float(t_exit),
            p0=p0,
            spectral_decomposition=False,
        )
    )

    absorbing_states = list(range(n_transient, len(selector.M_abs)))
    p_absorbing = p_full[n_transient:]
    full_absorbed = float(np.sum(p_absorbing))
    reduced_absorbed = float(p_reduced[-1])

    time_error = None
    stable_absorption = None
    stable_time_error = None
    if time_draw is not None:
        stable_absorption = full_absorbed
        time_error = full_absorbed - float(time_draw)
        stable_time_error = time_error

    selected_probability = None
    cdf_lower = None
    cdf_upper = None
    quantile_hit = None
    quantile_state = None
    selected_index = int(exit_state) - n_transient
    if full_absorbed > 0.0 and 0 <= selected_index < len(p_absorbing):
        conditional = p_absorbing / full_absorbed
        cdf = np.cumsum(conditional)
        selected_probability = float(conditional[selected_index])
        cdf_lower = float(cdf[selected_index - 1]) if selected_index > 0 else 0.0
        cdf_upper = float(cdf[selected_index])
        if outlet_draw is not None:
            draw_index = int(np.searchsorted(cdf, float(outlet_draw)))
            draw_index = min(draw_index, len(absorbing_states) - 1)
            quantile_state = int(absorbing_states[draw_index])
            quantile_hit = quantile_state == int(exit_state)

    return {
        "stable_absorption_probability": stable_absorption,
        "time_draw_absorption_error": stable_time_error,
        "master_equation_absorption_probability": full_absorbed,
        "reduced_master_equation_absorption_probability": reduced_absorbed,
        "master_reduced_absorption_error": reduced_absorbed - full_absorbed,
        "time_draw_master_equation_error": time_error,
        "master_equation_probability_sum": float(np.sum(p_full)),
        "master_equation_min_probability": float(np.min(p_full)),
        "master_equation_selected_exit_probability": selected_probability,
        "master_equation_outlet_cdf_lower": cdf_lower,
        "master_equation_outlet_cdf_upper": cdf_upper,
        "master_equation_outlet_quantile_hit": quantile_hit,
        "master_equation_outlet_quantile_state": quantile_state,
    }


def amsel_feature_report(table: StatesConnectivity, entry: int = 0) -> dict[str, object]:
    selector = AmselFPTASelector()
    return selector.diagnose_connectivity(table, entry=int(entry), include_outlets=True)


def rank1_clock_reference(
    *,
    features: dict[str, object],
    draws: list[float],
) -> dict[str, object]:
    reduced = features.get("reduced_kinetics")
    if not isinstance(reduced, dict) or not reduced.get("ok", False):
        return {"ok": False, "error": "reduced kinetics diagnostics unavailable"}
    if not draws:
        return {"ok": False, "error": "at least one probability draw is required"}

    effective_rate = float(reduced["effective_rate"])
    if not AmselFPTASelector._reduced_clock_consistent(
        features,
        effective_rate=effective_rate,
    ):
        return {
            "ok": False,
            "error": "reduced effective rate is inconsistent with MFPT",
        }
    time_draw = float(draws[0])
    if not math.isfinite(effective_rate) or effective_rate <= 0.0:
        return {"ok": False, "error": "effective rate must be positive and finite"}
    if not 0.0 <= time_draw < 1.0:
        return {"ok": False, "error": "time draw must satisfy 0 <= r < 1"}

    sampled_quantile_time = -math.log1p(-time_draw) / effective_rate
    mfpt = 1.0 / effective_rate
    ratio = math.inf
    if sampled_quantile_time > 0.0:
        ratio = mfpt / sampled_quantile_time

    return {
        "ok": True,
        "semantics": {
            "sampled-quantile": "inverse-cdf first-passage sample at the reported time draw",
            "deterministic-mfpt": "mean first-passage time of the reduced rank-1 clock",
        },
        "time_draw": time_draw,
        "effective_rate": effective_rate,
        "sampled_quantile_time": sampled_quantile_time,
        "mfpt": mfpt,
        "mfpt_over_sampled_quantile_time": ratio,
    }


def selector_payload(
    *,
    table: StatesConnectivity,
    case_name: str,
    draws: list[float],
    entry: int = 0,
    source: str = "pickle",
) -> dict[str, object]:
    reports = [
        run_selector(
            case_name=case_name,
            selector_name="legacy-fpta",
            selector_factory=legacy_fpta_selector,
            table=table,
            draws=draws,
        ),
        run_selector(
            case_name=case_name,
            selector_name="amsel-sampled",
            selector_factory=amsel_sampled_selector,
            table=table,
            draws=draws,
        ),
        run_selector(
            case_name=case_name,
            selector_name="amsel-mean",
            selector_factory=amsel_mean_selector,
            table=table,
            draws=draws,
        ),
        run_selector(
            case_name=case_name,
            selector_name="amsel-adaptive",
            selector_factory=amsel_adaptive_selector,
            table=table,
            draws=draws,
        ),
    ]
    df = table.get_table()
    states = set(df["state"]).union(set(df["state_connexion"])) if not df.empty else set()
    features = amsel_feature_report(table, entry=entry)
    return {
        "source": source,
        "draws": [float(value) for value in draws],
        "connectivity": {
            "rows": int(len(df)),
            "states": int(len(states)),
            "transient_rows": int(df["transient"].sum()) if "transient" in df else None,
        },
        "selectors": reports,
        "amsel_features": features,
        "clock_reference": rank1_clock_reference(features=features, draws=draws),
    }


def live_basin_payload(
    *,
    config_path: Path,
    initial_config: Path,
    reference_table: Path,
    visited_environments: Path,
    case_name: str,
    draws: list[float],
    entry: int,
) -> dict[str, object] | None:
    from pykmc import Config, ReferenceEventTable, System
    from pykmc.basins import BasinsGenericEvents
    from pykmc.enginemanager.lmpi.pool import ManagerFactory

    config = Config.from_ini_file(str(config_path))
    config.control.reference_table = str(reference_table)
    system = System.create_from_file(str(initial_config))
    references = ReferenceEventTable(config)
    with visited_environments.open("rb") as handle:
        known_environments = pickle.load(handle)

    factory = ManagerFactory(n_sessions=config.control.n_sessions, use_rank_0=True)
    manager = factory.launch()
    if manager is None:
        return None

    try:
        manager.initialize_sessions(config, system)
        basin = BasinsGenericEvents(
            config=config,
            reference_table=references,
            known_environments=known_environments,
            manager=manager,
        )
        basin._initialize(system)
        result = basin.construct_connexion_table()
        if not result.is_ok():
            return {
                "source": "live-lammps-mpi",
                "case": case_name,
                "ok": False,
                "stage": "construct_connexion_table",
                "error": str(result.err_value()),
            }
        mapping = basin.connectivity_table.reorder_states_index()
        basin.states = {mapping[old]: val for old, val in basin.states.items()}
        constructed_table = StatesConnectivity()
        constructed_table.df = basin.connectivity_table.get_table().copy()
        manager.use_local()
        result = basin.refine_absorbing(system)
        if not result.is_ok():
            payload = selector_payload(
                table=constructed_table,
                case_name=case_name,
                draws=draws,
                entry=entry,
                source="live-lammps-mpi",
            )
            payload["ok"] = True
            payload["refinement"] = {
                "ok": False,
                "stage": "refine_absorbing",
                "error": str(result.err_value()),
                "rate_source": "catalog",
            }
            return payload
        payload = selector_payload(
            table=basin.connectivity_table,
            case_name=case_name,
            draws=draws,
            entry=entry,
            source="live-lammps-mpi",
        )
        payload["ok"] = True
        payload["refinement"] = {"ok": True, "rate_source": "refined"}
        return payload
    finally:
        manager.close_all()


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
    parser.add_argument(
        "--live-basin",
        action="store_true",
        help="Build the Cu basin connectivity through the live LAMMPS/MPI manager.",
    )
    parser.add_argument("--config", type=Path, default=Path("tests/data/input_Cu.in"))
    parser.add_argument(
        "--initial-config",
        type=Path,
        default=Path("tests/data/initial_config_Cu.xyz"),
    )
    parser.add_argument(
        "--reference-table",
        type=Path,
        default=Path("tests/data/reference_table_Cu_fake.pickle"),
    )
    parser.add_argument(
        "--visited-environments",
        type=Path,
        default=Path("tests/data/visited_environments_Cu.pickle"),
    )
    args = parser.parse_args(argv)

    draws = _parse_draws(args.draws)
    if args.live_basin:
        payload = live_basin_payload(
            config_path=args.config,
            initial_config=args.initial_config,
            reference_table=args.reference_table,
            visited_environments=args.visited_environments,
            case_name=args.case_name,
            draws=draws,
            entry=args.entry,
        )
        if payload is None:
            return 0
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    table = load_connectivity(args.connectivity)
    payload = selector_payload(
        table=table,
        case_name=args.case_name,
        draws=draws,
        entry=args.entry,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
