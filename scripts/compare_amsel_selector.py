#!/usr/bin/env python
"""Compare PyKMC legacy basin selection with AMSEL-backed selection."""
from __future__ import annotations

import argparse
import json
import pickle
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
    selector = AmselFPTASelector()
    return selector.diagnose_connectivity(table, entry=int(entry), include_outlets=True)


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
            selector_name="amsel-adaptive",
            selector_factory=amsel_adaptive_selector,
            table=table,
            draws=draws,
        ),
    ]
    df = table.get_table()
    states = set(df["state"]).union(set(df["state_connexion"])) if not df.empty else set()
    return {
        "source": source,
        "connectivity": {
            "rows": int(len(df)),
            "states": int(len(states)),
            "transient_rows": int(df["transient"].sum()) if "transient" in df else None,
        },
        "selectors": reports,
        "amsel_features": amsel_feature_report(table, entry=entry),
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
