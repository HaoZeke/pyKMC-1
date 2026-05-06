#!/usr/bin/env python
"""Rank PyKMC catalog exit channels with AMSEL NGT committors."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import amsel
except ImportError as exc:  # pragma: no cover - exercised only without amsel.
    amsel = None
    _AMSEL_IMPORT_ERROR: Exception | None = exc
else:
    _AMSEL_IMPORT_ERROR = None


REQUIRED_COLUMNS = ("state", "state_connexion", "k_forward")
CATALOG_COLUMNS = (
    "event_connexion",
    "central_atom",
    "sym",
    "transient",
    "dE_forward",
    "k_forward",
    "dE_backward",
    "k_backward",
)


class ExitChannel:
    def __init__(self, channel_id: int, row_index: int, row: pd.Series) -> None:
        self.channel_id = channel_id
        self.row_index = row_index
        self.row = row


def load_connectivity(path: str | Path) -> Any:
    return pd.read_pickle(path)


def catalog_guidance_payload(
    connectivity_table: Any,
    *,
    case_name: str = "pykmc-catalog",
    entry: int = 0,
) -> dict[str, Any]:
    if amsel is None:
        return {
            "ok": False,
            "case": case_name,
            "entry": int(entry),
            "error": f"amsel is not installed ({_AMSEL_IMPORT_ERROR!r})",
        }

    try:
        df = _connectivity_df(connectivity_table)
        transient, absorbing, rates, channels, guidance_states = _split_absorbing_channels(df)
    except (TypeError, ValueError) as exc:
        return {
            "ok": False,
            "case": case_name,
            "entry": int(entry),
            "error": str(exc),
        }

    if not transient:
        return {
            "ok": False,
            "case": case_name,
            "entry": int(entry),
            "error": "connectivity table contains no transient source states",
        }
    if not channels:
        return {
            "ok": False,
            "case": case_name,
            "entry": int(entry),
            "error": "connectivity table contains no absorbing catalog rows",
        }

    problem = amsel.AmcProblem(transient=transient, absorbing=absorbing, rates=rates)
    rows = []
    for channel in channels:
        item = _channel_row(channel)
        try:
            result = problem.ngt(source=[int(entry)], target=[int(channel.channel_id)])
        except Exception as exc:  # noqa: BLE001 - report per-channel failures.
            item.update(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "committor": 0.0,
                    "rate": 0.0,
                    "mfpt": None,
                    "guidance_score": 0.0,
                }
            )
        else:
            committor = float(result.committor)
            item.update(
                {
                    "ok": True,
                    "committor": committor,
                    "rate": float(result.rate),
                    "mfpt": float(result.mfpt),
                    "guidance_score": committor,
                }
            )
        rows.append(item)

    rows.sort(
        key=lambda item: (
            bool(item.get("ok", False)),
            float(item.get("guidance_score", 0.0)),
            float(item.get("rate", 0.0)),
        ),
        reverse=True,
    )
    ok_rows = [row for row in rows if row["ok"]]
    transient_rows = _transient_state_guidance(
        problem=problem,
        df=df,
        guidance_states=guidance_states,
        entry=int(entry),
    )
    ok_transient_rows = [row for row in transient_rows if row["ok"]]
    return {
        "ok": len(ok_rows) == len(rows) and len(ok_transient_rows) == len(transient_rows),
        "case": case_name,
        "entry": int(entry),
        "n_transient": len(transient),
        "n_absorbing_channels": len(channels),
        "n_rates": len(rates),
        "channel_summary": {
            "ok": len(ok_rows) == len(rows),
            "committor_sum": sum(float(row["committor"]) for row in ok_rows),
            "rate_sum": sum(float(row["rate"]) for row in ok_rows),
            "failed_channels": len(rows) - len(ok_rows),
        },
        "channels": rows,
        "transient_summary": {
            "ok": len(ok_transient_rows) == len(transient_rows),
            "failed_states": len(transient_rows) - len(ok_transient_rows),
        },
        "transient_states": transient_rows,
    }


def live_basin_guidance_payload(
    *,
    config_path: Path,
    initial_config: Path,
    reference_table: Path,
    visited_environments: Path,
    case_name: str,
    entry: int,
    exploration_priority: str | None = None,
    max_expansions: int | None = None,
    max_closed_states: int | None = None,
) -> dict[str, Any] | None:
    from pykmc import Config, ReferenceEventTable, System
    from pykmc.basins import BasinsGenericEvents, StatesConnectivity
    from pykmc.enginemanager.lmpi.pool import ManagerFactory

    config = Config.from_ini_file(str(config_path))
    config.control.reference_table = str(reference_table)
    if exploration_priority is not None:
        config.basin.exploration_priority = exploration_priority
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
        result = basin.construct_connexion_table(
            max_expansions=max_expansions,
            max_closed_states=max_closed_states,
        )
        if not result.is_ok():
            payload = {
                "ok": False,
                "source": "live-lammps-mpi",
                "case": case_name,
                "stage": "construct_connexion_table",
                "error": str(result.err_value()),
            }
            _attach_exploration_payload(
                payload,
                basin=basin,
                priority=config.basin.exploration_priority,
                max_expansions=max_expansions,
                max_closed_states=max_closed_states,
            )
            return payload

        mapping = basin.connectivity_table.reorder_states_index()
        basin.states = {mapping[old]: val for old, val in basin.states.items()}
        constructed_table = StatesConnectivity()
        constructed_table.df = basin.connectivity_table.get_table().copy()

        manager.use_local()
        result = basin.refine_absorbing(system)
        table, refinement = _refinement_table_and_report(
            result,
            refined_table=basin.connectivity_table,
            catalog_table=constructed_table,
        )

        payload = catalog_guidance_payload(table, case_name=case_name, entry=entry)
        payload["source"] = "live-lammps-mpi"
        payload["refinement"] = refinement
        _attach_exploration_payload(
            payload,
            basin=basin,
            priority=config.basin.exploration_priority,
            max_expansions=max_expansions,
            max_closed_states=max_closed_states,
        )
        return payload
    finally:
        manager.close_all()


def write_csv(payload: dict[str, Any], out) -> None:
    rows = list(payload.get("channels", []))
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
    else:
        fieldnames = ["ok", "error"]
        rows = [{"ok": payload.get("ok", False), "error": payload.get("error", "")}]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _refinement_table_and_report(
    result: Any,
    *,
    refined_table: Any,
    catalog_table: Any,
) -> tuple[Any, dict[str, Any]]:
    if result.is_ok():
        return refined_table, {"ok": True, "rate_source": "refined"}
    if refined_table is None:
        return (
            catalog_table,
            {
                "ok": False,
                "stage": "refine_absorbing",
                "error": str(result.err_value()),
                "rate_source": "catalog",
            },
        )
    return (
        refined_table,
        {
            "ok": False,
            "stage": "refine_absorbing",
            "error": str(result.err_value()),
            "rate_source": "partial-refined",
        },
    )


def _attach_exploration_payload(
    payload: dict[str, Any],
    *,
    basin: Any,
    priority: str,
    max_expansions: int | None,
    max_closed_states: int | None,
) -> None:
    scores = getattr(basin, "last_exploration_guidance", {}) or {}
    payload["exploration"] = {
        "priority": priority,
        "max_expansions": max_expansions,
        "max_closed_states": max_closed_states,
        "expanded_states": [int(state) for state in getattr(basin, "exploration_order", [])],
        "closed_states": [int(state) for state in getattr(basin, "explored_states", [])],
        "states_to_explore": [int(state) for state in getattr(basin, "states_to_explore", [])],
        "guidance_scores": {
            str(int(state)): float(score)
            for state, score in sorted(scores.items(), key=lambda item: int(item[0]))
        },
    }


def _connectivity_df(connectivity_table: Any) -> pd.DataFrame:
    df = getattr(connectivity_table, "df", connectivity_table)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("connectivity input must be a pandas DataFrame or expose a .df DataFrame")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"connectivity table missing required columns: {missing}")
    if df.empty:
        raise ValueError("connectivity table is empty")
    return df.copy()


def _split_absorbing_channels(
    df: pd.DataFrame,
) -> tuple[
    list[int],
    list[int],
    list[tuple[int, int, float]],
    list[ExitChannel],
    list[int],
]:
    transient = sorted({int(value) for value in df["state"].to_numpy()})
    transient_set = set(transient)
    all_state_ids = {
        int(value)
        for value in pd.concat([df["state"], df["state_connexion"]], ignore_index=True)
    }
    next_channel_id = max(all_state_ids) + 1
    rates: list[tuple[int, int, float]] = []
    channels: list[ExitChannel] = []
    frontier_targets: set[int] = set()

    for row_index, row in df.iterrows():
        source = int(row["state"])
        target = int(row["state_connexion"])
        rate = float(row["k_forward"])
        if _is_transient_destination(row, target, transient_set):
            rates.append((source, target, rate))
            if target not in transient_set:
                frontier_targets.add(target)
            continue

        channel_id = next_channel_id
        next_channel_id += 1
        rates.append((source, channel_id, rate))
        channels.append(ExitChannel(channel_id=channel_id, row_index=int(row_index), row=row))

    absorbing = sorted(frontier_targets) + [channel.channel_id for channel in channels]
    guidance_states = sorted(transient_set | frontier_targets)
    return transient, absorbing, rates, channels, guidance_states


def _is_transient_destination(
    row: pd.Series,
    target: int,
    transient_set: set[int],
) -> bool:
    if "transient" in row.index and not pd.isna(row["transient"]):
        return bool(row["transient"])
    return target in transient_set


def _channel_row(channel: ExitChannel) -> dict[str, Any]:
    row = channel.row
    item = {
        "row_index": int(channel.row_index),
        "channel_id": int(channel.channel_id),
        "state": int(row["state"]),
        "state_connexion": int(row["state_connexion"]),
    }
    for column in CATALOG_COLUMNS:
        if column in row.index:
            item[column] = _json_scalar(row[column])
    return item


def _transient_state_guidance(
    *,
    problem: Any,
    df: pd.DataFrame,
    guidance_states: list[int],
    entry: int,
) -> list[dict[str, Any]]:
    rows = []
    for state in guidance_states:
        if state == entry:
            continue
        item = _incoming_transient_row(df, state)
        item["state"] = int(state)
        try:
            result = problem.ngt(source=[int(entry)], target=[int(state)])
        except Exception as exc:  # noqa: BLE001 - diagnostics report per-state failures.
            item.update(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "hit_committor": 0.0,
                    "rate": 0.0,
                    "mfpt": None,
                    "guidance_score": 0.0,
                }
            )
        else:
            hit_committor = float(result.committor)
            item.update(
                {
                    "ok": True,
                    "hit_committor": hit_committor,
                    "rate": float(result.rate),
                    "mfpt": float(result.mfpt),
                    "guidance_score": hit_committor,
                }
            )
        rows.append(item)

    rows.sort(
        key=lambda item: (
            bool(item.get("ok", False)),
            float(item.get("guidance_score", 0.0)),
            float(item.get("rate", 0.0)),
        ),
        reverse=True,
    )
    return rows


def _incoming_transient_row(df: pd.DataFrame, state: int) -> dict[str, Any]:
    incoming = df[df["state_connexion"].astype(int) == int(state)].copy()
    item: dict[str, Any] = {"incoming_count": int(len(incoming))}
    if incoming.empty:
        return item

    incoming = incoming.sort_values("k_forward", ascending=False)
    row = incoming.iloc[0]
    item["incoming_state"] = int(row["state"])
    item["incoming_rate"] = float(row["k_forward"])
    for column in ("event_connexion", "central_atom", "sym"):
        if column in row.index:
            item[f"incoming_{column}"] = _json_scalar(row[column])
    return item


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--connectivity",
        type=Path,
        default=Path("tests/data/basin_connectivity_Cu_fake.pickle"),
        help="Pickled connectivity DataFrame or object exposing a .df DataFrame.",
    )
    parser.add_argument("--case-name", default="pykmc-catalog")
    parser.add_argument("--entry", type=int, default=0)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--live-basin",
        action="store_true",
        help="Build and refine the basin through the live LAMMPS/MPI manager.",
    )
    parser.add_argument(
        "--exploration-priority",
        choices=("auto", "legacy", "amsel"),
        help="Override [Basin].exploration_priority for live basin construction.",
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        help="Stop live basin construction after this many expanded transient states.",
    )
    parser.add_argument(
        "--max-closed-states",
        type=int,
        help="Stop live basin construction after this many closed queue states.",
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

    if args.live_basin:
        payload = live_basin_guidance_payload(
            config_path=args.config,
            initial_config=args.initial_config,
            reference_table=args.reference_table,
            visited_environments=args.visited_environments,
            case_name=args.case_name,
            entry=args.entry,
            exploration_priority=args.exploration_priority,
            max_expansions=args.max_expansions,
            max_closed_states=args.max_closed_states,
        )
        if payload is None:
            return 0
    else:
        payload = catalog_guidance_payload(
            load_connectivity(args.connectivity),
            case_name=args.case_name,
            entry=args.entry,
        )

    if args.out is None:
        if args.format == "csv":
            import sys

            write_csv(payload, sys.stdout)
        else:
            print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        with args.out.open("w", newline="") as handle:
            if args.format == "csv":
                write_csv(payload, handle)
            else:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
