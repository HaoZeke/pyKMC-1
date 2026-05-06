#!/usr/bin/env python
"""Run recombination measurements across basin exploration budgets."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


AGGREGATES = {
    "trials": ("trials.csv", "sweep_trials.csv"),
    "basin_confidence": ("basin_confidence.csv", "sweep_basin_confidence.csv"),
    "basin_trace": ("basin_trace.csv", "sweep_basin_trace.csv"),
}
BUDGET_FIELDS = [
    "budget_label",
    "basin_max_closed_states",
    "basin_max_absorbing_refinements",
]
SUMMARY_FIELDS = [
    "budget_label",
    "basin_max_closed_states",
    "basin_max_absorbing_refinements",
    "selector",
    "trials",
    "kinetic_claim_ok",
    "recombined",
    "last_frontier_states",
    "last_frontier_committor",
    "last_frontier_rate",
    "last_order",
    "last_queue",
    "last_top_queue_state",
    "last_top_queue_guidance",
]


def budget_sweep_commands(
    *,
    measure_script: Path,
    python: str,
    out: Path,
    closed_states: list[int],
    absorbing_refinements: list[int],
    measure_args: list[str],
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for closed in closed_states:
        for absorb in absorbing_refinements:
            label = f"closed-{int(closed)}_absorb-{int(absorb)}"
            run_out = out / label
            commands.append(
                {
                    "budget_label": label,
                    "basin_max_closed_states": int(closed),
                    "basin_max_absorbing_refinements": int(absorb),
                    "out": str(run_out),
                    "command": [
                        python,
                        str(measure_script),
                        *measure_args,
                        "--basin-max-closed-states",
                        str(int(closed)),
                        "--basin-max-absorbing-refinements",
                        str(int(absorb)),
                        "--work-budget",
                        label,
                        "--out",
                        str(run_out),
                    ],
                }
            )
    return commands


def run_budget_sweep(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for command in commands:
        result = subprocess.run(
            command["command"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        run_out = Path(command["out"])
        run_out.mkdir(parents=True, exist_ok=True)
        (run_out / "sweep_harness.log").write_text(result.stdout)
        results.append(
            {
                "budget_label": command["budget_label"],
                "basin_max_closed_states": command["basin_max_closed_states"],
                "basin_max_absorbing_refinements": command[
                    "basin_max_absorbing_refinements"
                ],
                "returncode": result.returncode,
                "out": command["out"],
            }
        )
    return results


def write_aggregate_outputs(out: Path, commands: list[dict[str, Any]]) -> None:
    for _, (source_name, target_name) in AGGREGATES.items():
        rows: list[dict[str, Any]] = []
        fieldnames = list(BUDGET_FIELDS)
        for command in commands:
            source = Path(command["out"]) / source_name
            if not source.exists():
                continue
            with source.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for field in reader.fieldnames or []:
                    if field not in fieldnames:
                        fieldnames.append(field)
                for row in reader:
                    rows.append(
                        {
                            "budget_label": command["budget_label"],
                            "basin_max_closed_states": command[
                                "basin_max_closed_states"
                            ],
                            "basin_max_absorbing_refinements": command[
                                "basin_max_absorbing_refinements"
                            ],
                            **row,
                        }
                    )
        _write_csv(out / target_name, rows, fieldnames)
    write_sweep_summary(out)


def write_sweep_summary(out: Path) -> None:
    trial_rows = _read_csv(out / "sweep_trials.csv")
    confidence_rows = _read_csv(out / "sweep_basin_confidence.csv")
    trace_rows = _read_csv(out / "sweep_basin_trace.csv")
    groups = sorted(
        {
            _group_key(row)
            for rows in (trial_rows, confidence_rows, trace_rows)
            for row in rows
        }
    )
    summary_rows: list[dict[str, Any]] = []
    for key in groups:
        trials = [row for row in trial_rows if _group_key(row) == key]
        confidence = [row for row in confidence_rows if _group_key(row) == key]
        trace = [row for row in trace_rows if _group_key(row) == key]
        last_confidence = _last_step_row(confidence)
        last_trace = _last_step_row(trace)
        summary_rows.append(
            {
                "budget_label": key[0],
                "basin_max_closed_states": key[1],
                "basin_max_absorbing_refinements": key[2],
                "selector": key[3],
                "trials": len(trials),
                "kinetic_claim_ok": sum(
                    1 for row in trials if _csv_bool(row.get("kinetic_claim_ok"))
                ),
                "recombined": sum(
                    1 for row in trials if _csv_bool(row.get("recombined"))
                ),
                "last_frontier_states": last_confidence.get("frontier_states"),
                "last_frontier_committor": last_confidence.get("frontier_committor"),
                "last_frontier_rate": last_confidence.get("frontier_rate"),
                "last_order": last_trace.get("order"),
                "last_queue": last_trace.get("queue"),
                "last_top_queue_state": last_trace.get("top_queue_state"),
                "last_top_queue_guidance": last_trace.get("top_queue_guidance"),
            }
        )
    _write_csv(out / "sweep_summary.csv", summary_rows, SUMMARY_FIELDS)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("budget_label", "")),
        str(row.get("basin_max_closed_states", "")),
        str(row.get("basin_max_absorbing_refinements", "")),
        str(row.get("selector", "")),
    )


def _last_step_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda row: int(row.get("step") or 0))


def _csv_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def _parse_int_list(text: str) -> list[int]:
    values = [int(item) for item in text.split(",") if item]
    if not values:
        raise ValueError("budget list must contain at least one integer")
    return values


def _measure_args(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure-script", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--closed-states", required=True)
    parser.add_argument("--absorbing-refinements", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("measure_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    closed_states = _parse_int_list(args.closed_states)
    absorbing_refinements = _parse_int_list(args.absorbing_refinements)
    commands = budget_sweep_commands(
        measure_script=args.measure_script,
        python=args.python,
        out=args.out,
        closed_states=closed_states,
        absorbing_refinements=absorbing_refinements,
        measure_args=_measure_args(args.measure_args),
    )
    manifest = {
        "measure_script": str(args.measure_script),
        "python": args.python,
        "closed_states": closed_states,
        "absorbing_refinements": absorbing_refinements,
        "dry_run": bool(args.dry_run),
        "measure_args": _measure_args(args.measure_args),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "sweep_manifest.json", manifest)
    write_json(args.out / "sweep_commands.json", commands)
    if args.dry_run:
        return 0

    results = run_budget_sweep(commands)
    write_json(args.out / "sweep_results.json", results)
    write_aggregate_outputs(args.out, commands)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
