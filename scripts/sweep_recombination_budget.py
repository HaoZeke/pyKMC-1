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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
