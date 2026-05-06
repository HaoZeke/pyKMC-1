#!/usr/bin/env python
"""Measure vacancy-SIA recombination kinetics for PyKMC AMSEL comparisons."""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

CRYSTAL_TERMINATION_MARKER = "Only atoms with cristalline environment"
TRIAL_FIELDS = [
    "case",
    "selector",
    "trial",
    "seed",
    "recombined",
    "t_recombination_s",
    "censored_time_s",
    "kmc_steps",
    "detector_reason",
    "failed_refinements",
    "failed_refinement_committor",
    "usable_resolved_committor",
    "kinetic_claim_ok",
    "output_dir",
]
SURVIVAL_FIELDS = ["selector", "time_s", "n_at_risk", "n_events", "survival"]


def detect_recombination_from_log(text: str) -> dict[str, object]:
    if CRYSTAL_TERMINATION_MARKER in text:
        return {
            "recombined": True,
            "detector_reason": "all-crystal-environments",
        }
    return {
        "recombined": False,
        "detector_reason": "censored",
    }


def seed_schedule(
    *,
    base_seed: int,
    trials: int,
    priorities: list[str],
) -> list[dict[str, object]]:
    return [
        {"priority": priority, "trial": trial, "seed": int(base_seed) + trial}
        for trial in range(int(trials))
        for priority in priorities
    ]


def parse_pykmc_out(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("Step"):
            continue
        if set(line) == {"-"}:
            continue

        parts = line.split()
        if len(parts) < 8:
            continue
        rows.append(
            {
                "step": int(parts[0]),
                "dt_s": float(parts[1]),
                "time_s": float(parts[2]),
                "ref_event": int(parts[3]),
                "ea_ev": float(parts[4]),
                "k_evt_ps": float(parts[5]),
                "k_tot_ps": float(parts[6]),
                "energy_ev": float(parts[7]),
            }
        )
    return rows


def trial_row_from_outputs(
    *,
    case: str,
    selector: str,
    trial: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_rows = parse_pykmc_out(output_dir / "pykmc.out")
    log_text = (output_dir / "pykmc.log").read_text()
    detector = detect_recombination_from_log(log_text)
    final_time = output_rows[-1]["time_s"] if output_rows else 0.0
    recombined = bool(detector["recombined"])
    return {
        "case": case,
        "selector": selector,
        "trial": int(trial),
        "seed": int(seed),
        "recombined": recombined,
        "t_recombination_s": final_time if recombined else None,
        "censored_time_s": None if recombined else final_time,
        "kmc_steps": len(output_rows),
        "detector_reason": detector["detector_reason"],
        "output_dir": str(output_dir),
    }


def survival_rows(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    selectors = sorted({str(trial["selector"]) for trial in trials})
    for selector in selectors:
        group = [trial for trial in trials if str(trial["selector"]) == selector]
        event_times = sorted(
            {
                float(trial["t_recombination_s"])
                for trial in group
                if trial.get("recombined") and trial.get("t_recombination_s") is not None
            }
        )
        survival = Fraction(1, 1)
        for event_time in event_times:
            n_at_risk = sum(_trial_time(trial) >= event_time for trial in group)
            n_events = sum(
                trial.get("recombined")
                and trial.get("t_recombination_s") is not None
                and float(trial["t_recombination_s"]) == event_time
                for trial in group
            )
            if n_at_risk == 0:
                continue
            survival *= Fraction(n_at_risk - n_events, n_at_risk)
            rows.append(
                {
                    "selector": selector,
                    "time_s": event_time,
                    "n_at_risk": int(n_at_risk),
                    "n_events": int(n_events),
                    "survival": float(survival),
                }
            )
    return rows


def _trial_time(trial: dict[str, Any]) -> float:
    if trial.get("recombined"):
        return float(trial["t_recombination_s"])
    return float(trial["censored_time_s"])


def trial_commands(
    *,
    out: Path,
    case: str,
    template_input: Path,
    initial_config: Path,
    reference_table: Path | None,
    visited_environments: Path | None,
    partn_path: Path,
    priorities: list[str],
    trials: int,
    seed: int,
    max_steps: int,
    work_budget: str | None,
    mpi_ranks: int,
    mpirun: str,
    python: str,
) -> list[dict[str, Any]]:
    commands = []
    template_text = template_input.read_text()
    for item in seed_schedule(base_seed=seed, trials=trials, priorities=priorities):
        trial_dir = out / str(item["priority"]) / f"trial-{item['trial']}"
        write_trial_input(
            trial_dir / "input.in",
            template_text=template_text,
            initial_config=initial_config,
            reference_table=reference_table,
            visited_environments=visited_environments,
            max_steps=max_steps,
            priority=str(item["priority"]),
            partn_path=partn_path,
            seed=int(item["seed"]),
        )
        commands.append(
            {
                "case": case,
                "priority": item["priority"],
                "trial": item["trial"],
                "seed": item["seed"],
                "max_steps": int(max_steps),
                "work_budget": work_budget,
                "workdir": str(trial_dir),
                "command": [
                    mpirun,
                    "-np",
                    str(int(mpi_ranks)),
                    python,
                    "-m",
                    "pykmc",
                    "-in",
                    "input.in",
                ],
            }
        )
    return commands


def render_trial_input(
    *,
    template_text: str,
    initial_config: Path,
    reference_table: Path | None,
    visited_environments: Path | None,
    max_steps: int,
    priority: str,
    partn_path: Path,
    seed: int,
) -> str:
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(template_text)
    control = _section(config, "Control")
    partn = _section(config, "pARTn")
    basin = _section(config, "BASIN")

    config[control]["initial_config"] = str(initial_config)
    config[control]["n_steps"] = str(int(max_steps))
    config[control]["basin"] = "True"
    if reference_table is not None:
        config[control]["reference_table"] = str(reference_table)
    if visited_environments is not None:
        config[control]["visited_environments"] = str(visited_environments)
    config[partn]["path_artnso"] = str(partn_path)
    config[partn]["zseed"] = str(int(seed))
    config[basin]["exploration_priority"] = priority

    from io import StringIO

    out = StringIO()
    config.write(out)
    return out.getvalue()


def write_trial_input(
    path: Path,
    *,
    template_text: str,
    initial_config: Path,
    reference_table: Path | None,
    visited_environments: Path | None,
    max_steps: int,
    priority: str,
    partn_path: Path,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_trial_input(
            template_text=template_text,
            initial_config=initial_config,
            reference_table=reference_table,
            visited_environments=visited_environments,
            max_steps=max_steps,
            priority=priority,
            partn_path=partn_path,
            seed=seed,
        )
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def execute_trials(commands: list[dict[str, Any]], events_path: Path) -> list[dict[str, Any]]:
    rows = []
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w") as event_handle:
        for command in commands:
            workdir = Path(command["workdir"])
            result = subprocess.run(
                command["command"],
                cwd=workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            (workdir / "harness.log").write_text(result.stdout)
            event_handle.write(
                json.dumps(
                    {
                        "case": command["case"],
                        "priority": command["priority"],
                        "trial": command["trial"],
                        "seed": command["seed"],
                        "returncode": result.returncode,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            if (workdir / "pykmc.out").exists() and (workdir / "pykmc.log").exists():
                rows.append(
                    trial_row_from_outputs(
                        case=str(command["case"]),
                        selector=str(command["priority"]),
                        trial=int(command["trial"]),
                        seed=int(command["seed"]),
                        output_dir=workdir,
                    )
                )
            else:
                rows.append(
                    {
                        "case": command["case"],
                        "selector": command["priority"],
                        "trial": int(command["trial"]),
                        "seed": int(command["seed"]),
                        "recombined": False,
                        "t_recombination_s": None,
                        "censored_time_s": 0.0,
                        "kmc_steps": 0,
                        "detector_reason": f"returncode-{result.returncode}",
                        "output_dir": str(workdir),
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("ni-vac-sia", "cu-vac-sia"), required=True)
    parser.add_argument("--template-input", type=Path)
    parser.add_argument("--initial-config", type=Path)
    parser.add_argument("--reference-table", type=Path)
    parser.add_argument("--visited-environments", type=Path)
    parser.add_argument("--partn-path", type=Path, required=True)
    parser.add_argument(
        "--priority",
        choices=("legacy", "amsel"),
        action="append",
        required=True,
    )
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--work-budget")
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    defaults = case_defaults(args.case)
    template_input = args.template_input or defaults["template_input"]
    initial_config = args.initial_config or defaults["initial_config"]

    commands = trial_commands(
        out=args.out,
        case=args.case,
        template_input=template_input,
        initial_config=initial_config,
        reference_table=args.reference_table,
        visited_environments=args.visited_environments,
        partn_path=args.partn_path,
        priorities=args.priority,
        trials=args.trials,
        seed=args.seed,
        max_steps=args.max_steps,
        work_budget=args.work_budget,
        mpi_ranks=args.mpi_ranks,
        mpirun=args.mpirun,
        python=args.python,
    )
    manifest = {
        "case": args.case,
        "template_input": str(template_input),
        "initial_config": str(initial_config),
        "reference_table": str(args.reference_table) if args.reference_table else None,
        "visited_environments": (
            str(args.visited_environments) if args.visited_environments else None
        ),
        "partn_path": str(args.partn_path),
        "priorities": args.priority,
        "trials": args.trials,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "work_budget": args.work_budget,
        "mpi_ranks": args.mpi_ranks,
        "dry_run": bool(args.dry_run),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "commands.json", commands)
    events_path = args.out / "events.jsonl"
    if args.dry_run:
        write_csv(args.out / "trials.csv", [], TRIAL_FIELDS)
        write_csv(args.out / "survival.csv", [], SURVIVAL_FIELDS)
        events_path.write_text("")
        return 0

    trial_rows = execute_trials(commands, events_path)
    write_csv(args.out / "trials.csv", trial_rows, TRIAL_FIELDS)
    write_csv(args.out / "survival.csv", survival_rows(trial_rows), SURVIVAL_FIELDS)
    return 0


def case_defaults(case: str) -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    if case == "ni-vac-sia":
        example_dir = repo_root / "examples" / "Ni_fcc_4000at_monovacancy+sia"
        return {
            "template_input": example_dir / "input.in",
            "initial_config": example_dir / "initial_config.xyz",
        }
    if case == "cu-vac-sia":
        example_dir = repo_root / "examples" / "Cu_fcc_sia"
        return {
            "template_input": example_dir / "input.in",
            "initial_config": example_dir / "cu_fcc_defects.xyz",
        }
    raise ValueError(f"unknown case: {case}")


def _section(config: configparser.ConfigParser, name: str) -> str:
    for section in config.sections():
        if section.lower() == name.lower():
            return section
    config.add_section(name)
    return name


if __name__ == "__main__":
    raise SystemExit(main())
