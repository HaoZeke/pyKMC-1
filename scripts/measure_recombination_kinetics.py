#!/usr/bin/env python
"""Measure vacancy-SIA recombination kinetics for PyKMC AMSEL comparisons."""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

try:
    import amsel as _amsel
except ImportError:  # pragma: no cover - exercised in environments without AMSEL.
    _amsel = None

CRYSTAL_TERMINATION_MARKER = "Only atoms with cristalline environment"
ZERO_EVENT_DISCOVERY_MARKER = "No events have been found, empty reference events table"
BASIN_REFINEMENT_BUDGET_RE = re.compile(
    r"Basin absorbing refinement skipped (?P<count>\d+) exits; "
    r"unresolved_committor=(?P<committor>[0-9.eE+-]+); "
    r"unresolved_rate=(?P<rate>[0-9.eE+-]+)"
)
BASIN_FRONTIER_BUDGET_RE = re.compile(
    r"Basin exploration budget left (?P<count>\d+) frontier states; "
    r"unresolved_committor=(?P<committor>[0-9.eE+-]+); "
    r"unresolved_rate=(?P<rate>[0-9.eE+-]+)"
)
BASIN_FRONTIER_BOUNDARY_RE = re.compile(
    r"Basin frontier boundary absorbed (?P<count>\d+) states; "
    r"boundary_committor=(?P<committor>[0-9.eE+-]+); "
    r"boundary_rate=(?P<rate>[0-9.eE+-]+)"
)
BASIN_TRACE_RE = re.compile(
    r"Basin exploration trace order=(?P<order>[0-9,]*); "
    r"queue=(?P<queue>[0-9,]*); "
    r"guidance=(?P<guidance>[0-9:.,eE+-]*)"
    r"(?:; closed_events=(?P<closed_events>[^;]*))?"
    r"(?:; closed_processes=(?P<closed_processes>[^;]*))?"
    r"(?:; closed_guidance=(?P<closed_guidance>[0-9:.,eE+-]*))?"
)
KINETIC_GUARD_COMMITTOR_TOL = 1.0e-12
# The AMSEL event_completeness certificate exposes `needs_more_search` as its
# authoritative gate (logical OR of confidence_gap > 0 and missing_mass_gap >
# 0). The kinetic guard reads that field directly instead of duplicating the
# tolerance here. Tune via pyKMC's [BASIN] amsel_completeness_* knobs.
PROCESS_COVERAGE_RE = re.compile(
    r"AMSEL process coverage env=(?P<env>[^;]+); "
    r"attempts=(?P<attempts>\d+); "
    r"observations=(?P<observations>\d+); "
    r"unique_processes=(?P<unique>\d+); "
    r"singleton_processes=(?P<singletons>\d+); "
    r"missing_process_mass=(?P<missing_process>[0-9.eE+-]+); "
    r"missing_rate_mass=(?P<missing_rate>[0-9.eE+-]+); "
    r"needs_more_search=(?P<needs_more>True|False)"
)
STEP_RE = re.compile(r"^Step\s*:\s*(?P<step>\d+)\s*$")
TRIAL_FIELDS = [
    "case",
    "selector",
    "trial",
    "seed",
    "recombined",
    "t_recombination_s",
    "censored_time_s",
    "kmc_steps",
    "cpu_time_s",
    "wall_time_s",
    "total_cpu_time_s",
    "total_wall_time_s",
    "detector_reason",
    "event_discovery_status",
    "event_searches",
    "final_noncrystal_atoms",
    "min_noncrystal_atoms",
    "trajectory_recombination_frame",
    "failed_refinements",
    "failed_refinement_committor",
    "usable_resolved_committor",
    "kinetic_claim_ok",
    "coverage_envs_observed",
    "coverage_total_attempts",
    "coverage_total_observations",
    "coverage_max_missing_process_mass",
    "coverage_max_missing_rate_mass",
    "coverage_needs_more_search",
    "output_dir",
]
SURVIVAL_FIELDS = ["selector", "time_s", "n_at_risk", "n_events", "survival"]
BASIN_CONFIDENCE_FIELDS = [
    "case",
    "selector",
    "trial",
    "seed",
    "step",
    "frontier_states",
    "frontier_committor",
    "frontier_rate",
    "skipped_absorbing_exits",
    "skipped_absorbing_committor",
    "skipped_absorbing_rate",
]
BASIN_TRACE_FIELDS = [
    "case",
    "selector",
    "trial",
    "seed",
    "step",
    "order",
    "queue",
    "guidance",
    "order_count",
    "queue_count",
    "top_queue_state",
    "top_queue_guidance",
    "closed_events",
    "closed_event_families",
    "closed_event_family_count",
    "closed_processes",
    "closed_process_signatures",
    "closed_process_signature_count",
    "closed_process_singleton_count",
    "process_completeness_observations",
    "process_completeness_unique",
    "process_missing_mass_estimate",
    "closed_guidance",
    "closed_guidance_sum",
    "closed_top_guidance",
]


def detect_recombination_from_log(text: str) -> dict[str, object]:
    if CRYSTAL_TERMINATION_MARKER in text:
        return {
            "recombined": True,
            "detector_reason": "all-crystal-environments",
        }
    if ZERO_EVENT_DISCOVERY_MARKER in text:
        return {
            "recombined": False,
            "detector_reason": "zero-event-discovery",
        }
    return {
        "recombined": False,
        "detector_reason": "censored",
    }


def event_discovery_status_from_log(text: str) -> str:
    if ZERO_EVENT_DISCOVERY_MARKER in text:
        return "zero-events"
    return "not-zero-event"


def trajectory_noncrystal_counts(path: Path) -> list[int]:
    if not path.exists():
        return []
    try:
        from ase.io import iread
        from pykmc import AtomicEnvironment, NeighborsList, System
    except ImportError:
        return []

    settings = atomic_environment_settings(path.with_name("input.in"))
    counts: list[int] = []
    for atoms in iread(path, index=":"):
        system = System(
            types=atoms.get_chemical_symbols(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
        )
        system.update_positions(system.positions)
        neighbors = NeighborsList(system, settings["rnei"], settings["rcut"])
        environment = AtomicEnvironment(
            settings["style"],
            neighbors.neighbors_list["rnei"],
            neighbors.neighbors_list.get("rcut"),
            settings["neighbors_add"],
        )
        counts.append(
            sum(env != "crystal" for env in environment.atomic_environment_list)
        )
    return counts


def atomic_environment_settings(input_path: Path) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "style": "cna/graph",
        "rnei": 3.0,
        "rcut": 6.5,
        "neighbors_add": 0,
    }
    if not input_path.exists():
        return settings
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(input_path)
    section = _optional_section(config, "AtomicEnvironment")
    if section is None:
        return settings
    settings["style"] = config[section].get("style", str(settings["style"]))
    settings["rnei"] = config[section].getfloat("rnei", fallback=float(settings["rnei"]))
    settings["rcut"] = config[section].getfloat("rcut", fallback=float(settings["rcut"]))
    settings["neighbors_add"] = config[section].getint(
        "neighbors_add",
        fallback=int(settings["neighbors_add"]),
    )
    return settings


def trajectory_recombination_summary(
    noncrystal_counts: list[int],
    output_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "final_noncrystal_atoms": None,
        "min_noncrystal_atoms": None,
        "trajectory_recombination_frame": None,
        "trajectory_recombination_time_s": None,
    }
    if not noncrystal_counts:
        return summary

    summary["final_noncrystal_atoms"] = int(noncrystal_counts[-1])
    summary["min_noncrystal_atoms"] = int(min(noncrystal_counts))
    for frame_index, noncrystal_count in enumerate(noncrystal_counts):
        if int(noncrystal_count) != 0:
            continue
        summary["trajectory_recombination_frame"] = int(frame_index)
        if frame_index == 0:
            summary["trajectory_recombination_time_s"] = 0.0
        elif output_rows:
            output_index = min(frame_index - 1, len(output_rows) - 1)
            summary["trajectory_recombination_time_s"] = output_rows[output_index][
                "time_s"
            ]
        else:
            summary["trajectory_recombination_time_s"] = 0.0
        break
    return summary


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
                "cpu_time_s": float(parts[8]) if len(parts) > 8 else None,
                "wall_time_s": float(parts[9]) if len(parts) > 9 else None,
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
    trajectory_summary = trajectory_recombination_summary(
        trajectory_noncrystal_counts(output_dir / "trajkmc.xyz"),
        output_rows,
    )
    final_time = output_rows[-1]["time_s"] if output_rows else 0.0
    total_cpu_time = _sum_optional_float(row["cpu_time_s"] for row in output_rows)
    total_wall_time = _sum_optional_float(row["wall_time_s"] for row in output_rows)
    recombined = bool(detector["recombined"])
    detector_reason = detector["detector_reason"]
    if (
        not recombined
        and trajectory_summary["trajectory_recombination_frame"] is not None
    ):
        recombined = True
        detector_reason = "trajectory-all-crystal"
        final_time = float(trajectory_summary["trajectory_recombination_time_s"])
    row = {
        "case": case,
        "selector": selector,
        "trial": int(trial),
        "seed": int(seed),
        "recombined": recombined,
        "t_recombination_s": final_time if recombined else None,
        "censored_time_s": None if recombined else final_time,
        "kmc_steps": len(output_rows),
        "cpu_time_s": output_rows[-1]["cpu_time_s"] if output_rows else None,
        "wall_time_s": output_rows[-1]["wall_time_s"] if output_rows else None,
        "total_cpu_time_s": total_cpu_time,
        "total_wall_time_s": total_wall_time,
        "detector_reason": detector_reason,
        "event_discovery_status": event_discovery_status_from_log(log_text),
        "event_searches": None,
        "final_noncrystal_atoms": trajectory_summary["final_noncrystal_atoms"],
        "min_noncrystal_atoms": trajectory_summary["min_noncrystal_atoms"],
        "trajectory_recombination_frame": trajectory_summary[
            "trajectory_recombination_frame"
        ],
        "coverage_resampling_disabled": coverage_resampling_disabled_from_input(
            output_dir / "input.in"
        ),
        "output_dir": str(output_dir),
    }
    return apply_kinetic_guard(row, diagnostics=None, log_text=log_text)


def _sum_optional_float(values) -> float | None:
    total = 0.0
    seen = False
    for value in values:
        if value is None:
            continue
        total += float(value)
        seen = True
    if not seen:
        return None
    return total


def apply_kinetic_guard(
    row: dict[str, Any],
    *,
    diagnostics: dict[str, Any] | None,
    log_text: str,
) -> dict[str, Any]:
    guarded = dict(row)
    guarded["failed_refinements"] = int(guarded.get("failed_refinements") or 0)
    guarded["failed_refinement_committor"] = float(
        guarded.get("failed_refinement_committor") or 0.0
    )
    guarded["usable_resolved_committor"] = guarded.get("usable_resolved_committor")
    guarded["kinetic_claim_ok"] = bool(guarded.get("kinetic_claim_ok", True))

    if diagnostics is not None:
        failed_committor = float(diagnostics.get("failed_refinement_committor") or 0.0)
        guarded["failed_refinement_committor"] = failed_committor
        guarded["usable_resolved_committor"] = diagnostics.get(
            "usable_resolved_committor"
        )
        guarded["failed_refinements"] = int(
            diagnostics.get("failed_refinements") or guarded["failed_refinements"]
        )
        if (
            failed_committor > KINETIC_GUARD_COMMITTOR_TOL
            or diagnostics.get("kinetic_claim_ok") is False
        ):
            guarded["kinetic_claim_ok"] = False

    if "Basin fails with error" in log_text and "EVENT_NOT_FOUND" in log_text:
        guarded["failed_refinements"] += 1
        guarded["kinetic_claim_ok"] = False
    for match in BASIN_REFINEMENT_BUDGET_RE.finditer(log_text):
        committor = float(match.group("committor"))
        guarded["failed_refinements"] += int(match.group("count"))
        guarded["failed_refinement_committor"] += committor
        if committor > KINETIC_GUARD_COMMITTOR_TOL:
            guarded["kinetic_claim_ok"] = False
    for match in BASIN_FRONTIER_BUDGET_RE.finditer(log_text):
        committor = float(match.group("committor"))
        guarded["failed_refinements"] += int(match.group("count"))
        guarded["failed_refinement_committor"] += committor
        if committor > KINETIC_GUARD_COMMITTOR_TOL:
            guarded["kinetic_claim_ok"] = False
    if guarded["failed_refinement_committor"] > KINETIC_GUARD_COMMITTOR_TOL:
        guarded["kinetic_claim_ok"] = False
    if not guarded.get("recombined", False) and int(guarded.get("kmc_steps") or 0) == 0:
        guarded["kinetic_claim_ok"] = False

    # AMSEL process-coverage certificates expose the controller's
    # rate-material resampling gate. A trial is not suitable for kinetic claims
    # while any logged certificate still needs more search.
    latest_coverage_by_env: dict[str, dict[str, object]] = {}
    for match in PROCESS_COVERAGE_RE.finditer(log_text):
        latest_coverage_by_env[match.group("env")] = {
            "attempts": int(match.group("attempts")),
            "observations": int(match.group("observations")),
            "missing_process": float(match.group("missing_process")),
            "missing_rate": float(match.group("missing_rate")),
            "needs_more": match.group("needs_more") == "True",
        }
    if latest_coverage_by_env:
        coverage_max_missing = max(
            float(item["missing_process"])
            for item in latest_coverage_by_env.values()
        )
        coverage_max_missing_rate = max(
            float(item["missing_rate"])
            for item in latest_coverage_by_env.values()
        )
        coverage_needs_more = any(
            bool(item["needs_more"]) for item in latest_coverage_by_env.values()
        )
        guarded["coverage_envs_observed"] = len(latest_coverage_by_env)
        guarded["coverage_total_attempts"] = sum(
            int(item["attempts"]) for item in latest_coverage_by_env.values()
        )
        guarded["coverage_total_observations"] = sum(
            int(item["observations"]) for item in latest_coverage_by_env.values()
        )
        guarded["coverage_max_missing_process_mass"] = coverage_max_missing
        guarded["coverage_max_missing_rate_mass"] = coverage_max_missing_rate
        guarded["coverage_needs_more_search"] = coverage_needs_more
        selector = str(guarded.get("selector", ""))
        if (
            coverage_needs_more
            and not selector.startswith("legacy")
            and not bool(guarded.get("coverage_resampling_disabled", False))
        ):
            guarded["kinetic_claim_ok"] = False
    return guarded


def coverage_resampling_disabled_from_input(input_path: Path) -> bool:
    if not input_path.exists():
        return False
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(input_path)
    section = _optional_section(config, "Control")
    if section is None:
        return False
    return config[section].getboolean("disable_coverage_resampling", fallback=False)


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


def basin_confidence_rows_from_log(
    *,
    case: str,
    selector: str,
    trial: int,
    seed: int,
    log_text: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_step: int | None = None
    row_by_step: dict[int, dict[str, Any]] = {}

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        step_match = STEP_RE.match(line)
        if step_match is not None:
            current_step = int(step_match.group("step"))
            continue

        frontier_match = BASIN_FRONTIER_BUDGET_RE.search(line)
        if frontier_match is not None:
            step = int(current_step or 0)
            row = _empty_basin_confidence_row(
                case=case,
                selector=selector,
                trial=trial,
                seed=seed,
                step=step,
            )
            row["frontier_states"] = int(frontier_match.group("count"))
            row["frontier_committor"] = float(frontier_match.group("committor"))
            row["frontier_rate"] = float(frontier_match.group("rate"))
            row_by_step[step] = row
            rows.append(row)
            continue

        boundary_match = BASIN_FRONTIER_BOUNDARY_RE.search(line)
        if boundary_match is not None:
            step = int(current_step or 0)
            row = _empty_basin_confidence_row(
                case=case,
                selector=selector,
                trial=trial,
                seed=seed,
                step=step,
            )
            row["frontier_states"] = int(boundary_match.group("count"))
            row["frontier_committor"] = float(boundary_match.group("committor"))
            row["frontier_rate"] = float(boundary_match.group("rate"))
            row_by_step[step] = row
            rows.append(row)
            continue

        absorbing_match = BASIN_REFINEMENT_BUDGET_RE.search(line)
        if absorbing_match is not None:
            step = int(current_step or 0)
            row = row_by_step.get(step)
            if row is None:
                row = _empty_basin_confidence_row(
                    case=case,
                    selector=selector,
                    trial=trial,
                    seed=seed,
                    step=step,
                )
                row_by_step[step] = row
                rows.append(row)
            row["skipped_absorbing_exits"] = int(absorbing_match.group("count"))
            row["skipped_absorbing_committor"] = float(
                absorbing_match.group("committor")
            )
            row["skipped_absorbing_rate"] = float(absorbing_match.group("rate"))

    return rows


def collect_basin_confidence_rows(
    commands: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in commands:
        log_path = Path(command["workdir"]) / "pykmc.log"
        if not log_path.exists():
            continue
        rows.extend(
            basin_confidence_rows_from_log(
                case=str(command["case"]),
                selector=str(command["priority"]),
                trial=int(command["trial"]),
                seed=int(command["seed"]),
                log_text=log_path.read_text(),
            )
        )
    return rows


def basin_trace_rows_from_log(
    *,
    case: str,
    selector: str,
    trial: int,
    seed: int,
    log_text: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_step: int | None = None
    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        step_match = STEP_RE.match(line)
        if step_match is not None:
            current_step = int(step_match.group("step"))
            continue

        trace_match = BASIN_TRACE_RE.search(line)
        if trace_match is None:
            continue

        order = _parse_state_list(trace_match.group("order"))
        queue = _parse_state_list(trace_match.group("queue"))
        guidance_items, guidance_by_state = _parse_guidance(
            trace_match.group("guidance")
        )
        closed_event_items, closed_event_families = _parse_closed_events(
            trace_match.group("closed_events") or ""
        )
        (
            closed_process_items,
            closed_process_signatures,
            closed_process_singleton_count,
            closed_process_counts,
        ) = _parse_closed_processes(trace_match.group("closed_processes") or "")
        process_completeness = _process_completeness(closed_process_counts)
        closed_guidance_items, closed_guidance_by_state = _parse_guidance(
            trace_match.group("closed_guidance") or ""
        )
        closed_guidance_values = list(closed_guidance_by_state.values())
        top_queue_state = queue[0] if queue else None
        top_queue_guidance = (
            guidance_by_state.get(top_queue_state)
            if top_queue_state is not None
            else None
        )
        rows.append(
            {
                "case": case,
                "selector": selector,
                "trial": int(trial),
                "seed": int(seed),
                "step": int(current_step or 0),
                "order": " ".join(str(state) for state in order),
                "queue": " ".join(str(state) for state in queue),
                "guidance": " ".join(guidance_items),
                "order_count": len(order),
                "queue_count": len(queue),
                "top_queue_state": top_queue_state,
                "top_queue_guidance": top_queue_guidance,
                "closed_events": " ".join(closed_event_items),
                "closed_event_families": " ".join(
                    str(event) for event in closed_event_families
                ),
                "closed_event_family_count": len(closed_event_families),
                "closed_processes": " ".join(closed_process_items),
                "closed_process_signatures": " ".join(closed_process_signatures),
                "closed_process_signature_count": len(closed_process_signatures),
                "closed_process_singleton_count": closed_process_singleton_count,
                **process_completeness,
                "closed_guidance": " ".join(closed_guidance_items),
                "closed_guidance_sum": float(sum(closed_guidance_values)),
                "closed_top_guidance": float(max(closed_guidance_values, default=0.0)),
            }
        )
    return rows


def collect_basin_trace_rows(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for command in commands:
        log_path = Path(command["workdir"]) / "pykmc.log"
        if not log_path.exists():
            continue
        rows.extend(
            basin_trace_rows_from_log(
                case=str(command["case"]),
                selector=str(command["priority"]),
                trial=int(command["trial"]),
                seed=int(command["seed"]),
                log_text=log_path.read_text(),
            )
        )
    return rows


def _parse_state_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def _parse_guidance(text: str) -> tuple[list[str], dict[int, float]]:
    items = [item for item in text.split(",") if item]
    guidance: dict[int, float] = {}
    for item in items:
        state_text, _, value_text = item.partition(":")
        if not state_text or not value_text:
            continue
        guidance[int(state_text)] = float(value_text)
    return items, guidance


def _parse_closed_events(text: str) -> tuple[list[str], list[int]]:
    items = [item for item in text.split(",") if item]
    families: list[int] = []
    seen: set[int] = set()
    for item in items:
        _, _, event_text = item.partition(":")
        if not event_text or event_text == "NA":
            continue
        event = int(event_text)
        if event in seen:
            continue
        seen.add(event)
        families.append(event)
    return items, families


def _parse_closed_processes(
    text: str,
) -> tuple[list[str], list[str], int, Counter[str]]:
    items = [item for item in text.split(",") if item]
    signatures: list[str] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for item in items:
        _, _, signature = item.partition(":")
        if not signature or signature == "NA":
            continue
        counts[signature] += 1
        if signature in seen:
            continue
        seen.add(signature)
        signatures.append(signature)
    singleton_count = sum(1 for signature in signatures if counts[signature] == 1)
    return items, signatures, singleton_count, counts


def _process_completeness(process_counts: Counter[str]) -> dict[str, Any]:
    if _amsel is not None and hasattr(_amsel, "event_completeness"):
        certificate = _amsel.event_completeness(
            process_counts=dict(process_counts),
        )
        return {
            "process_completeness_observations": certificate.observations,
            "process_completeness_unique": certificate.unique_processes,
            "process_missing_mass_estimate": certificate.unseen_process_probability,
        }

    observations = sum(process_counts.values())
    unique = len(process_counts)
    singleton = sum(1 for count in process_counts.values() if count == 1)
    missing_mass = float(singleton) / float(observations) if observations else 1.0
    return {
        "process_completeness_observations": observations,
        "process_completeness_unique": unique,
        "process_missing_mass_estimate": missing_mass,
    }


def _empty_basin_confidence_row(
    *,
    case: str,
    selector: str,
    trial: int,
    seed: int,
    step: int,
) -> dict[str, Any]:
    return {
        "case": case,
        "selector": selector,
        "trial": int(trial),
        "seed": int(seed),
        "step": int(step),
        "frontier_states": 0,
        "frontier_committor": 0.0,
        "frontier_rate": 0.0,
        "skipped_absorbing_exits": 0,
        "skipped_absorbing_committor": 0.0,
        "skipped_absorbing_rate": 0.0,
    }


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
    event_searches: int | None,
    refine_thr: float | None,
    priorities: list[str],
    trials: int,
    seed: int,
    max_steps: int,
    work_budget: str | None,
    trial_timeout_s: float | None,
    basin_energy_thr: float | None,
    basin_max_expansions: int | None,
    basin_max_closed_states: int | None,
    basin_max_absorbing_refinements: int | None,
    basin_frontier_committor_tol: float | None,
    amsel_selector: str,
    amsel_exploration_priority: str,
    amsel_duplicate_family_penalty: float = 1.0,
    amsel_min_guidance: float = 0.0,
    disable_coverage_resampling: bool = False,
    basin_search_registry_path: str | None = None,
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
            template_dir=template_input.parent,
            initial_config=initial_config,
            reference_table=reference_table,
            visited_environments=visited_environments,
            max_steps=max_steps,
            event_searches=event_searches,
            refine_thr=refine_thr,
            priority=str(item["priority"]),
            partn_path=partn_path,
            seed=int(item["seed"]),
            basin_energy_thr=basin_energy_thr,
            basin_max_expansions=basin_max_expansions,
            basin_max_closed_states=basin_max_closed_states,
            basin_max_absorbing_refinements=basin_max_absorbing_refinements,
            basin_frontier_committor_tol=basin_frontier_committor_tol,
            amsel_selector=amsel_selector,
            amsel_exploration_priority=amsel_exploration_priority,
            amsel_duplicate_family_penalty=amsel_duplicate_family_penalty,
            amsel_min_guidance=amsel_min_guidance,
            disable_coverage_resampling=disable_coverage_resampling,
            basin_search_registry_path=basin_search_registry_path,
        )
        commands.append(
            {
                "case": case,
                "priority": item["priority"],
                "trial": item["trial"],
                "seed": item["seed"],
                "max_steps": int(max_steps),
                "work_budget": work_budget,
                "event_searches": event_searches,
                "trial_timeout_s": trial_timeout_s,
                "basin_energy_thr": basin_energy_thr,
                "basin_max_expansions": basin_max_expansions,
                "basin_max_closed_states": basin_max_closed_states,
                "basin_max_absorbing_refinements": basin_max_absorbing_refinements,
                "basin_frontier_committor_tol": basin_frontier_committor_tol,
                "amsel_selector": amsel_selector,
                "amsel_exploration_priority": amsel_exploration_priority,
                "amsel_duplicate_family_penalty": amsel_duplicate_family_penalty,
                "amsel_min_guidance": amsel_min_guidance,
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
    template_dir: Path,
    initial_config: Path,
    reference_table: Path | None,
    visited_environments: Path | None,
    max_steps: int,
    event_searches: int | None,
    refine_thr: float | None,
    priority: str,
    partn_path: Path,
    seed: int,
    basin_energy_thr: float | None,
    basin_max_expansions: int | None,
    basin_max_closed_states: int | None,
    basin_max_absorbing_refinements: int | None,
    basin_frontier_committor_tol: float | None,
    amsel_selector: str,
    amsel_exploration_priority: str,
    amsel_duplicate_family_penalty: float = 1.0,
    amsel_min_guidance: float = 0.0,
    disable_coverage_resampling: bool = False,
    basin_search_registry_path: str | None = None,
) -> str:
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(template_text)
    control = _section(config, "Control")
    partn = _section(config, "pARTn")
    basin = _section(config, "BASIN")

    config[control]["initial_config"] = str(initial_config)
    config[control]["n_steps"] = str(int(max_steps))
    config[control]["random_seed"] = str(int(seed))
    config[control]["basin"] = "True"
    if disable_coverage_resampling or priority == "legacy":
        config[control]["disable_coverage_resampling"] = "True"
    if basin_search_registry_path is not None:
        config[control]["basin_search_registry_path"] = str(
            basin_search_registry_path
        )
    if refine_thr is not None:
        config[control]["refine_thr"] = str(float(refine_thr))
    if reference_table is not None:
        config[control]["reference_table"] = str(reference_table)
    if visited_environments is not None:
        config[control]["visited_environments"] = str(visited_environments)
    if event_searches is not None:
        event_search = _section(config, "EventSearch")
        config[event_search]["nsearch"] = str(int(event_searches))
    config[partn]["path_artnso"] = str(partn_path)
    config[partn]["zseed"] = str(int(seed))
    config[basin]["exploration_priority"] = _exploration_priority_for_priority(
        priority=priority,
        amsel_exploration_priority=amsel_exploration_priority,
    )
    config[basin]["exploration_duplicate_family_penalty"] = str(
        float(amsel_duplicate_family_penalty)
    )
    config[basin]["exploration_min_guidance"] = str(float(amsel_min_guidance))
    config[basin]["selector"] = _selector_for_priority(
        priority=priority,
        amsel_selector=amsel_selector,
    )
    if basin_energy_thr is not None:
        config[basin]["energy_thr"] = str(float(basin_energy_thr))
    if basin_max_expansions is not None:
        config[basin]["max_expansions"] = str(int(basin_max_expansions))
    if basin_max_closed_states is not None:
        config[basin]["max_closed_states"] = str(int(basin_max_closed_states))
    if basin_max_absorbing_refinements is not None:
        config[basin]["max_absorbing_refinements"] = str(
            int(basin_max_absorbing_refinements)
        )
    if basin_frontier_committor_tol is not None:
        config[basin]["frontier_committor_tol"] = str(
            float(basin_frontier_committor_tol)
        )
    absolutize_lammps_paths(config, template_dir=template_dir)

    from io import StringIO

    out = StringIO()
    config.write(out)
    return out.getvalue()


def write_trial_input(
    path: Path,
    *,
    template_text: str,
    template_dir: Path,
    initial_config: Path,
    reference_table: Path | None,
    visited_environments: Path | None,
    max_steps: int,
    event_searches: int | None,
    refine_thr: float | None,
    priority: str,
    partn_path: Path,
    seed: int,
    basin_energy_thr: float | None,
    basin_max_expansions: int | None,
    basin_max_closed_states: int | None,
    basin_max_absorbing_refinements: int | None,
    basin_frontier_committor_tol: float | None,
    amsel_selector: str,
    amsel_exploration_priority: str,
    amsel_duplicate_family_penalty: float,
    amsel_min_guidance: float,
    disable_coverage_resampling: bool = False,
    basin_search_registry_path: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_trial_input(
            template_text=template_text,
            template_dir=template_dir,
            initial_config=initial_config,
            reference_table=reference_table,
            visited_environments=visited_environments,
            max_steps=max_steps,
            event_searches=event_searches,
            refine_thr=refine_thr,
            priority=priority,
            partn_path=partn_path,
            seed=seed,
            basin_energy_thr=basin_energy_thr,
            basin_max_expansions=basin_max_expansions,
            basin_max_closed_states=basin_max_closed_states,
            basin_max_absorbing_refinements=basin_max_absorbing_refinements,
            basin_frontier_committor_tol=basin_frontier_committor_tol,
            amsel_selector=amsel_selector,
            amsel_exploration_priority=amsel_exploration_priority,
            amsel_duplicate_family_penalty=amsel_duplicate_family_penalty,
            amsel_min_guidance=amsel_min_guidance,
            disable_coverage_resampling=disable_coverage_resampling,
            basin_search_registry_path=basin_search_registry_path,
        )
    )


def _selector_for_priority(*, priority: str, amsel_selector: str) -> str:
    if priority == "legacy":
        return "legacy-fpta"
    if priority == "amsel":
        return amsel_selector
    raise ValueError(f"unknown priority: {priority}")


def _exploration_priority_for_priority(
    *,
    priority: str,
    amsel_exploration_priority: str,
) -> str:
    if priority == "legacy":
        return "legacy"
    if priority == "amsel":
        return amsel_exploration_priority
    raise ValueError(f"unknown priority: {priority}")


def absolutize_lammps_paths(
    config: configparser.ConfigParser,
    *,
    template_dir: Path,
) -> None:
    lammps = _optional_section(config, "Lammps")
    if lammps is None:
        return
    pair_coeff = config[lammps].get("pair_coeff")
    if pair_coeff is None:
        return
    tokens = []
    for token in pair_coeff.split():
        candidate = template_dir / token
        if not Path(token).is_absolute() and candidate.exists():
            tokens.append(str(candidate.resolve()))
        else:
            tokens.append(token)
    config[lammps]["pair_coeff"] = " ".join(tokens)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fieldnames} for row in rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def execute_trials(
    commands: list[dict[str, Any]],
    events_path: Path,
    *,
    trial_timeout_s: float | None = None,
) -> list[dict[str, Any]]:
    rows = []
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("w") as event_handle:
        for command in commands:
            workdir = Path(command["workdir"])
            try:
                result = subprocess.run(
                    command["command"],
                    cwd=workdir,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=trial_timeout_s,
                )
            except subprocess.TimeoutExpired as error:
                (workdir / "harness.log").write_text(_timeout_output(error))
                event_handle.write(
                    json.dumps(
                        {
                            "case": command["case"],
                            "priority": command["priority"],
                            "trial": command["trial"],
                            "seed": command["seed"],
                            "returncode": None,
                            "timed_out": True,
                            "timeout_s": trial_timeout_s,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                if (workdir / "pykmc.out").exists() and (
                    workdir / "pykmc.log"
                ).exists():
                    row = trial_row_from_outputs(
                        case=str(command["case"]),
                        selector=str(command["priority"]),
                        trial=int(command["trial"]),
                        seed=int(command["seed"]),
                        output_dir=workdir,
                    )
                    row["detector_reason"] = f"timeout-{trial_timeout_s}s"
                    row["event_searches"] = command.get("event_searches")
                    row["kinetic_claim_ok"] = False
                    rows.append(row)
                else:
                    rows.append(
                        apply_kinetic_guard(
                            {
                                "case": command["case"],
                                "selector": command["priority"],
                                "trial": int(command["trial"]),
                                "seed": int(command["seed"]),
                                "recombined": False,
                                "t_recombination_s": None,
                                "censored_time_s": 0.0,
                                "kmc_steps": 0,
                                "cpu_time_s": None,
                                "wall_time_s": None,
                                "total_cpu_time_s": None,
                                "total_wall_time_s": None,
                                "detector_reason": f"timeout-{trial_timeout_s}s",
                                "event_discovery_status": "unknown",
                                "event_searches": command.get("event_searches"),
                                "final_noncrystal_atoms": None,
                                "min_noncrystal_atoms": None,
                                "trajectory_recombination_frame": None,
                                "kinetic_claim_ok": False,
                                "output_dir": str(workdir),
                            },
                            diagnostics=None,
                            log_text="",
                        )
                    )
                continue
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
                rows[-1]["event_searches"] = command.get("event_searches")
            else:
                rows.append(
                    apply_kinetic_guard(
                        {
                            "case": command["case"],
                            "selector": command["priority"],
                            "trial": int(command["trial"]),
                            "seed": int(command["seed"]),
                            "recombined": False,
                            "t_recombination_s": None,
                            "censored_time_s": 0.0,
                            "kmc_steps": 0,
                            "cpu_time_s": None,
                            "wall_time_s": None,
                            "total_cpu_time_s": None,
                            "total_wall_time_s": None,
                            "detector_reason": f"returncode-{result.returncode}",
                            "event_discovery_status": "unknown",
                            "event_searches": command.get("event_searches"),
                            "kinetic_claim_ok": False,
                            "output_dir": str(workdir),
                        },
                        diagnostics=None,
                        log_text="",
                    )
                )
    return rows


def _timeout_output(error: subprocess.TimeoutExpired) -> str:
    output = error.output or ""
    if isinstance(output, bytes):
        return output.decode(errors="replace")
    return str(output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("ni-vac-sia", "cu-vac-sia", "generic-defect"),
        required=True,
    )
    parser.add_argument("--template-input", type=Path)
    parser.add_argument("--initial-config", type=Path)
    parser.add_argument("--reference-table", type=Path)
    parser.add_argument("--visited-environments", type=Path)
    parser.add_argument("--allow-preloaded-catalog", action="store_true")
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
    parser.add_argument("--event-searches", type=int)
    parser.add_argument("--refine-thr", type=float)
    parser.add_argument("--trial-timeout-s", type=float)
    parser.add_argument("--basin-energy-thr", type=float)
    parser.add_argument("--basin-max-expansions", type=int)
    parser.add_argument("--basin-max-closed-states", type=int)
    parser.add_argument("--basin-max-absorbing-refinements", type=int)
    parser.add_argument("--basin-frontier-committor-tol", type=float)
    parser.add_argument(
        "--amsel-selector",
        choices=("amsel-sampled", "amsel-mean", "amsel-adaptive"),
        default="amsel-adaptive",
    )
    parser.add_argument(
        "--amsel-exploration-priority",
        choices=("amsel", "amsel-diverse"),
        default="amsel",
    )
    parser.add_argument("--amsel-duplicate-family-penalty", type=float, default=1.0)
    parser.add_argument("--amsel-min-guidance", type=float, default=0.0)
    parser.add_argument(
        "--disable-coverage-resampling",
        action="store_true",
        help=(
            "Skip the AMSEL process-coverage resampling pass at every KMC "
            "step. Each trial input.in receives "
            "[Control] disable_coverage_resampling=True so per-step wall "
            "time is bounded for paired benchmark sweeps."
        ),
    )
    parser.add_argument(
        "--basin-search-registry-path",
        type=str,
        default=None,
        help=(
            "Path to a JSONL file shared across trials in this run. "
            "When set, every trial input.in receives "
            "[Control] basin_search_registry_path so AMSEL "
            "BasinSearchRegistry deduplicates ARTn refinement claims by "
            "saddle-displacement mode signature."
        ),
    )
    parser.add_argument("--work-budget")
    parser.add_argument("--mpi-ranks", type=int, default=8)
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if (
        args.reference_table is not None or args.visited_environments is not None
    ) and not args.allow_preloaded_catalog:
        parser.error(
            "--reference-table and --visited-environments require "
            "--allow-preloaded-catalog"
        )
    defaults = {} if args.case == "generic-defect" else case_defaults(args.case)
    template_input = args.template_input or defaults.get("template_input")
    initial_config = args.initial_config or defaults.get("initial_config")
    if template_input is None or initial_config is None:
        parser.error("--template-input and --initial-config are required for generic-defect")

    commands = trial_commands(
        out=args.out,
        case=args.case,
        template_input=template_input,
        initial_config=initial_config,
        reference_table=args.reference_table,
        visited_environments=args.visited_environments,
        partn_path=args.partn_path,
        event_searches=args.event_searches,
        refine_thr=args.refine_thr,
        priorities=args.priority,
        trials=args.trials,
        seed=args.seed,
        max_steps=args.max_steps,
        work_budget=args.work_budget,
        trial_timeout_s=args.trial_timeout_s,
        basin_energy_thr=args.basin_energy_thr,
        basin_max_expansions=args.basin_max_expansions,
        basin_max_closed_states=args.basin_max_closed_states,
        basin_max_absorbing_refinements=args.basin_max_absorbing_refinements,
        basin_frontier_committor_tol=args.basin_frontier_committor_tol,
        amsel_selector=args.amsel_selector,
        amsel_exploration_priority=args.amsel_exploration_priority,
        amsel_duplicate_family_penalty=args.amsel_duplicate_family_penalty,
        amsel_min_guidance=args.amsel_min_guidance,
        disable_coverage_resampling=args.disable_coverage_resampling,
        basin_search_registry_path=args.basin_search_registry_path,
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
        "allow_preloaded_catalog": bool(args.allow_preloaded_catalog),
        "partn_path": str(args.partn_path),
        "priorities": args.priority,
        "trials": args.trials,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "event_searches": args.event_searches,
        "refine_thr": args.refine_thr,
        "trial_timeout_s": args.trial_timeout_s,
        "basin_energy_thr": args.basin_energy_thr,
        "basin_max_expansions": args.basin_max_expansions,
        "basin_max_closed_states": args.basin_max_closed_states,
        "basin_max_absorbing_refinements": args.basin_max_absorbing_refinements,
        "basin_frontier_committor_tol": args.basin_frontier_committor_tol,
        "amsel_selector": args.amsel_selector,
        "amsel_exploration_priority": args.amsel_exploration_priority,
        "amsel_duplicate_family_penalty": args.amsel_duplicate_family_penalty,
        "amsel_min_guidance": args.amsel_min_guidance,
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
        write_csv(
            args.out / "basin_confidence.csv",
            [],
            BASIN_CONFIDENCE_FIELDS,
        )
        write_csv(args.out / "basin_trace.csv", [], BASIN_TRACE_FIELDS)
        events_path.write_text("")
        return 0

    trial_rows = execute_trials(
        commands,
        events_path,
        trial_timeout_s=args.trial_timeout_s,
    )
    write_csv(args.out / "trials.csv", trial_rows, TRIAL_FIELDS)
    write_csv(args.out / "survival.csv", survival_rows(trial_rows), SURVIVAL_FIELDS)
    write_csv(
        args.out / "basin_confidence.csv",
        collect_basin_confidence_rows(commands),
        BASIN_CONFIDENCE_FIELDS,
    )
    write_csv(
        args.out / "basin_trace.csv",
        collect_basin_trace_rows(commands),
        BASIN_TRACE_FIELDS,
    )
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


def _optional_section(config: configparser.ConfigParser, name: str) -> str | None:
    for section in config.sections():
        if section.lower() == name.lower():
            return section
    return None


if __name__ == "__main__":
    raise SystemExit(main())
