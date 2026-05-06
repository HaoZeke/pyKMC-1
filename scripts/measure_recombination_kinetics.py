#!/usr/bin/env python
"""Measure vacancy-SIA recombination kinetics for PyKMC AMSEL comparisons."""
from __future__ import annotations

from pathlib import Path
from typing import Any

CRYSTAL_TERMINATION_MARKER = "Only atoms with cristalline environment"


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
        survival = 1.0
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
            survival *= 1.0 - (n_events / n_at_risk)
            rows.append(
                {
                    "selector": selector,
                    "time_s": event_time,
                    "n_at_risk": int(n_at_risk),
                    "n_events": int(n_events),
                    "survival": survival,
                }
            )
    return rows


def _trial_time(trial: dict[str, Any]) -> float:
    if trial.get("recombined"):
        return float(trial["t_recombination_s"])
    return float(trial["censored_time_s"])
