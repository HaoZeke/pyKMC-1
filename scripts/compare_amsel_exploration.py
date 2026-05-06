#!/usr/bin/env python
"""Compare fixed-work AMSEL and legacy basin exploration payloads."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def summarize_payload(payload: dict[str, Any], *, queue_head: int = 12) -> dict[str, Any]:
    exploration = payload.get("exploration") or {}
    refinement = payload.get("refinement") or {}
    channel_summary = payload.get("channel_summary") or {}
    transient_states = payload.get("transient_states") or []

    closed_states = [int(state) for state in exploration.get("closed_states", [])]
    expanded_states = [int(state) for state in exploration.get("expanded_states", [])]
    states_to_explore = [
        int(state) for state in exploration.get("states_to_explore", [])[:queue_head]
    ]
    resolved_committor = _float_or_zero(channel_summary.get("committor_sum"))
    resolved_rate = _float_or_zero(channel_summary.get("rate_sum"))
    frontier_committor = sum(
        _float_or_zero(row.get("hit_committor"))
        for row in transient_states
        if row.get("ok", True)
    )
    channels = payload.get("channels") or []
    processes = _processes(channels)
    if resolved_rate == 0.0:
        resolved_rate = sum(float(process["rate_sum"]) for process in processes)
    failed_refinement_committor = sum(
        _float_or_zero(channel.get("committor"))
        for channel in channels
        if _is_failed_refinement_channel(channel)
    )
    failed_refinement_rate = sum(
        _float_or_zero(channel.get("rate"))
        for channel in channels
        if _is_failed_refinement_channel(channel)
    )
    usable_resolved_committor = max(0.0, resolved_committor - failed_refinement_committor)
    usable_resolved_rate = max(0.0, resolved_rate - failed_refinement_rate)
    top_process = processes[0]["event_connexion"] if processes else None
    top_process_refinement_ok = None
    if top_process is not None:
        top_process_refinement_ok = not any(
            _is_failed_refinement_channel(channel)
            and _event_value(channel.get("event_connexion")) == top_process
            for channel in channels
        )
    accounted_committor = resolved_committor + frontier_committor
    kinetic_confidence = None
    if accounted_committor > 0.0:
        kinetic_confidence = resolved_committor / accounted_committor

    return {
        "case": payload.get("case"),
        "ok": bool(payload.get("ok", False)),
        "priority": exploration.get("priority"),
        "max_expansions": exploration.get("max_expansions"),
        "max_closed_states": exploration.get("max_closed_states"),
        "rate_source": refinement.get("rate_source"),
        "refinement_ok": refinement.get("ok"),
        "expanded_states": expanded_states,
        "closed_states": closed_states,
        "closed_nonentry_states": [state for state in closed_states if state != 0],
        "closed_count": len(closed_states),
        "resolved_committor": resolved_committor,
        "resolved_rate": resolved_rate,
        "failed_refinement_committor": failed_refinement_committor,
        "failed_refinement_rate": failed_refinement_rate,
        "usable_resolved_committor": usable_resolved_committor,
        "usable_resolved_rate": usable_resolved_rate,
        "frontier_committor": frontier_committor,
        "accounted_committor": accounted_committor,
        "kinetic_confidence": kinetic_confidence,
        "kinetic_claim_ok": refinement.get("ok") is True,
        "processes": processes,
        "top_process_event_connexion": top_process,
        "top_process_refinement_ok": top_process_refinement_ok,
        "queue_head": states_to_explore,
    }


def comparison_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [summarize_payload(payload) for payload in payloads]
    return {
        "ok": all(bool(row["ok"]) for row in rows),
        "rows": rows,
        "gain": _gain(rows),
    }


def load_payloads(paths: list[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in paths]


def write_csv(payload: dict[str, Any], out) -> None:
    fieldnames = [
        "priority",
        "ok",
        "rate_source",
        "refinement_ok",
        "closed_count",
        "closed_nonentry_states",
        "resolved_committor",
        "resolved_rate",
        "failed_refinement_committor",
        "failed_refinement_rate",
        "usable_resolved_committor",
        "usable_resolved_rate",
        "frontier_committor",
        "accounted_committor",
        "kinetic_confidence",
        "kinetic_claim_ok",
        "top_process_event_connexion",
        "top_process_refinement_ok",
        "queue_head",
    ]
    rows = [_csv_row(row, fieldnames=fieldnames) for row in payload["rows"]]
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _gain(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_priority = {row.get("priority"): row for row in rows}
    baseline = by_priority.get("legacy")
    challenger = by_priority.get("amsel")
    if baseline is None or challenger is None:
        return {"ok": False, "error": "comparison needs legacy and amsel rows"}

    baseline_resolved = float(baseline["resolved_committor"])
    challenger_resolved = float(challenger["resolved_committor"])
    baseline_confidence = baseline.get("kinetic_confidence")
    challenger_confidence = challenger.get("kinetic_confidence")
    ratio = None
    if baseline_resolved > 0.0:
        ratio = challenger_resolved / baseline_resolved

    closed_budget = None
    if baseline.get("closed_count") == challenger.get("closed_count"):
        closed_budget = int(baseline["closed_count"])

    return {
        "ok": True,
        "baseline_priority": "legacy",
        "challenger_priority": "amsel",
        "closed_budget": closed_budget,
        "resolved_committor_delta": challenger_resolved - baseline_resolved,
        "resolved_committor_ratio": ratio,
        "resolved_rate_delta": float(challenger["resolved_rate"])
        - float(baseline["resolved_rate"]),
        "usable_resolved_committor_delta": float(
            challenger["usable_resolved_committor"]
        )
        - float(baseline["usable_resolved_committor"]),
        "usable_resolved_rate_delta": float(challenger["usable_resolved_rate"])
        - float(baseline["usable_resolved_rate"]),
        "kinetic_confidence_delta": _delta(challenger_confidence, baseline_confidence),
        "frontier_committor_delta": float(challenger["frontier_committor"])
        - float(baseline["frontier_committor"]),
        "baseline_top_process_event_connexion": baseline.get(
            "top_process_event_connexion"
        ),
        "challenger_top_process_event_connexion": challenger.get(
            "top_process_event_connexion"
        ),
        "top_process_changed": baseline.get("top_process_event_connexion")
        != challenger.get("top_process_event_connexion"),
    }


def _csv_row(row: dict[str, Any], *, fieldnames: list[str]) -> dict[str, Any]:
    csv_row = row.copy()
    for key in ("closed_nonentry_states", "queue_head"):
        csv_row[key] = " ".join(str(value) for value in row[key])
    csv_row.pop("processes", None)
    return {key: csv_row.get(key) for key in fieldnames}


def _processes(channels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[Any, dict[str, Any]] = {}
    for channel in channels:
        if not channel.get("ok", True):
            continue
        event = _event_value(channel.get("event_connexion"))
        item = grouped.setdefault(
            event,
            {
                "event_connexion": event,
                "count": 0,
                "committor_sum": 0.0,
                "rate_sum": 0.0,
                "state_connexions": [],
            },
        )
        item["count"] += 1
        item["committor_sum"] += _float_or_zero(channel.get("committor"))
        item["rate_sum"] += _float_or_zero(channel.get("rate"))
        if "state_connexion" in channel:
            item["state_connexions"].append(int(channel["state_connexion"]))

    rows = list(grouped.values())
    for row in rows:
        row["state_connexions"] = sorted(set(row["state_connexions"]))
    rows.sort(
        key=lambda row: (
            float(row["committor_sum"]),
            float(row["rate_sum"]),
            str(row["event_connexion"]),
        ),
        reverse=True,
    )
    return rows


def _is_failed_refinement_channel(channel: dict[str, Any]) -> bool:
    return (
        channel.get("refinement_ok") is False
        or channel.get("rate_source") == "failed-refinement"
    )


def _event_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payloads", nargs="+", type=Path)
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    payload = comparison_payload(load_payloads(args.payloads))
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
