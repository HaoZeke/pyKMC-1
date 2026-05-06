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
    frontier_committor = sum(
        _float_or_zero(row.get("hit_committor"))
        for row in transient_states
        if row.get("ok", True)
    )

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
        "frontier_committor": frontier_committor,
        "accounted_committor": resolved_committor + frontier_committor,
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
    rows = [_csv_row(row) for row in payload["rows"]]
    fieldnames = [
        "priority",
        "ok",
        "rate_source",
        "refinement_ok",
        "closed_count",
        "closed_nonentry_states",
        "resolved_committor",
        "frontier_committor",
        "accounted_committor",
        "queue_head",
    ]
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
        "frontier_committor_delta": float(challenger["frontier_committor"])
        - float(baseline["frontier_committor"]),
    }


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    csv_row = row.copy()
    for key in ("closed_nonentry_states", "queue_head"):
        csv_row[key] = " ".join(str(value) for value in row[key])
    return csv_row


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
