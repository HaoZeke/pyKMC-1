import importlib.util
from io import StringIO
from pathlib import Path

import pytest


def _load_script():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "compare_amsel_exploration.py"
    )
    spec = importlib.util.spec_from_file_location("compare_amsel_exploration", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload(priority, closed, resolved, frontier, event_connexion=1, rate=2.0):
    return {
        "ok": True,
        "case": f"case-{priority}",
        "refinement": {"ok": True, "rate_source": "refined"},
        "channel_summary": {"committor_sum": resolved, "rate_sum": rate},
        "channels": [
            {
                "ok": True,
                "event_connexion": event_connexion,
                "state_connexion": closed[-1],
                "committor": resolved,
                "rate": rate,
            }
        ],
        "transient_states": [
            {"state": 10 + idx, "hit_committor": value}
            for idx, value in enumerate(frontier)
        ],
        "exploration": {
            "priority": priority,
            "max_closed_states": 2,
            "expanded_states": [0],
            "closed_states": closed,
            "states_to_explore": [13, 14, 1],
            "guidance_scores": {"13": 0.125, "14": 0.125, "1": 2.5e-9},
        },
    }


def test_summarize_payload_reports_resolved_and_frontier_mass():
    script = _load_script()
    row = script.summarize_payload(
        _payload("amsel", closed=[0, 13], resolved=0.125, frontier=[0.125, 0.125])
    )

    assert row["priority"] == "amsel"
    assert row["closed_states"] == [0, 13]
    assert row["closed_nonentry_states"] == [13]
    assert row["resolved_committor"] == pytest.approx(0.125)
    assert row["resolved_rate"] == pytest.approx(2.0)
    assert row["frontier_committor"] == pytest.approx(0.25)
    assert row["accounted_committor"] == pytest.approx(0.375)
    assert row["kinetic_confidence"] == pytest.approx(0.125 / 0.375)
    assert row["top_process_event_connexion"] == 1
    assert row["processes"] == [
        {
            "event_connexion": 1,
            "count": 1,
            "committor_sum": 0.125,
            "rate_sum": 2.0,
            "state_connexions": [13],
        }
    ]
    assert row["queue_head"] == [13, 14, 1]


def test_comparison_payload_reports_amsel_gain_over_legacy():
    script = _load_script()
    payload = script.comparison_payload(
        [
            _payload(
                "legacy",
                closed=[0, 1],
                resolved=2.5e-9,
                frontier=[0.125, 0.125],
                event_connexion=0,
                rate=1.0e-8,
            ),
            _payload(
                "amsel",
                closed=[0, 13],
                resolved=0.125,
                frontier=[0.125],
                event_connexion=1,
                rate=2.0,
            ),
        ]
    )

    assert [row["priority"] for row in payload["rows"]] == ["legacy", "amsel"]
    assert payload["gain"]["baseline_priority"] == "legacy"
    assert payload["gain"]["challenger_priority"] == "amsel"
    assert payload["gain"]["resolved_committor_delta"] == pytest.approx(0.1249999975)
    assert payload["gain"]["kinetic_confidence_delta"] == pytest.approx(
        0.5 - (2.5e-9 / (0.2500000025))
    )
    assert payload["gain"]["top_process_changed"] is True
    assert payload["gain"]["baseline_top_process_event_connexion"] == 0
    assert payload["gain"]["challenger_top_process_event_connexion"] == 1
    assert payload["gain"]["closed_budget"] == 2


def test_write_csv_filters_non_tabular_fields():
    script = _load_script()
    payload = script.comparison_payload(
        [
            _payload("legacy", closed=[0, 1], resolved=2.5e-9, frontier=[0.125]),
            _payload("amsel", closed=[0, 13], resolved=0.125, frontier=[0.125]),
        ]
    )
    out = StringIO()

    script.write_csv(payload, out)

    assert "priority,ok,rate_source" in out.getvalue()
    assert "legacy" in out.getvalue()
    assert "amsel" in out.getvalue()
