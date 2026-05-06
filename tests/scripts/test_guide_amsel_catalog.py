import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("amsel")


def _load_script():
    script = Path(__file__).resolve().parents[2] / "scripts" / "guide_amsel_catalog.py"
    spec = importlib.util.spec_from_file_location("guide_amsel_catalog", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _connectivity(df: pd.DataFrame):
    return SimpleNamespace(df=df)


def test_catalog_guidance_ranks_exact_absorbing_channels():
    script = _load_script()
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0, 0],
                "state_connexion": [10, 20],
                "event_connexion": [101, 202],
                "central_atom": [3, 4],
                "sym": [0, 1],
                "transient": [False, False],
                "dE_forward": [0.4, 0.2],
                "k_forward": [1.0, 3.0],
                "dE_backward": [0.1, 0.1],
                "k_backward": [0.5, 0.5],
            }
        )
    )

    payload = script.catalog_guidance_payload(table, case_name="two-exits", entry=0)

    assert payload["ok"] is True
    assert payload["case"] == "two-exits"
    assert payload["entry"] == 0
    assert payload["channel_summary"]["committor_sum"] == pytest.approx(1.0)
    assert [row["event_connexion"] for row in payload["channels"]] == [202, 101]
    assert [row["state_connexion"] for row in payload["channels"]] == [20, 10]
    assert payload["channels"][0]["committor"] == pytest.approx(0.75)
    assert payload["channels"][1]["committor"] == pytest.approx(0.25)
    assert payload["channels"][0]["guidance_score"] == pytest.approx(0.75)
    assert payload["channels"][1]["guidance_score"] == pytest.approx(0.25)


def test_catalog_guidance_splits_exit_rows_after_transient_detour():
    script = _load_script()
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0, 0, 1],
                "state_connexion": [10, 1, 20],
                "event_connexion": [101, 303, 202],
                "central_atom": [3, 5, 4],
                "sym": [0, 2, 1],
                "transient": [False, True, False],
                "dE_forward": [0.4, 0.3, 0.2],
                "k_forward": [1.0, 9.0, 2.0],
                "dE_backward": [0.1, 0.1, 0.1],
                "k_backward": [0.5, 0.5, 0.5],
            }
        )
    )

    payload = script.catalog_guidance_payload(table, case_name="detour", entry=0)

    assert payload["ok"] is True
    assert [row["event_connexion"] for row in payload["channels"]] == [202, 101]
    assert payload["channels"][0]["committor"] == pytest.approx(0.9)
    assert payload["channels"][1]["committor"] == pytest.approx(0.1)
    assert payload["channels"][0]["state"] == 1
    assert payload["channels"][1]["state"] == 0


def test_transient_state_guidance_ranks_states_by_hitting_probability():
    script = _load_script()
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0, 0, 1, 2],
                "state_connexion": [1, 2, 10, 20],
                "event_connexion": [301, 302, 101, 202],
                "central_atom": [5, 6, 3, 4],
                "sym": [0, 0, 0, 0],
                "transient": [True, True, False, False],
                "dE_forward": [0.1, 0.5, 0.4, 0.2],
                "k_forward": [9.0, 1.0, 1.0, 1.0],
                "dE_backward": [0.1, 0.1, 0.1, 0.1],
                "k_backward": [0.5, 0.5, 0.5, 0.5],
            }
        )
    )

    payload = script.catalog_guidance_payload(table, case_name="frontier", entry=0)

    assert payload["ok"] is True
    assert [row["state"] for row in payload["transient_states"]] == [1, 2]
    assert payload["transient_states"][0]["hit_committor"] == pytest.approx(0.9)
    assert payload["transient_states"][1]["hit_committor"] == pytest.approx(0.1)
    assert payload["transient_states"][0]["incoming_event_connexion"] == 301
    assert payload["transient_states"][1]["incoming_event_connexion"] == 302


def test_frontier_transient_rows_are_guidance_targets_not_exit_channels():
    script = _load_script()
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0, 0],
                "state_connexion": [1, 10],
                "event_connexion": [301, 101],
                "central_atom": [5, 3],
                "sym": [0, 0],
                "transient": [True, False],
                "dE_forward": [0.1, 0.4],
                "k_forward": [9.0, 1.0],
                "dE_backward": [0.1, 0.1],
                "k_backward": [0.5, 0.5],
            }
        )
    )

    payload = script.catalog_guidance_payload(table, case_name="frontier", entry=0)

    assert payload["ok"] is True
    assert [row["state_connexion"] for row in payload["channels"]] == [10]
    assert payload["channel_summary"]["committor_sum"] == pytest.approx(0.1)
    assert [row["state"] for row in payload["transient_states"]] == [1]
    assert payload["transient_states"][0]["hit_committor"] == pytest.approx(0.9)
    assert payload["transient_states"][0]["incoming_event_connexion"] == 301


def test_exploration_payload_records_budgeted_queue_state():
    script = _load_script()
    payload = {"ok": True}
    basin = SimpleNamespace(
        exploration_order=[0],
        explored_states=[0, 1],
        states_to_explore=[13, 14, 1],
        last_exploration_guidance={13: 0.375, 14: 0.375, 1: 0.25},
    )

    script._attach_exploration_payload(
        payload,
        basin=basin,
        priority="amsel",
        max_expansions=1,
    )

    assert payload["exploration"] == {
        "priority": "amsel",
        "max_expansions": 1,
        "expanded_states": [0],
        "closed_states": [0, 1],
        "states_to_explore": [13, 14, 1],
        "guidance_scores": {"1": 0.25, "13": 0.375, "14": 0.375},
    }
