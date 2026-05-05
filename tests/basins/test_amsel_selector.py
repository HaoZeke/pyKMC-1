from __future__ import annotations

import pandas as pd
import pytest

amsel = pytest.importorskip("amsel")

from pykmc.basins import AmselFPTASelector, StatesConnectivity


class SequenceRng:
    def __init__(self, draws):
        self.draws = iter(draws)

    def random(self):
        return next(self.draws)


def _connectivity(df: pd.DataFrame) -> StatesConnectivity:
    table = StatesConnectivity()
    table.df = df
    return table


def test_adaptive_selector_uses_mean_clock_on_rank1(monkeypatch):
    draws = iter([0.25])
    monkeypatch.setattr("numpy.random.random", lambda: next(draws))

    selector = AmselFPTASelector(clock_mode="adaptive", rank_tol=1.0e-6)
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0],
                "state_connexion": [10],
                "k_forward": [42.0],
            }
        )
    )

    result = selector.select_from_connectivity(table)
    assert result.is_ok()
    assert selector.last_clock_mode == "mean"
    assert selector.last_reduced_kinetics is not None
    assert selector.last_reduced_kinetics.one_rate_clock_is_plausible(1.0e-6)
    assert result.ok_value().t_exit == pytest.approx(1.0 / 42.0)
    assert result.ok_value().exit_state == 10


def test_adaptive_selector_uses_sampled_clock_on_rankk(monkeypatch):
    draws = iter([0.5, 0.0])
    monkeypatch.setattr("numpy.random.random", lambda: next(draws))

    selector = AmselFPTASelector(clock_mode="adaptive", rank_tol=1.0e-6)
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0, 1, 0, 1],
                "state_connexion": [1, 0, 10, 20],
                "k_forward": [1.0e-4, 1.0e-4, 1.0, 1.0e-4],
            }
        )
    )

    result = selector.select_from_connectivity(table)
    assert result.is_ok()
    assert selector.last_clock_mode == "sampled"
    assert selector.last_reduced_kinetics is not None
    assert selector.last_reduced_kinetics.slow_subspace_rank == 2
    assert selector.last_reduced_kinetics.rank1_invalidity > 0.0
    assert result.ok_value().t_exit > 0.0
    assert result.ok_value().exit_state in (10, 20)


def test_selector_uses_injected_rng():
    selector = AmselFPTASelector(
        clock_mode="sampled",
        rng=SequenceRng([0.5, 0.0]),
    )
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0, 1, 0, 1],
                "state_connexion": [1, 0, 10, 20],
                "k_forward": [1.0e-4, 1.0e-4, 1.0, 1.0e-4],
            }
        )
    )

    result = selector.select_from_connectivity(table)

    assert result.is_ok()
    assert result.ok_value().t_exit > 0.0
    assert result.ok_value().exit_state == 10


def test_selector_exposes_independent_amsel_diagnostics():
    selector = AmselFPTASelector()
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0],
                "state_connexion": [10],
                "k_forward": [2.0],
            }
        )
    )

    report = selector.diagnose_connectivity(table)

    assert report["ok"] is True
    assert report["mrm_moments"]["ok"] is True
    assert report["reduced_kinetics"]["ok"] is True
    assert report["ngt_outlets"][0]["ok"] is True
    assert report["ngt_outlets"][0]["committor"] == pytest.approx(1.0)
    assert report["reduced_kinetics"]["slow_subspace_rank"] == 1
    assert selector.last_diagnostics == report


def test_adaptive_selector_samples_when_diagnostics_reject_mean_clock():
    selector = AmselFPTASelector(clock_mode="adaptive", rng=SequenceRng([0.5, 0.0]))
    table = _connectivity(
        pd.DataFrame(
            {
                "state": [0],
                "state_connexion": [10],
                "k_forward": [2.0],
            }
        )
    )

    selector.diagnose_connectivity = lambda *_args, **_kwargs: {
        "ok": True,
        "mrm_moments": {"ok": False, "error": "near singular"},
        "reduced_kinetics": {
            "ok": True,
            "slow_subspace_rank": 1,
            "rank1_invalidity": 0.0,
        },
    }

    result = selector.select_from_connectivity(table)

    assert result.is_ok()
    assert selector.last_clock_mode == "sampled"
    assert result.ok_value().exit_state == 10
