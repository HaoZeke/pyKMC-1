from types import SimpleNamespace

import pandas as pd

from pykmc.basins import StatesConnectivity
from pykmc.basins.amsel_selection import AmselFPTASelector
import pykmc.basins.amsel_selection as amsel_selection


def test_adaptive_selector_rejects_reduced_clock_inconsistent_with_mfpt(monkeypatch):
    table = StatesConnectivity()
    table.df = pd.DataFrame(
        [
            {"state": 0, "state_connexion": 1, "k_forward": 1.0},
        ]
    )

    class FakeProblem:
        def __init__(self, *, transient, absorbing, rates):
            pass

        def reduced_kinetics(self, *, entry):
            return SimpleNamespace(
                effective_rate=1.0e-30,
                one_rate_clock_is_plausible=lambda rank_tol: True,
            )

        def fpta(self, *, entry, r):
            return SimpleNamespace(t_exit=12.0, weights=[1.0])

        def mrm(self, *, entry):
            return SimpleNamespace(tau_total=10.0, rate_to_absorbing=[0.1])

    monkeypatch.setattr(amsel_selection, "_AMSEL_AVAILABLE", True)
    monkeypatch.setattr(
        amsel_selection,
        "_amsel",
        SimpleNamespace(AmcProblem=FakeProblem, AmselError=Exception),
    )
    monkeypatch.setattr(
        AmselFPTASelector,
        "diagnose_connectivity",
        lambda self, connectivity_table, entry, include_outlets=False: {
            "ok": True,
            "mrm_moments": {"ok": True, "mean": 10.0},
            "reduced_kinetics": {
                "ok": True,
                "effective_rate": 1.0e-30,
            },
        },
    )

    selector = AmselFPTASelector(
        clock_mode="adaptive",
        rng=SimpleNamespace(random=lambda: 0.5),
    )
    result = selector.select_from_connectivity(table)

    assert result.is_ok()
    assert result.ok_value().t_exit == 12.0
    assert selector.last_clock_mode == "sampled"
