"""ASV benchmarks for AMSEL-backed basin operations."""
from __future__ import annotations

import itertools

import pandas as pd

from pykmc.basins import AmselFPTASelector, FPTASelector, StatesConnectivity
from pykmc.basins.amsel_guidance import amsel_state_guidance_scores


class CycleRng:
    def __init__(self, values):
        self._values = itertools.cycle(float(value) for value in values)

    def random(self):
        return next(self._values)


class BasinAmselSuite:
    params = [4, 8, 16]
    param_names = ["n_transient"]

    def setup(self, n_transient):
        self.table = make_basin_table(n_transient)
        self.legacy_selector = FPTASelector(rng=CycleRng([0.37, 0.42]))
        self.amsel_selector = AmselFPTASelector(
            clock_mode="sampled",
            rng=CycleRng([0.37, 0.42]),
        )

    def time_legacy_fpta_selector(self, n_transient):
        self.legacy_selector.select_from_connectivity(self.table)

    def time_amsel_sampled_selector(self, n_transient):
        self.amsel_selector.select_from_connectivity(self.table)

    def time_amsel_frontier_guidance(self, n_transient):
        amsel_state_guidance_scores(self.table, entry=0)


def make_basin_table(n_transient):
    table = StatesConnectivity()
    rows = []
    next_absorbing = n_transient
    for state in range(n_transient):
        if state + 1 < n_transient:
            rows.append(
                _row(
                    state=state,
                    state_connexion=state + 1,
                    transient=True,
                    k_forward=1.0 + 0.1 * state,
                )
            )
        rows.append(
            _row(
                state=state,
                state_connexion=next_absorbing,
                transient=False,
                k_forward=0.25 + 0.05 * state,
            )
        )
        next_absorbing += 1
    rows.append(
        _row(
            state=0,
            state_connexion=next_absorbing,
            transient=True,
            k_forward=2.0,
        )
    )
    table.df = pd.DataFrame(rows)
    return table


def _row(state, state_connexion, transient, k_forward):
    return {
        "state": state,
        "state_connexion": state_connexion,
        "event_connexion": state_connexion,
        "central_atom": state,
        "sym": 0,
        "transient": transient,
        "dE_forward": 0.0,
        "k_forward": k_forward,
        "dE_backward": 0.0,
        "k_backward": 0.0,
    }
