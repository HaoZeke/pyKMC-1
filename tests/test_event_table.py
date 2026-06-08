import pandas as pd
import numpy as np
from types import SimpleNamespace

from pykmc.event_table import (
    ActiveEventTable,
    ReferenceEventTable,
    duplicate_event_error_info,
)
from pykmc.result import ErrorType, EventRefinementOutput


def test_duplicate_event_error_info_preserves_matched_process_identity():
    event = pd.Series(
        {
            "event_id": "env-a",
            "id_final": "env-b",
            "energy_barrier": 0.21,
            "k": 2.0,
        }
    )
    matched = pd.Series(
        {
            "idx_ref": 5,
            "event_id": "env-a",
            "id_final": "env-b",
            "energy_barrier": 0.20,
            "k": 3.0,
        }
    )

    error = duplicate_event_error_info(event, matched)

    assert error.type is ErrorType.EVENT_NOT_NEW
    assert error.variables == {
        "matched_idx_ref": 5,
        "event_id": "env-a",
        "id_final": "env-b",
        "energy_barrier": 0.20,
        "k": 3.0,
    }


def test_reference_event_add_links_two_direction_events_recursively():
    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.kdb = None
    table.table = pd.DataFrame(
        {
            "idx_ref": pd.Series(dtype="int64"),
            "event_id": pd.Series(dtype="str"),
            "id_final": pd.Series(dtype="str"),
            "energy_barrier": pd.Series(dtype="float64"),
            "k": pd.Series(dtype="float64"),
            "idx_backward": pd.Series(dtype="int64"),
        }
    )
    dfevent = pd.DataFrame(
        [
            {
                "idx_ref": -1,
                "event_id": "env-a",
                "id_final": "env-b",
                "energy_barrier": 0.10,
                "k": 2.0,
                "idx_backward": -1,
            },
            {
                "idx_ref": -1,
                "event_id": "env-b",
                "id_final": "env-a",
                "energy_barrier": 0.12,
                "k": 3.0,
                "idx_backward": -1,
            },
        ]
    )

    table.add(dfevent)

    assert table.table["idx_ref"].tolist() == [0, 1]
    assert table.table["idx_backward"].tolist() == [1, 0]


def test_reference_event_series_uses_directional_vineyard_prefactors(monkeypatch):
    calls = []

    def fake_compute_rate(dE_forward, dE_backward, config, **kwargs):
        calls.append((dE_forward, dE_backward, kwargs))
        return kwargs["prefactor_inv_s"] / 1.0e12

    import pykmc.event_table as event_table

    monkeypatch.setattr(event_table, "compute_rate", fake_compute_rate)
    monkeypatch.setattr(event_table, "graph", lambda *_args, **_kwargs: ["env"])
    monkeypatch.setattr(
        event_table,
        "unique_symmetries",
        lambda *_args, **_kwargs: ([np.eye(3)], [np.array([0])]),
    )
    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.config = SimpleNamespace(
        atomicenvironment=SimpleNamespace(rnei=0.1, rcut=0.5),
        ira=SimpleNamespace(sym_thr=0.1),
        rateconstant=SimpleNamespace(style="amsel-vtst", T=300.0),
    )
    positions = np.array([[1.0, 1.0, 1.0]], dtype=float)
    cell = np.eye(3) * 10.0

    forward, backward = table._build_event_series(
        min1_positions=positions,
        saddle_positions=positions + np.array([[0.1, 0.0, 0.0]]),
        min2_positions=positions + np.array([[0.2, 0.0, 0.0]]),
        index_move=0,
        dE_forward=0.2,
        dE_backward=0.3,
        cell=cell,
        prefactor_inv_s=1.1e13,
        product_prefactor_inv_s=2.2e13,
        prefactor_source="vineyard-finite-difference",
        saddle_freq_invcm=120.0,
        barrier_omega_rad_per_s=2.4e13,
    )

    assert forward["k"] == 11.0
    assert backward["k"] == 22.0
    assert forward["prefactor_inv_s"] == 1.1e13
    assert backward["prefactor_inv_s"] == 2.2e13
    assert forward["prefactor_source"] == "vineyard-finite-difference"
    assert backward["prefactor_source"] == "vineyard-finite-difference"
    assert calls[0][2]["prefactor_inv_s"] == 1.1e13
    assert calls[1][2]["prefactor_inv_s"] == 2.2e13
    assert calls[0][2]["saddle_freq_invcm"] == 120.0
    assert calls[1][2]["barrier_omega_rad_per_s"] == 2.4e13


def test_active_event_series_uses_vineyard_prefactor(monkeypatch):
    calls = []

    def fake_compute_rate(dE_forward, dE_backward, config, **kwargs):
        calls.append((dE_forward, dE_backward, kwargs))
        return kwargs["prefactor_inv_s"] / 1.0e12

    import pykmc.event_table as event_table

    monkeypatch.setattr(event_table, "compute_rate", fake_compute_rate)
    table = ActiveEventTable.__new__(ActiveEventTable)
    table.config = SimpleNamespace(
        rateconstant=SimpleNamespace(style="amsel-vtst", T=300.0),
    )
    event = EventRefinementOutput(
        central_atom_index=0,
        saddle_positions=np.array([[0.1, 0.0, 0.0]], dtype=float),
        E_saddle=0.2,
        min2_positions=np.array([[0.2, 0.0, 0.0]], dtype=float),
        dE_forward=0.2,
        num_reference_event=3,
        refined="F",
        prefactor_inv_s=1.1e13,
        prefactor_source="vineyard-finite-difference",
        saddle_freq_invcm=120.0,
        barrier_omega_rad_per_s=2.4e13,
    )

    row = table.build_event_series(event)

    assert row["k"] == 11.0
    assert row["prefactor_inv_s"] == 1.1e13
    assert row["prefactor_source"] == "vineyard-finite-difference"
    assert calls[0][2]["prefactor_inv_s"] == 1.1e13
    assert calls[0][2]["saddle_freq_invcm"] == 120.0
