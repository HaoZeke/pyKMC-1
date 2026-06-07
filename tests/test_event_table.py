import pandas as pd
import numpy as np
from types import SimpleNamespace

from pykmc.event_table import ReferenceEventTable, duplicate_event_error_info
from pykmc.result import ErrorType


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


def test_reference_event_series_uses_directional_vineyard_prefactors(monkeypatch):
    calls = []

    def fake_compute_rate(dE_forward, dE_backward, config, **kwargs):
        calls.append((dE_forward, dE_backward, kwargs))
        return kwargs["prefactor_inv_s"] / 1.0e12

    import pykmc.event_table as event_table

    monkeypatch.setattr(event_table, "compute_rate", fake_compute_rate)
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
