import pandas as pd
import numpy as np
from types import SimpleNamespace

import pytest

from pykmc.result import Err, ErrorInfo, ErrorType, EventSearchOutput
from pykmc.event_table import (
    ActiveEventTable,
    ReferenceEventTable,
    duplicate_event_error_info,
)
from pykmc.result import EventRefinementOutput


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


def test_reference_event_table_raises_when_configured_kdb_open_fails(monkeypatch):
    import pykmc.basins.amsel_kdb_catalog as kdb_catalog

    class FailingKdb:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("open failed")

    monkeypatch.setattr(kdb_catalog, "AmselKdbCatalog", FailingKdb)
    config = SimpleNamespace(
        control=SimpleNamespace(reference_table=None, kdb_path="shared-kdb"),
        rateconstant=SimpleNamespace(T=300.0),
    )

    with pytest.raises(RuntimeError, match="shared-kdb"):
        ReferenceEventTable(config)


def test_reference_event_add_raises_when_configured_kdb_store_fails():
    class FailingKdb:
        def store_row(self, _row):
            raise RuntimeError("store failed")

    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.kdb = FailingKdb()
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

    with pytest.raises(RuntimeError, match="store failed"):
        table.add(
            pd.Series(
                {
                    "idx_ref": -1,
                    "event_id": "env-a",
                    "id_final": "env-b",
                    "energy_barrier": 0.10,
                    "k": 2.0,
                    "idx_backward": -1,
                }
            )
        )


def test_reference_event_ingest_rows_recomputes_cached_rate_for_current_config(
    monkeypatch,
):
    calls = []

    def fake_compute_rate(dE_forward, dE_backward, config, **kwargs):
        calls.append((dE_forward, dE_backward, config.rateconstant.T, kwargs))
        return config.rateconstant.T / 100.0

    import pykmc.event_table as event_table

    monkeypatch.setattr(event_table, "compute_rate", fake_compute_rate)
    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.config = SimpleNamespace(
        rateconstant=SimpleNamespace(style="amsel-vtst", T=700.0),
    )
    table.table = pd.DataFrame(
        {
            "idx_ref": pd.Series(dtype="int64"),
            "event_id": pd.Series(dtype="str"),
            "id_final": pd.Series(dtype="str"),
            "energy_barrier": pd.Series(dtype="float64"),
            "reverse_energy_barrier": pd.Series(dtype="float64"),
            "k": pd.Series(dtype="float64"),
            "prefactor_inv_s": pd.Series(dtype="float64"),
            "prefactor_source": pd.Series(dtype="str"),
            "saddle_freq_invcm": pd.Series(dtype="float64"),
            "barrier_omega_rad_per_s": pd.Series(dtype="float64"),
            "idx_backward": pd.Series(dtype="int64"),
        }
    )

    table.ingest_rows(
        [
            pd.Series(
                {
                    "idx_ref": 99,
                    "event_id": "env-a",
                    "id_final": "env-b",
                    "energy_barrier": 0.25,
                    "reverse_energy_barrier": 0.35,
                    "k": 1.0,
                    "prefactor_inv_s": 6.0e12,
                    "prefactor_source": "vineyard-finite-difference",
                    "saddle_freq_invcm": 120.0,
                    "barrier_omega_rad_per_s": 2.4e13,
                    "idx_backward": 100,
                }
            )
        ]
    )

    assert table.table.loc[0, "idx_ref"] == 0
    assert table.table.loc[0, "idx_backward"] == 0
    assert table.table.loc[0, "k"] == 7.0
    assert calls == [
        (
            0.25,
            0.35,
            700.0,
            {
                "prefactor_inv_s": 6.0e12,
                "prefactor_source": "vineyard-finite-difference",
                "saddle_freq_invcm": 120.0,
                "barrier_omega_rad_per_s": 2.4e13,
            },
        )
    ]


def test_reference_event_add_persists_reconstructable_kdb_rows():
    class CapturingKdb:
        def __init__(self):
            self.rows = []

        def store_row(self, row):
            self.rows.append(row.copy())

    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.kdb = CapturingKdb()
    table.table = pd.DataFrame(
        {
            "idx_ref": pd.Series(dtype="int64"),
            "event_id": pd.Series(dtype="str"),
            "id_final": pd.Series(dtype="str"),
            "energy_barrier": pd.Series(dtype="float64"),
            "reverse_energy_barrier": pd.Series(dtype="float64"),
            "k": pd.Series(dtype="float64"),
            "prefactor_inv_s": pd.Series(dtype="float64"),
            "idx_backward": pd.Series(dtype="int64"),
        }
    )

    table.add(
        pd.DataFrame(
            [
                {
                    "idx_ref": -1,
                    "event_id": "env-a",
                    "id_final": "env-b",
                    "energy_barrier": 0.10,
                    "k": 2.0,
                    "prefactor_inv_s": 1.1e13,
                    "idx_backward": -1,
                },
                {
                    "idx_ref": -1,
                    "event_id": "env-b",
                    "id_final": "env-a",
                    "energy_barrier": 0.12,
                    "k": 3.0,
                    "prefactor_inv_s": 2.2e13,
                    "idx_backward": -1,
                },
            ]
        )
    )

    assert [row["reverse_energy_barrier"] for row in table.kdb.rows] == [0.12, 0.10]
    assert [row["prefactor_inv_s"] for row in table.kdb.rows] == [1.1e13, 2.2e13]


def test_reference_event_add_with_prefactors_skips_duplicate_prefactor_work():
    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.kdb = None
    table.config = SimpleNamespace(
        eventsearch=SimpleNamespace(
            emin_event=0.01,
            emax_event=2.0,
            backward_emin_event=0.01,
            energy_asymmetry=10.0,
        ),
        atomicenvironment=SimpleNamespace(rnei=0.1, rcut=0.5),
        ira=SimpleNamespace(kmax_factor=2.0, sym_thr=0.1),
        psr=SimpleNamespace(matching_score_thr=0.4),
        rateconstant=SimpleNamespace(style="constant"),
    )
    saddle = np.array([[0.1, 0.0, 0.0]], dtype=float)
    table.table = pd.DataFrame(
        [
            {
                "idx_ref": 0,
                "event_id": "env-a",
                "id_final": "env-b",
                "energy_barrier": 0.2,
                "saddle_positions": saddle,
                "idx_backward": 0,
            }
        ]
    )
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
        saddle_positions=saddle.copy(),
        min2_positions=np.array([[0.2, 0.0, 0.0]], dtype=float),
        dE_forward=0.2,
        dE_backward=0.2,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
    )
    calls = []

    def fail_prefactor(_events):
        calls.append("prefactor")

    import pykmc.event_table as event_table

    event_table_graph = event_table.graph
    event_table_symmetries = event_table.unique_symmetries
    try:
        event_table.graph = lambda *_args, **_kwargs: ["env-a"]
        event_table.unique_symmetries = lambda *_args, **_kwargs: (
            [np.eye(3)],
            [np.array([0])],
        )

        results = table.add_events_with_prefactors([event], fail_prefactor)
    finally:
        event_table.graph = event_table_graph
        event_table.unique_symmetries = event_table_symmetries

    assert calls == []
    assert len(results) == 1
    assert results[0].is_err()
    assert results[0].err_value().type is ErrorType.EVENT_NOT_NEW
    assert len(table.table) == 1


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


def test_reference_event_series_preserves_event_local_atom_types(monkeypatch):
    import pykmc.event_table as event_table

    monkeypatch.setattr(event_table, "compute_rate", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(event_table, "graph", lambda *_args, **_kwargs: ["env"])
    monkeypatch.setattr(
        event_table,
        "unique_symmetries",
        lambda *_args, **_kwargs: ([np.eye(3)], [np.array([0, 1])]),
    )
    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.config = SimpleNamespace(
        atomicenvironment=SimpleNamespace(rnei=0.1, rcut=0.5),
        ira=SimpleNamespace(sym_thr=0.1),
        rateconstant=SimpleNamespace(style="amsel-vtst", T=300.0),
    )
    positions = np.array([[1.0, 1.0, 1.0], [1.2, 1.0, 1.0]], dtype=float)
    cell = np.eye(3) * 10.0

    forward, backward = table._build_event_series(
        min1_positions=positions,
        saddle_positions=positions + np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]),
        min2_positions=positions + np.array([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        index_move=0,
        dE_forward=0.2,
        dE_backward=0.3,
        cell=cell,
        types=np.array(["Fe", "Cr"]),
    )

    np.testing.assert_array_equal(forward["types"], np.array(["Fe", "Cr"]))
    np.testing.assert_array_equal(backward["types"], np.array(["Fe", "Cr"]))


def test_matching_event_uses_geometry_fallback_when_ira_is_unavailable(monkeypatch):
    import pykmc.event_table as event_table

    table = ReferenceEventTable.__new__(ReferenceEventTable)
    table.config = SimpleNamespace(
        ira=SimpleNamespace(kmax_factor=2.0),
        psr=SimpleNamespace(matching_score_thr=0.4),
    )
    saddle = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=float)
    table.table = pd.DataFrame(
        [
            {
                "idx_ref": 7,
                "event_id": "env-a",
                "energy_barrier": 0.2,
                "saddle_positions": saddle,
            }
        ]
    )
    dfevent = pd.Series(
        {
            "event_id": "env-a",
            "energy_barrier": 0.21,
            "saddle_positions": saddle.copy(),
        }
    )
    monkeypatch.setattr(
        event_table,
        "simple_ira",
        lambda *_args, **_kwargs: Err(
            ErrorInfo(
                type=ErrorType.PSR_NO_MATCH_FOUND,
                message="native matcher unavailable",
            )
        ),
    )

    matched = table.matching_event(dfevent)

    assert matched is not None
    assert matched["idx_ref"] == 7


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
