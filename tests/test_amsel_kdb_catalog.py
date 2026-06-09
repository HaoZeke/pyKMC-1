import sys
import types

import numpy as np
import pandas as pd

from pykmc.basins.amsel_kdb_catalog import AmselKdbCatalog


def test_amsel_kdb_catalog_stores_prefactor_not_temperature_rate(monkeypatch):
    inserted = []

    class FakeKdbProcess:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeKdbStore:
        def __init__(self, path):
            self.path = path

        def insert(self, env_hash, process):
            inserted.append((env_hash, process))

    monkeypatch.setitem(
        sys.modules,
        "amsel",
        types.SimpleNamespace(KdbStore=FakeKdbStore, KdbProcess=FakeKdbProcess),
    )
    catalog = AmselKdbCatalog("catalog.lmdb", discovery_temperature=500.0)

    catalog.store_row(
        pd.Series(
            {
                "event_id": "env-a",
                "id_final": "env-b",
                "energy_barrier": 0.2,
                "k": 42.0,
                "prefactor_inv_s": 6.5e12,
            }
        )
    )

    assert inserted[0][0] == b"env-a"
    process = inserted[0][1]
    assert process.barrier_ev == 0.2
    assert process.prefactor_inv_s == 6.5e12
    assert process.discovery_temperature == 500.0


def test_amsel_kdb_catalog_defaults_missing_prefactor_to_attempt_frequency(
    monkeypatch,
):
    inserted = []

    class FakeKdbProcess:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeKdbStore:
        def __init__(self, path):
            self.path = path

        def insert(self, env_hash, process):
            inserted.append((env_hash, process))

    monkeypatch.setitem(
        sys.modules,
        "amsel",
        types.SimpleNamespace(KdbStore=FakeKdbStore, KdbProcess=FakeKdbProcess),
    )
    catalog = AmselKdbCatalog("catalog.lmdb", discovery_temperature=300.0)

    catalog.store_row(
        pd.Series({"event_id": "env-a", "energy_barrier": 0.2, "k": 42.0})
    )

    assert inserted[0][1].prefactor_inv_s == 1.0e13


def test_amsel_kdb_catalog_compacts_large_keys_and_payloads(tmp_path, monkeypatch):
    inserted = {}

    class FakeKdbProcess:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeKdbStore:
        def __init__(self, path):
            self.path = path

        def insert(self, env_hash, process):
            if len(env_hash) > 128:
                raise AssertionError("env key too large")
            if len(process.product_env_hash) > 128:
                raise AssertionError("product key too large")
            if len(process.metadata_json) > 512:
                raise AssertionError("metadata payload too large")
            inserted[env_hash] = process

        def lookup(self, env_hash):
            return [inserted[env_hash]]

    monkeypatch.setitem(
        sys.modules,
        "amsel",
        types.SimpleNamespace(KdbStore=FakeKdbStore, KdbProcess=FakeKdbProcess),
    )
    catalog = AmselKdbCatalog(str(tmp_path / "amsel.kdb"), discovery_temperature=300.0)
    positions = np.arange(900, dtype=float).reshape(300, 3)
    event_id = b"env-" + (b"x" * 2048)
    product_id = b"prod-" + (b"y" * 2048)

    catalog.store_row(
        pd.Series(
            {
                "event_id": event_id,
                "id_final": product_id,
                "initial_positions": positions,
                "saddle_positions": positions + 0.1,
                "final_positions": positions + 0.2,
                "energy_barrier": 0.2,
                "k": 42.0,
                "prefactor_inv_s": 6.5e12,
            }
        )
    )

    rows = catalog.lookup_rows(event_id)

    assert len(rows) == 1
    assert rows[0]["event_id"] == event_id
    assert rows[0]["id_final"] == product_id
    np.testing.assert_allclose(rows[0]["initial_positions"], positions)


def test_amsel_kdb_catalog_deduplicates_stable_process_signature(
    tmp_path,
    monkeypatch,
):
    inserted = {}

    class FakeKdbProcess:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeKdbStore:
        def __init__(self, path):
            self.path = path

        def insert(self, env_hash, process):
            inserted.setdefault(env_hash, []).append(process)

        def lookup(self, env_hash):
            return list(inserted.get(env_hash, []))

    monkeypatch.setitem(
        sys.modules,
        "amsel",
        types.SimpleNamespace(KdbStore=FakeKdbStore, KdbProcess=FakeKdbProcess),
    )
    catalog = AmselKdbCatalog(str(tmp_path / "amsel.kdb"), discovery_temperature=300.0)
    saddle = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=float)
    row = {
        "idx_ref": 1,
        "idx_backward": 2,
        "event_id": "env-a",
        "id_final": "env-b",
        "id_saddle": "env-s",
        "energy_barrier": 0.2,
        "saddle_positions": saddle,
        "types": np.array(["Cu", "Cu"]),
        "k": 1.0,
        "prefactor_inv_s": 6.5e12,
    }
    duplicate = dict(row)
    duplicate["idx_ref"] = 77
    duplicate["idx_backward"] = 78
    duplicate["k"] = 9.0

    catalog.store_row(pd.Series(row))
    catalog.store_row(pd.Series(duplicate))

    assert len(inserted[b"env-a"]) == 1
    rows = catalog.lookup_rows("env-a")
    assert len(rows) == 1
    assert rows[0]["idx_ref"] == 1
