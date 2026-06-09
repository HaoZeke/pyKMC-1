import sys
import types

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
