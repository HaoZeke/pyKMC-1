"""amsel KDB as pyKMC's persistent, cross-chain process catalogue.

pyKMC re-discovers events with pARTn every step and persists only to a
local reference_table.pickle. At a defect configuration most pARTn
searches return "no event found", so the kMC starves. The amsel KDB
(amsel.KdbStore, an LMDB process catalogue keyed by environment hash)
fixes this: every discovered reference event is stored keyed by its
pyKMC event_id (the IRA atomic-environment graph hash), and on each new
environment the catalogue is queried first -- a HIT injects the cached
events and SKIPS the pARTn search. The same store backs every chain, so
events found once are reused everywhere (and the KDB's product_env_hash
+ superbasin_exit projection become available for recombination).

A pyKMC reference-event row (positions, symmetry matrices, barrier, rate)
is serialized into a KdbProcess: the heavy numpy payload is base64-pickled
into metadata_json; barrier_ev / prefactor_inv_s carry the kinetics;
product_env_hash = id_final so the env_hash -> product digraph (the
superbasin coarse graph) is populated for free.
"""
from __future__ import annotations

import base64
import pickle
from typing import Any

import pandas as pd


def _key(event_id: Any) -> bytes:
    if isinstance(event_id, bytes):
        return event_id
    return str(event_id).encode("utf-8")


class AmselKdbCatalog:
    """Thin wrapper over amsel.KdbStore for pyKMC reference events."""

    def __init__(self, path: str, discovery_temperature: float = 0.0):
        from amsel import KdbStore

        self.store = KdbStore(str(path))
        self.T = float(discovery_temperature)

    def store_row(self, row: pd.Series) -> None:
        """Persist one reference-event row keyed by its event_id."""
        from amsel import KdbProcess

        d = {k: row[k] for k in row.index}
        blob = base64.b64encode(pickle.dumps(d)).decode("ascii")
        barrier = float(d.get("energy_barrier", 0.0) or 0.0)
        rate = float(d.get("k", 0.0) or 0.0)
        product_env = (
            _key(d.get("id_final", b""))
            if d.get("id_final") not in (None, "")
            else b""
        )
        self.store.insert(
            _key(d.get("event_id", b"")),
            KdbProcess(
                saddle_con=b"",
                product_con=b"",
                barrier_ev=barrier,
                prefactor_inv_s=rate if rate > 0 else 1.0e13,
                discovery_temperature=self.T,
                context_signature=b"",
                usage_hint="RefineFirst",
                product_env_hash=product_env,
                metadata_json=blob,
            ),
        )

    def lookup_rows(self, event_id: Any) -> list[pd.Series]:
        """Cached reference-event rows for an environment, or []."""
        try:
            procs = self.store.lookup(_key(event_id))
        except Exception:
            return []
        rows: list[pd.Series] = []
        for p in procs:
            meta = getattr(p, "metadata_json", "") or ""
            if not meta:
                continue
            try:
                d = pickle.loads(base64.b64decode(meta))
                rows.append(pd.Series(d))
            except Exception:
                continue
        return rows

    def __len__(self) -> int:
        try:
            return len(self.store)
        except Exception:
            return 0
