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
is serialized into a sidecar payload addressed by compact metadata_json;
barrier_ev / prefactor_inv_s carry the process model; product_env_hash =
id_final so the env_hash -> product digraph (the superbasin coarse graph)
is populated for free.
"""
from __future__ import annotations

import base64
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_HASH_PREFIX = b"sha256:"
_MAX_INLINE_KEY_BYTES = 96
_ROW_DIGEST_FIELD = "pykmc_row_sha256"
_PROCESS_SIGNATURE_FIELD = "pykmc_process_signature"


def _key(event_id: Any) -> bytes:
    if isinstance(event_id, bytes):
        raw = event_id
    else:
        raw = str(event_id).encode("utf-8")
    if len(raw) <= _MAX_INLINE_KEY_BYTES:
        return raw
    return _HASH_PREFIX + hashlib.sha256(raw).hexdigest().encode("ascii")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _stable_id(value: Any) -> object:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    return str(value)


def _rounded_float(value: Any, ndigits: int = 6) -> float | None:
    optional = _optional_float(value)
    if optional is None:
        return None
    return round(optional, ndigits)


def _types_signature(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    values = np.asarray(value)
    if values.ndim == 0:
        return [str(values.item())]
    return [str(item) for item in values.tolist()]


def _centered_geometry_digest(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    positions = np.asarray(value, dtype=float)
    if positions.size == 0:
        return ""
    if positions.ndim != 2:
        return None
    centered = positions - positions.mean(axis=0)
    rounded = np.round(centered, decimals=4)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def _process_signature(row: dict[str, Any]) -> str:
    payload = {
        "event_id": _stable_id(row.get("event_id")),
        "id_final": _stable_id(row.get("id_final")),
        "id_saddle": _stable_id(row.get("id_saddle")),
        "energy_barrier": _rounded_float(row.get("energy_barrier")),
        "reverse_energy_barrier": _rounded_float(row.get("reverse_energy_barrier")),
        "types": _types_signature(row.get("types")),
        "saddle_geometry": _centered_geometry_digest(row.get("saddle_positions")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class AmselKdbCatalog:
    """Thin wrapper over amsel.KdbStore for pyKMC reference events."""

    def __init__(self, path: str, discovery_temperature: float = 0.0):
        from amsel import KdbStore

        self.row_dir = Path(str(path) + ".pykmc_rows")
        self.store = KdbStore(str(path))
        self.T = float(discovery_temperature)

    def store_row(self, row: pd.Series) -> None:
        """Persist one reference-event row keyed by its event_id."""
        from amsel import KdbProcess

        d = {k: row[k] for k in row.index}
        env_key = _key(d.get("event_id", b""))
        process_signature = _process_signature(d)
        if self._contains_process_signature(env_key, process_signature):
            return
        metadata = self._store_row_payload(d, process_signature)
        barrier = _optional_float(d.get("energy_barrier"))
        prefactor = _optional_float(d.get("prefactor_inv_s"))
        if barrier is None:
            barrier = 0.0
        if prefactor is None or prefactor <= 0.0:
            prefactor = 1.0e13
        product_env = (
            _key(d.get("id_final", b""))
            if d.get("id_final") not in (None, "")
            else b""
        )
        self.store.insert(
            env_key,
            KdbProcess(
                saddle_con=b"",
                product_con=b"",
                barrier_ev=barrier,
                prefactor_inv_s=prefactor,
                discovery_temperature=self.T,
                context_signature=b"",
                usage_hint="RefineFirst",
                product_env_hash=product_env,
                metadata_json=metadata,
            ),
        )

    def _store_row_payload(self, row: dict[str, Any], process_signature: str) -> str:
        payload = pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)
        digest = hashlib.sha256(payload).hexdigest()
        self.row_dir.mkdir(parents=True, exist_ok=True)
        target = self.row_dir / f"{digest}.pkl"
        if not target.exists():
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(payload)
            tmp.replace(target)
        return json.dumps(
            {
                _ROW_DIGEST_FIELD: digest,
                _PROCESS_SIGNATURE_FIELD: process_signature,
            },
            separators=(",", ":"),
        )

    def _contains_process_signature(self, env_key: bytes, signature: str) -> bool:
        try:
            procs = self.store.lookup(env_key)
        except Exception:
            return False
        for process in procs:
            process_signature = self._metadata_process_signature(
                getattr(process, "metadata_json", "") or ""
            )
            if process_signature == signature:
                return True
        return False

    def lookup_rows(self, event_id: Any) -> list[pd.Series]:
        """Cached reference-event rows for an environment, or []."""
        try:
            procs = self.store.lookup(_key(event_id))
        except Exception:
            return []
        rows: list[pd.Series] = []
        seen_signatures: set[str] = set()
        for p in procs:
            meta = getattr(p, "metadata_json", "") or ""
            if not meta:
                continue
            try:
                d = self._load_row_payload(meta)
                signature = self._metadata_process_signature(meta)
                if signature is None:
                    signature = _process_signature(d)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                rows.append(pd.Series(d))
            except Exception:
                continue
        return rows

    def _metadata_process_signature(self, metadata: str) -> str | None:
        try:
            decoded = json.loads(metadata)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            signature = decoded.get(_PROCESS_SIGNATURE_FIELD)
            if signature:
                return str(signature)
            if _ROW_DIGEST_FIELD in decoded:
                try:
                    return _process_signature(self._load_row_payload(metadata))
                except Exception:
                    return None
        try:
            return _process_signature(self._load_row_payload(metadata))
        except Exception:
            return None

    def _load_row_payload(self, metadata: str) -> dict[str, Any]:
        try:
            decoded = json.loads(metadata)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict) and _ROW_DIGEST_FIELD in decoded:
            digest = str(decoded[_ROW_DIGEST_FIELD])
            return pickle.loads((self.row_dir / f"{digest}.pkl").read_bytes())
        return pickle.loads(base64.b64decode(metadata))

    def __len__(self) -> int:
        try:
            return len(self.store)
        except Exception:
            return 0
