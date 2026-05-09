"""AMSEL basin-search registry wrapper for pyKMC.

Wraps :class:`amsel.BasinSearchRegistry` with a pyKMC-friendly surface that
takes the displacement vector ``saddle_positions - reactant_positions`` as
the mode signature. Used by :class:`pykmc.basins.basin.BasinsGenericEvents`
to deduplicate ARTn refinement attempts on absorbing exits whose mode is
sufficiently similar to a search already claimed at the same state.

The registry is JSONL-backed; passing ``path=None`` keeps the
record set in memory and is useful when a per-trial registry should not
persist (the recombination harness overwrites trial dirs anyway).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


try:
    from amsel import BasinSearchRegistry, BasinSearchClaim, basin_search_registry_summary

    _AMSEL_REGISTRY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dep
    BasinSearchRegistry = None  # type: ignore[assignment]
    BasinSearchClaim = None  # type: ignore[assignment]
    basin_search_registry_summary = None  # type: ignore[assignment]
    _AMSEL_REGISTRY_AVAILABLE = False


class _NullClaim:
    """Stand-in for BasinSearchClaim when amsel is unavailable."""

    accepted = True
    reason = "amsel-registry-unavailable"
    duplicate_of = None
    similarity = None
    record = None


class BasinSearchRegistryAdapter:
    """pyKMC-side wrapper around :class:`amsel.BasinSearchRegistry`.

    Parameters
    ----------
    path : Path or None
        JSONL file backing the registry. ``None`` keeps records in memory
        only (still useful within a single trial).
    similarity_threshold : float
        Cosine similarity above which two modes count as the same search
        channel. Default 0.98 matches the amsel default.
    """

    def __init__(
        self,
        path: Path | None,
        *,
        similarity_threshold: float = 0.98,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.similarity_threshold = float(similarity_threshold)
        self._available = bool(_AMSEL_REGISTRY_AVAILABLE) and self.path is not None
        if self._available:
            self._registry = BasinSearchRegistry(
                self.path,
                mode_similarity_threshold=self.similarity_threshold,
            )
        else:
            self._registry = None

    @property
    def available(self) -> bool:
        return self._available

    def claim_refinement(
        self,
        *,
        state: int,
        wuid: int,
        saddle_positions: np.ndarray,
        reactant_positions: np.ndarray,
        displacement_type: str = "absorbing-refinement",
    ):
        """Try to claim an ARTn refinement channel.

        Returns a ``BasinSearchClaim`` (or a null stand-in when amsel is
        unavailable). When the claim is *not* accepted, the caller should
        skip the ARTn job and reuse the prior result keyed by
        ``claim.duplicate_of``.
        """
        if not self._available:
            return _NullClaim()
        mode = np.asarray(saddle_positions, dtype=float) - np.asarray(
            reactant_positions, dtype=float
        )
        if mode.ndim == 1:
            if mode.size % 3 != 0:
                return _NullClaim()
            mode = mode.reshape(-1, 3)
        if not np.all(np.isfinite(mode)):
            return _NullClaim()
        norm = float(np.linalg.norm(mode))
        if norm <= 0.0:
            return _NullClaim()
        return self._registry.claim_search(
            state=int(state),
            wuid=int(wuid),
            mode=mode,
            displacement_type=displacement_type,
        )

    def mark_completed(self, wuid: int, *, result: str = "ok") -> None:
        if not self._available:
            return
        try:
            self._registry.mark_completed(int(wuid), result=result)
        except AttributeError:
            return

    def summary(self) -> dict[str, Any]:
        if not self._available or basin_search_registry_summary is None:
            return {
                "available": False,
                "active": 0,
                "completed": 0,
                "suppressed": 0,
            }
        return dict(basin_search_registry_summary(self._registry))
