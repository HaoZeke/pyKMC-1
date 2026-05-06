"""AMSEL guidance utilities for basin graph expansion."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from .connectivity import StatesConnectivity


try:
    import amsel as _amsel

    _AMSEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when amsel missing
    _amsel = None
    _AMSEL_AVAILABLE = False


def amsel_state_guidance_scores(
    connectivity_table: StatesConnectivity,
    *,
    entry: int = 0,
) -> dict[int, float]:
    """Return AMSEL NGT hitting probabilities for basin states.

    The returned score for a state is the probability of reaching that
    state or exit-state bucket from ``entry`` before any other unresolved
    frontier or absorbing exit.
    """
    if not _AMSEL_AVAILABLE:
        return {}

    df = _connectivity_df(connectivity_table)
    if df.empty:
        return {}

    transient, absorbing, rates, state_targets = _guidance_problem(df)
    if not transient or not absorbing:
        return {}

    try:
        problem = _amsel.AmcProblem(transient=transient, absorbing=absorbing, rates=rates)
    except Exception:  # noqa: BLE001 - guidance must not block basin exploration.
        return {}

    scores: dict[int, float] = {}
    for state, targets in state_targets.items():
        if int(state) == int(entry):
            continue
        committor = 0.0
        target_ok = False
        for target in targets:
            try:
                result = problem.ngt(source=[int(entry)], target=[int(target)])
            except Exception:  # noqa: BLE001 - skip failed state targets independently.
                continue
            committor += float(result.committor)
            target_ok = True
        if target_ok:
            scores[int(state)] = committor
    return scores


def amsel_rank_basin_frontier(
    *,
    candidate_states,
    guidance: dict[int, float],
    event_families: dict[int, object | None],
    closed_event_families: set[object],
    duplicate_family_penalty: float = 1.0,
    min_guidance: float = 0.0,
) -> list[int] | None:
    """Return AMSEL-ranked candidate states when the frontier API is available."""

    if not _AMSEL_AVAILABLE or not hasattr(_amsel, "rank_basin_frontier"):
        return None
    try:
        candidates = _amsel.rank_basin_frontier(
            candidate_states=[int(state) for state in candidate_states],
            guidance={int(state): float(score) for state, score in guidance.items()},
            event_families=event_families,
            closed_event_families=closed_event_families,
            duplicate_family_penalty=float(duplicate_family_penalty),
            min_guidance=float(min_guidance),
        )
    except Exception:  # noqa: BLE001 - basin exploration can fall back to local order.
        return None
    return [int(candidate.state) for candidate in candidates]


def _connectivity_df(connectivity_table: StatesConnectivity | Any) -> pd.DataFrame:
    df = getattr(connectivity_table, "df", connectivity_table)
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    required = {"state", "state_connexion", "k_forward"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df.copy()


def _guidance_problem(
    df: pd.DataFrame,
) -> tuple[
    list[int],
    list[int],
    list[tuple[int, int, float]],
    dict[int, list[int]],
]:
    transient = sorted({int(value) for value in df["state"].to_numpy()})
    transient_set = set(transient)
    all_state_ids = {
        int(value)
        for value in pd.concat([df["state"], df["state_connexion"]], ignore_index=True)
    }
    next_channel_id = max(all_state_ids) + 1
    rates: list[tuple[int, int, float]] = []
    absorbing: set[int] = set()
    state_targets: dict[int, list[int]] = defaultdict(list)

    for _, row in df.iterrows():
        source = int(row["state"])
        target = int(row["state_connexion"])
        rate = float(row["k_forward"])
        if _is_transient_destination(row, target, transient_set):
            rates.append((source, target, rate))
            state_targets[target].append(target)
            if target not in transient_set:
                absorbing.add(target)
            continue

        channel_id = next_channel_id
        next_channel_id += 1
        rates.append((source, channel_id, rate))
        absorbing.add(channel_id)
        state_targets[target].append(channel_id)

    return transient, sorted(absorbing), rates, dict(state_targets)


def _is_transient_destination(
    row: pd.Series,
    target: int,
    transient_set: set[int],
) -> bool:
    if "transient" in row.index:
        return bool(row["transient"])
    return int(target) in transient_set
