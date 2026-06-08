"""amsel-driven local defect-annihilation hints.

This module identifies a topology with an over-coordinated source atom and
an under-coordinated target void. The same candidate can be used in two
ways: direct downhill injection after product minimization validates a
defect-removing sink, or a directed pARTn initial push that lets the saddle
search test the local annihilation path without material-specific labels.
"""
from __future__ import annotations

import hashlib
from collections import Counter, OrderedDict

import numpy as np


_TOPOLOGY_CACHE_MAX = 16
_TOPOLOGY_CACHE: OrderedDict[
    tuple[tuple[int, ...], bytes, float, bytes],
    tuple[int, tuple[float, float, float], float, float] | None,
] = OrderedDict()


def _topology_cache_key(positions, cell, cutoff_mult: float):
    pos = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
    celld = np.ascontiguousarray(_cell_diag(cell).astype(np.float64))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(pos.tobytes())
    digest.update(celld.tobytes())
    return tuple(pos.shape), celld.tobytes(), float(cutoff_mult), digest.digest()


def _cacheable_topology(topology):
    if topology is None:
        return None
    source_atom, target_centroid, distance, nn = topology
    return (
        int(source_atom),
        tuple(float(x) for x in target_centroid),
        float(distance),
        float(nn),
    )


def _public_topology(topology):
    if topology is None:
        return None
    source_atom, target_centroid, distance, nn = topology
    return int(source_atom), list(target_centroid), float(distance), float(nn)


def _remember_topology(key, topology):
    _TOPOLOGY_CACHE[key] = _cacheable_topology(topology)
    _TOPOLOGY_CACHE.move_to_end(key)
    while len(_TOPOLOGY_CACHE) > _TOPOLOGY_CACHE_MAX:
        _TOPOLOGY_CACHE.popitem(last=False)


def _amsel_defect_annihilation_candidate(positions, cell, cutoff_mult: float = 1.08):
    try:
        from amsel import defect_annihilation_candidate
    except ImportError:
        return None
    try:
        return defect_annihilation_candidate(
            np.asarray(positions, dtype=float).tolist(),
            _cell_diag(cell).tolist(),
            cutoff_mult=cutoff_mult,
        )
    except Exception:
        return None


def _cell_diag(cell) -> np.ndarray:
    cell = np.asarray(cell, dtype=float)
    return np.diag(cell) if cell.ndim == 2 else cell


def _coordination(pos: np.ndarray, cell: np.ndarray, cutoff: float) -> np.ndarray:
    n = pos.shape[0]
    c2 = cutoff * cutoff
    cn = np.zeros(n, dtype=int)
    for i in range(n):
        d = pos - pos[i]
        for ax in range(3):
            if cell[ax] > 0:
                d[:, ax] -= cell[ax] * np.round(d[:, ax] / cell[ax])
        cn[i] = int(((d * d).sum(axis=1) < c2).sum()) - 1
    return cn


def _nearest_recomb_topology(positions, cell, cutoff_mult: float = 1.08):
    """Return the nearest source atom and target void for local annihilation."""
    key = _topology_cache_key(positions, cell, cutoff_mult)
    if key in _TOPOLOGY_CACHE:
        _TOPOLOGY_CACHE.move_to_end(key)
        return _public_topology(_TOPOLOGY_CACHE[key])
    candidate = _amsel_defect_annihilation_candidate(
        positions, cell, cutoff_mult=cutoff_mult
    )
    if candidate is not None:
        topology = (
            int(candidate.source_atom),
            list(candidate.target_centroid),
            float(candidate.distance),
            float(candidate.nn_spacing),
        )
        _remember_topology(key, topology)
        return _public_topology(_TOPOLOGY_CACHE[key])
    try:
        from amsel import defect_clusters, estimate_nn_spacing
    except ImportError:
        _remember_topology(key, None)
        return None
    pos = np.asarray(positions, dtype=float)
    celld = _cell_diag(cell)
    n = pos.shape[0]
    if n == 0:
        _remember_topology(key, None)
        return None
    nn = estimate_nn_spacing(pos.tolist(), celld.tolist(), [], 6.0)
    if nn is None or nn <= 0.0:
        _remember_topology(key, None)
        return None
    cutoff = cutoff_mult * nn
    cn = _coordination(pos, celld, cutoff)
    if cn.size == 0:
        _remember_topology(key, None)
        return None
    bulk_cn = int(Counter(int(c) for c in cn).most_common(1)[0][0])
    under = [int(i) for i in np.where(cn < bulk_cn)[0]]
    over = [int(i) for i in np.where(cn > bulk_cn)[0]]
    if not under or not over:
        _remember_topology(key, None)
        return None
    pos_list = pos.tolist()
    cell_list = celld.tolist()

    def _largest_cluster_centroid(idx_set):
        # Cluster only idx_set via amsel.defect_clusters (synthetic codes:
        # selected atoms -> non-bulk 8, rest -> bulk 1), then take the
        # largest connected component so stray defect-signal atoms do not
        # skew the local target/source geometry.
        codes = [1] * n
        for i in idx_set:
            codes[i] = 8
        clusters = defect_clusters(pos_list, cell_list, idx_set, cutoff, codes, 1)
        if not clusters:
            return None, None
        # defect_clusters returns members as indices INTO idx_set (local),
        # not global atom indices -- map them back through idx_set.
        local = max(clusters, key=len)
        biggest = [int(idx_set[k]) for k in local]
        cpos = pos[np.asarray(biggest, dtype=int)]
        ref = cpos[0]
        d = cpos - ref
        for ax in range(3):
            if celld[ax] > 0:
                d[:, ax] -= celld[ax] * np.round(d[:, ax] / celld[ax])
        return ref + d.mean(axis=0), biggest

    v_centroid, _v_cluster = _largest_cluster_centroid(under)
    if v_centroid is None:
        _remember_topology(key, None)
        return None
    # Nearest over-coordinated source atom to the target void.
    over_arr = np.asarray(over, dtype=int)
    dd = pos[over_arr] - v_centroid
    for ax in range(3):
        if celld[ax] > 0:
            dd[:, ax] -= celld[ax] * np.round(dd[:, ax] / celld[ax])
    dist = np.sqrt((dd * dd).sum(axis=1))
    j = int(np.argmin(dist))
    topology = int(over_arr[j]), v_centroid.tolist(), float(dist[j]), float(nn)
    _remember_topology(key, topology)
    return _public_topology(_TOPOLOGY_CACHE[key])


def detect_recomb(
    positions, cell, cutoff_mult: float = 1.08, capture_mult: float = 1.6
):
    """Detect a local defect-annihilation topology inside direct-capture range.

    Returns (source_atom_index, target_centroid_xyz) when an over-coordinated
    source atom sits within ``capture_mult * nn`` of the under-coordinated
    target void, else None.
    """
    topology = _nearest_recomb_topology(positions, cell, cutoff_mult=cutoff_mult)
    if topology is None:
        return None
    source_atom, v_centroid, distance, nn = topology
    if distance > capture_mult * nn:
        return None
    return int(source_atom), v_centroid


def recombination_search_center(
    positions,
    cell,
    cutoff_mult: float = 1.08,
    capture_mult: float | None = None,
):
    """Return the source atom that should receive a directed search seed."""
    topology = _nearest_recomb_topology(positions, cell, cutoff_mult=cutoff_mult)
    if topology is None:
        return None
    source_atom, _v_centroid, distance, nn = topology
    if capture_mult is not None and distance > capture_mult * nn:
        return None
    return int(source_atom)


def build_product(positions, cell, source_atom: int, target_centroid):
    """Product geometry from source-to-target translation via amsel."""
    from amsel import build_recomb_product_positions

    celld = _cell_diag(cell).tolist()
    return build_recomb_product_positions(
        np.asarray(positions, dtype=float).tolist(),
        celld,
        [int(source_atom)],
        [list(target_centroid)],
    )


def recomb_push(
    positions,
    cell,
    central_atom_idx: int,
    push_step_size: float = 0.1,
    cutoff_mult: float = 1.08,
    capture_mult: float | None = None,
    topology=None,
):
    """Per-atom initial-push array (nat, 3) seeding a pARTn search ALONG
    the local annihilation direction, or None when this central atom is not
    the source atom.

    The push is the min-image displacement reactant -> recombined product
    (from amsel.build_recomb_product_positions), restricted to the atoms
    that actually move and rescaled so the largest displacement equals
    push_step_size. Fed to pARTn via set("push", ...) (push_mode=input),
    it points the min-mode search at the topology-suggested product instead
    of a random direction. numpy (nat, 3) C-order maps to ARTn's fortran
    (3, nat).
    """
    det = topology
    if det is None:
        det = _nearest_recomb_topology(positions, cell, cutoff_mult=cutoff_mult)
    if det is None:
        return None
    source_atom, v_centroid, distance, nn = det
    if capture_mult is not None and distance > capture_mult * nn:
        return None
    # Only seed when the search is centred on the selected source atom.
    if int(central_atom_idx) != int(source_atom):
        return None
    product = np.asarray(
        build_product(positions, cell, source_atom, v_centroid), dtype=float
    )
    pos = np.asarray(positions, dtype=float)
    celld = _cell_diag(cell)
    disp = product - pos
    for ax in range(3):
        if celld[ax] > 0:
            disp[:, ax] -= celld[ax] * np.round(disp[:, ax] / celld[ax])
    norm = np.sqrt((disp * disp).sum(axis=1))
    dmax = float(norm.max())
    if dmax <= 1e-6:
        return None
    return disp * (push_step_size / dmax)


def n_defects(positions, cell, cutoff_mult: float = 1.08) -> int:
    """Count atoms whose coordination deviates from bulk (defect signal)."""
    try:
        from amsel import estimate_nn_spacing
    except ImportError:
        return -1
    pos = np.asarray(positions, dtype=float)
    celld = _cell_diag(cell)
    nn = estimate_nn_spacing(pos.tolist(), celld.tolist(), [], 6.0)
    if nn is None or nn <= 0.0:
        return -1
    cn = _coordination(pos, celld, cutoff_mult * nn)
    if cn.size == 0:
        return 0
    bulk_cn = int(Counter(int(c) for c in cn).most_common(1)[0][0])
    return int((cn != bulk_cn).sum())
