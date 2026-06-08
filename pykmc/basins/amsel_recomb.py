"""amsel-driven barrierless recombination capture for vacancy/SIA pairs.

The recombined perfect crystal is a strong attractive sink: one side is a
defect pair, the other is a defect-free minimum at much lower energy. The
capture step (SIA dropping into the vacancy) is downhill with little or no
barrier, so a saddle / min-mode search (pARTn) never proposes it -- it
just rolls back downhill and reports "no event found". The kMC then only
ever sees the lateral migration hops and random-walks forever without
recombining.

This module supplies the missing transition directly, using amsel only:

  detect_recomb : CNA/coordination detect an over-coordinated SIA atom
                  within the capture radius of an under-coordinated
                  vacancy site (the recombinable topology).
  build_product : amsel.build_recomb_product_positions translates the SIA
                  filler onto the vacancy site -> the recombined geometry.

The caller relaxes that product; if it removes defects it is applied as a
single downhill, effectively absorbing kMC step (the attractive sink).
This is the pyKMC analogue of the eOn product-difference dimer seed --
make the capture transition available so the sink does the rest.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


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
    """Return the nearest SIA filler and vacancy centroid for a V/SIA topology."""
    try:
        from amsel import defect_clusters, estimate_nn_spacing
    except ImportError:
        return None
    pos = np.asarray(positions, dtype=float)
    celld = _cell_diag(cell)
    n = pos.shape[0]
    if n == 0:
        return None
    nn = estimate_nn_spacing(pos.tolist(), celld.tolist(), [], 6.0) or 2.56
    cutoff = cutoff_mult * nn
    cn = _coordination(pos, celld, cutoff)
    if cn.size == 0:
        return None
    bulk_cn = int(Counter(int(c) for c in cn).most_common(1)[0][0])
    under = [int(i) for i in np.where(cn < bulk_cn)[0]]
    over = [int(i) for i in np.where(cn > bulk_cn)[0]]
    if not under or not over:
        return None
    pos_list = pos.tolist()
    cell_list = celld.tolist()

    def _largest_cluster_centroid(idx_set):
        # Cluster only idx_set via amsel.defect_clusters (synthetic codes:
        # selected atoms -> non-bulk 8, rest -> bulk 1), take the largest
        # cluster -- the real defect (the vacancy ring / the SIA cage) --
        # NOT the mean of all deviant atoms, which a stray neighbour skews.
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
        return None
    # Nearest over-coordinated SIA atom to the vacancy site.
    over_arr = np.asarray(over, dtype=int)
    dd = pos[over_arr] - v_centroid
    for ax in range(3):
        if celld[ax] > 0:
            dd[:, ax] -= celld[ax] * np.round(dd[:, ax] / celld[ax])
    dist = np.sqrt((dd * dd).sum(axis=1))
    j = int(np.argmin(dist))
    return int(over_arr[j]), v_centroid.tolist(), float(dist[j]), float(nn)


def detect_recomb(
    positions, cell, cutoff_mult: float = 1.08, capture_mult: float = 1.6
):
    """Detect a recombinable V/SIA topology.

    Returns (source_sia_atom_index, vacancy_centroid_xyz) when an
    over-coordinated SIA atom sits within ``capture_mult * nn`` of the
    under-coordinated vacancy site, else None. amsel-only (estimate_nn
    + coordination; CNA-tolerant point defects are caught by the
    coordination deviation).
    """
    topology = _nearest_recomb_topology(positions, cell, cutoff_mult=cutoff_mult)
    if topology is None:
        return None
    source_atom, v_centroid, distance, nn = topology
    if distance > capture_mult * nn:
        return None
    return int(source_atom), v_centroid


def recombination_search_center(positions, cell, cutoff_mult: float = 1.08):
    """Return the SIA filler atom that should receive a recombination seed."""
    topology = _nearest_recomb_topology(positions, cell, cutoff_mult=cutoff_mult)
    if topology is None:
        return None
    source_atom, _v_centroid, _distance, _nn = topology
    return int(source_atom)


def build_product(positions, cell, source_atom: int, target_centroid):
    """Recombined-product geometry via amsel.build_recomb_product_positions:
    translate the SIA filler atom onto the vacancy site."""
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
    capture_mult: float = 1.6,
):
    """Per-atom initial-push array (nat, 3) seeding a pARTn search ALONG
    the recombination direction, or None when this central atom is not a
    recombinable SIA filler.

    The push is the min-image displacement reactant -> recombined product
    (from amsel.build_recomb_product_positions), restricted to the atoms
    that actually move and rescaled so the largest displacement equals
    push_step_size. Fed to pARTn via set("push", ...) (push_mode=input),
    it points the min-mode search straight at the attractive recombined
    sink instead of a random direction. numpy (nat, 3) C-order maps to
    ARTn's fortran (3, nat).
    """
    det = _nearest_recomb_topology(positions, cell, cutoff_mult=cutoff_mult)
    if det is None:
        return None
    source_atom, v_centroid, distance, nn = det
    if distance > capture_mult * nn:
        return None
    # Only seed when the search is centred on the SIA filler itself.
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
    nn = estimate_nn_spacing(pos.tolist(), celld.tolist(), [], 6.0) or 2.56
    cn = _coordination(pos, celld, cutoff_mult * nn)
    if cn.size == 0:
        return 0
    bulk_cn = int(Counter(int(c) for c in cn).most_common(1)[0][0])
    return int((cn != bulk_cn).sum())
