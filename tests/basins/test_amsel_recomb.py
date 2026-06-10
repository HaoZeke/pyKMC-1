from types import SimpleNamespace

import numpy as np

from pykmc.basins import amsel_recomb


def test_recomb_push_requires_capture_radius(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 20.0

    monkeypatch.setattr(
        amsel_recomb,
        "_nearest_recomb_topology",
        lambda positions, cell, cutoff_mult=1.08: (1, [0.0, 0.0, 0.0], 4.0, 2.0),
    )

    def fake_product(positions, cell, source_atom, target_centroid):
        product = np.asarray(positions, dtype=float).copy()
        product[int(source_atom)] = np.asarray(target_centroid, dtype=float)
        return product

    monkeypatch.setattr(amsel_recomb, "build_product", fake_product)

    push = amsel_recomb.recomb_push(
        positions,
        cell,
        central_atom_idx=1,
        capture_mult=1.5,
    )

    assert push is None


def test_recomb_push_requires_defect_reducing_product(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 20.0

    monkeypatch.setattr(
        amsel_recomb,
        "_nearest_recomb_topology",
        lambda positions, cell, cutoff_mult=1.08: (1, [0.0, 0.0, 0.0], 0.2, 2.0),
    )

    def fake_product(positions, cell, source_atom, target_centroid):
        return np.asarray(positions, dtype=float).copy()

    monkeypatch.setattr(amsel_recomb, "build_product", fake_product)
    monkeypatch.setattr(amsel_recomb, "n_defects", lambda positions, cell: 4)

    push = amsel_recomb.recomb_push(
        positions,
        cell,
        central_atom_idx=1,
        capture_mult=1.5,
    )

    assert push is None


def test_recombination_search_center_requires_capture_radius(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 20.0

    monkeypatch.setattr(
        amsel_recomb,
        "_nearest_recomb_topology",
        lambda positions, cell, cutoff_mult=1.08: (1, [0.0, 0.0, 0.0], 4.0, 2.0),
    )

    assert (
        amsel_recomb.recombination_search_center(
            positions,
            cell,
            capture_mult=1.5,
        )
        is None
    )


def test_recombination_search_center_defaults_to_local_candidate(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 20.0

    monkeypatch.setattr(
        amsel_recomb,
        "_nearest_recomb_topology",
        lambda positions, cell, cutoff_mult=1.08: (1, [0.0, 0.0, 0.0], 4.0, 2.0),
    )

    assert amsel_recomb.recombination_search_center(positions, cell) == 1


def test_recombination_search_center_requires_defect_reduction(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 20.0

    monkeypatch.setattr(
        amsel_recomb,
        "_nearest_recomb_topology",
        lambda positions, cell, cutoff_mult=1.08: (1, [0.0, 0.0, 0.0], 0.2, 2.0),
    )

    def fake_product(positions, cell, source_atom, target_centroid):
        return np.asarray(positions, dtype=float).copy()

    monkeypatch.setattr(amsel_recomb, "build_product", fake_product)
    monkeypatch.setattr(amsel_recomb, "n_defects", lambda positions, cell: 4)

    assert amsel_recomb.recombination_search_center(positions, cell) is None


def test_nearest_recomb_topology_uses_generic_amsel_candidate(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 10.0
    candidate = SimpleNamespace(
        source_atom=1,
        target_centroid=(0.25, 0.0, 0.0),
        distance=0.75,
        nn_spacing=1.0,
    )

    monkeypatch.setattr(
        amsel_recomb,
        "_amsel_defect_annihilation_candidate",
        lambda positions, cell, cutoff_mult=1.08: candidate,
    )

    assert amsel_recomb._nearest_recomb_topology(positions, cell) == (
        1,
        [0.25, 0.0, 0.0],
        0.75,
        1.0,
    )


def test_nearest_recomb_topology_scores_source_cluster_by_product_defects(
    monkeypatch,
):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.9, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 10.0
    candidate = SimpleNamespace(
        source_atom=1,
        source_cluster=(1, 2),
        target_centroid=(0.0, 0.0, 0.0),
        distance=0.2,
        nn_spacing=1.0,
    )

    def fake_product(positions, cell, source_atom, target_centroid):
        product = np.asarray(positions, dtype=float).copy()
        if int(source_atom) == 2:
            product[int(source_atom)] = np.asarray(target_centroid, dtype=float)
        return product

    def fake_n_defects(positions, cell):
        if np.allclose(np.asarray(positions, dtype=float)[2], [0.0, 0.0, 0.0]):
            return 2
        return 6

    monkeypatch.setattr(
        amsel_recomb,
        "_amsel_defect_annihilation_candidate",
        lambda positions, cell, cutoff_mult=1.08: candidate,
    )
    monkeypatch.setattr(amsel_recomb, "build_product", fake_product)
    monkeypatch.setattr(amsel_recomb, "n_defects", fake_n_defects)
    amsel_recomb._TOPOLOGY_CACHE.clear()

    assert amsel_recomb._nearest_recomb_topology(positions, cell) == (
        2,
        [0.0, 0.0, 0.0],
        0.9,
        1.0,
    )


def test_nearest_recomb_topology_scores_target_shell_projection(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 10.0
    candidate = SimpleNamespace(
        source_atom=1,
        source_cluster=(1,),
        target_cluster=(0,),
        target_centroid=(5.0, 0.0, 0.0),
        distance=3.0,
        nn_spacing=1.0,
    )

    def fake_product(positions, cell, source_atom, target_centroid):
        product = np.asarray(positions, dtype=float).copy()
        product[int(source_atom)] = np.asarray(target_centroid, dtype=float)
        return product

    def fake_n_defects(positions, cell):
        source = np.asarray(positions, dtype=float)[1]
        if np.allclose(source, [1.0, 0.0, 0.0]):
            return 2
        return 6

    monkeypatch.setattr(
        amsel_recomb,
        "_amsel_defect_annihilation_candidate",
        lambda positions, cell, cutoff_mult=1.08: candidate,
    )
    monkeypatch.setattr(amsel_recomb, "build_product", fake_product)
    monkeypatch.setattr(amsel_recomb, "n_defects", fake_n_defects)
    amsel_recomb._TOPOLOGY_CACHE.clear()

    assert amsel_recomb._nearest_recomb_topology(positions, cell) == (
        1,
        [1.0, 0.0, 0.0],
        1.0,
        1.0,
    )


def test_nearest_recomb_topology_uses_geometric_target_for_large_systems(
    monkeypatch,
):
    positions = np.zeros((129, 3), dtype=float)
    positions[1] = [2.0, 0.0, 0.0]
    cell = np.eye(3) * 20.0
    candidate = SimpleNamespace(
        source_atom=1,
        source_cluster=(1,),
        target_cluster=(0,),
        target_centroid=(5.0, 0.0, 0.0),
        distance=3.0,
        nn_spacing=1.0,
    )

    def fail_n_defects(positions, cell):
        raise AssertionError("large-system topology lookup must not score products")

    monkeypatch.setattr(
        amsel_recomb,
        "_amsel_defect_annihilation_candidate",
        lambda positions, cell, cutoff_mult=1.08: candidate,
    )
    monkeypatch.setattr(amsel_recomb, "n_defects", fail_n_defects)
    amsel_recomb._TOPOLOGY_CACHE.clear()

    assert amsel_recomb._nearest_recomb_topology(positions, cell) == (
        1,
        [5.0, 0.0, 0.0],
        3.0,
        1.0,
    )


def test_nearest_recomb_topology_keeps_amsel_target_when_unscored_projection_is_noop(
    monkeypatch,
):
    positions = np.zeros((129, 3), dtype=float)
    positions[1] = [2.0, 0.0, 0.0]
    cell = np.eye(3) * 20.0
    candidate = SimpleNamespace(
        source_atom=1,
        source_cluster=(1,),
        target_cluster=(0,),
        target_centroid=(5.0, 0.0, 0.0),
        distance=3.0,
        nn_spacing=1.0,
    )

    def fail_n_defects(positions, cell):
        raise AssertionError("large-system topology lookup must not score products")

    monkeypatch.setattr(
        amsel_recomb,
        "_amsel_defect_annihilation_candidate",
        lambda positions, cell, cutoff_mult=1.08: candidate,
    )
    monkeypatch.setattr(amsel_recomb, "n_defects", fail_n_defects)
    amsel_recomb._TOPOLOGY_CACHE.clear()

    assert amsel_recomb._nearest_recomb_topology(positions, cell) == (
        1,
        [5.0, 0.0, 0.0],
        3.0,
        1.0,
    )


def test_nearest_recomb_topology_reuses_position_cache(monkeypatch):
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 10.0
    candidate = SimpleNamespace(
        source_atom=1,
        target_centroid=(0.25, 0.0, 0.0),
        distance=0.75,
        nn_spacing=1.0,
    )
    calls = []

    def fake_candidate(positions, cell, cutoff_mult=1.08):
        calls.append(1)
        return candidate

    monkeypatch.setattr(
        amsel_recomb,
        "_amsel_defect_annihilation_candidate",
        fake_candidate,
    )
    amsel_recomb._TOPOLOGY_CACHE.clear()

    assert amsel_recomb._nearest_recomb_topology(positions, cell) == (
        1,
        [0.25, 0.0, 0.0],
        0.75,
        1.0,
    )
    assert amsel_recomb._nearest_recomb_topology(positions.copy(), cell.copy()) == (
        1,
        [0.25, 0.0, 0.0],
        0.75,
        1.0,
    )
    assert len(calls) == 1
