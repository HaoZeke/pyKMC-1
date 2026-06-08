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
