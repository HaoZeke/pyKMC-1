import numpy as np

from pykmc.activevolume.active_volume import active_volume_atom_index


def test_active_volume_atom_index_returns_scalar_position():
    atom_map = np.array([4, 9, 12], dtype=int)

    assert active_volume_atom_index(atom_map, 9) == 1
