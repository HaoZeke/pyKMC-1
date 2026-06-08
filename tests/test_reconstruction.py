from types import SimpleNamespace

import numpy as np

from pykmc import Reconstruction
from pykmc.config import BasinConfig


class RecordingManager:
    def __init__(self, minimized_positions):
        self.minimized_positions = list(minimized_positions)
        self.minimize_commands = []

    def global_minimize_with_results(self, config, positions=None):
        self.minimize_commands.append(config.lammps.minimize)
        return self.minimized_positions.pop(0).copy(), -1.0


def test_reconstruction_uses_explicit_minimize_command_without_mutating_config():
    min1 = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    min2 = np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]])
    saddle = np.array([[1.0, 1.0, 1.0], [2.5, 1.0, 1.0]])
    config = SimpleNamespace(
        lammps=SimpleNamespace(minimize="1e-10 1e-12 10000 10000"),
        psr=SimpleNamespace(matching_score_thr=0.1),
    )
    manager = RecordingManager([min1, min2])

    result = Reconstruction(config, manager).reconstruct(
        min1,
        min2,
        saddle,
        np.diag([10.0, 10.0, 10.0]),
        delr_thr=0.1,
        minimize_command="1.0e-6 1.0e-8 10 10",
    )

    assert result.is_ok()
    assert manager.minimize_commands == [
        "1.0e-6 1.0e-8 10 10",
        "1.0e-6 1.0e-8 10 10",
    ]
    assert config.lammps.minimize == "1e-10 1e-12 10000 10000"


def test_basin_reconstruction_uses_main_minimizer_by_default():
    assert BasinConfig().reconstruction_minimize is None
    assert BasinConfig().selector == "auto"
    assert BasinConfig().exploration_priority == "auto"
    assert BasinConfig().frontier_search_nevalf_max == 80
