import random
from types import SimpleNamespace

import numpy as np

from pykmc.kmc import KMC, basin_exploration_trace_line


def test_kmc_seeds_python_and_numpy_rngs_from_control_config():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))

    kmc._seed_rngs()

    expected_python = random.Random(12345).random()
    expected_numpy = np.random.RandomState(12345).random_sample()
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy


def test_central_atoms_research_covers_distinct_atoms_before_resampling():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-a", "env-a", "crystal"]
    )
    random.seed(0)

    central_atoms = kmc.central_atoms_research(["env-a"], nsearch=3)

    assert sorted(central_atoms) == [0, 1, 2]


def test_basin_exploration_trace_line_reports_order_queue_and_guidance():
    basin = SimpleNamespace(
        exploration_order=[0, 13],
        states_to_explore=[14, 1],
        last_exploration_guidance={14: 0.375, 1: 0.25},
        exploration_decisions=[
            {
                "state": 0,
                "event_family": None,
                "process_signature": None,
                "guidance": 0.0,
            },
            {
                "state": 13,
                "event_family": 1,
                "process_signature": "1:3330",
                "guidance": 0.375,
            },
        ],
    )

    assert basin_exploration_trace_line(basin) == (
        "\t :=> Basin exploration trace order=0,13; "
        "queue=14,1; guidance=14:3.750000e-01,1:2.500000e-01; "
        "closed_events=0:NA,13:1; "
        "closed_processes=0:NA,13:1:3330; "
        "closed_guidance=0:0.000000e+00,13:3.750000e-01"
    )


def test_basin_exploration_trace_line_skips_empty_trace():
    assert basin_exploration_trace_line(SimpleNamespace()) is None
