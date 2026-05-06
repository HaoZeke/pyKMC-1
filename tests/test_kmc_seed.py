import random
from types import SimpleNamespace

import numpy as np

from pykmc.kmc import KMC


def test_kmc_seeds_python_and_numpy_rngs_from_control_config():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))

    kmc._seed_rngs()

    expected_python = random.Random(12345).random()
    expected_numpy = np.random.RandomState(12345).random_sample()
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
