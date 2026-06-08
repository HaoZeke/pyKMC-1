import numpy as np

import pykmc.symmetries as symmetries


def test_unique_symmetries_uses_identity_when_sofi_probe_fails(monkeypatch):
    class FakeSymmetries:
        matrix = np.array([np.eye(3), -np.eye(3)])
        perm = np.array([np.array([0, 1]), np.array([1, 0])])

    class FakeSOFI:
        def compute(self, *_args):
            return FakeSymmetries()

    monkeypatch.setattr(
        symmetries,
        "_sofi_subprocess_available",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(symmetries.ira_mod, "SOFI", lambda: FakeSOFI())

    matrices, permutations = symmetries.unique_symmetries(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.0, 0.2, 0.0], [1.0, 0.0, 0.0]]),
        sym_thr=0.1,
    )

    assert matrices.shape == (1, 3, 3)
    assert permutations.shape == (1, 2)
    np.testing.assert_allclose(matrices[0], np.eye(3))
    np.testing.assert_array_equal(permutations[0], np.arange(2))
