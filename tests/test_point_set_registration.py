import numpy as np

import pykmc.point_set_registration as psr
from pykmc.result import ErrorType


def test_simple_ira_returns_error_when_native_matcher_is_unavailable(monkeypatch):
    class FakeIRA:
        def match(self, *_args):
            return np.eye(3), np.zeros(3), np.arange(1), 0.0

    monkeypatch.setattr(psr.ira_mod, "IRA", lambda: FakeIRA())
    monkeypatch.setattr(
        psr,
        "_ira_match_subprocess_available",
        lambda: False,
        raising=False,
    )

    result = psr.simple_ira(
        1,
        ["X"],
        np.zeros((1, 3), dtype=float),
        1,
        ["X"],
        np.zeros((1, 3), dtype=float),
        2.0,
    )

    assert not result.is_ok()
    assert result.err_value().type is ErrorType.PSR_NO_MATCH_FOUND
