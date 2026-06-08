import numpy as np
import pandas as pd
from types import SimpleNamespace

import pykmc.point_set_registration as psr
from pykmc.point_set_registration import PointSetRegistration
from pykmc.result import Err, ErrorInfo
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


def test_point_set_registration_uses_translation_fallback_without_ira(monkeypatch):
    monkeypatch.setattr(
        psr,
        "simple_ira",
        lambda *_args, **_kwargs: Err(
            ErrorInfo(
                type=ErrorType.PSR_NO_MATCH_FOUND,
                message="native matcher unavailable",
            )
        ),
    )
    config = SimpleNamespace(
        psr=SimpleNamespace(style="ira", matching_score_thr=0.4),
        ira=SimpleNamespace(kmax_factor=2.0),
    )
    system = SimpleNamespace(
        positions=np.array([[1.0, 2.0, 3.0], [1.2, 2.0, 3.0]], dtype=float),
        cell=np.eye(3) * 10.0,
    )
    dfevent = pd.Series(
        {
            "initial_positions": np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                dtype=float,
            )
        }
    )
    neighbors_list = SimpleNamespace(get_neighbors=lambda _style, _idx: [0, 1])

    result = PointSetRegistration(
        config,
        system,
        dfevent,
        neighbors_list,
        central_atom_index=0,
    ).match()

    assert result.is_ok()
    np.testing.assert_allclose(result.ok_value().rotation_matrix, np.eye(3))
    np.testing.assert_allclose(result.ok_value().translation_matrix, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.ok_value().permutation_matrix, np.arange(2))
    assert result.ok_value().matching_score == 0.0


def test_point_set_registration_fallback_assigns_permuted_typed_neighbors(
    monkeypatch,
):
    calls = []

    def unavailable_ira(nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor):
        calls.append((list(typ1), list(typ2)))
        return Err(
            ErrorInfo(
                type=ErrorType.PSR_NO_MATCH_FOUND,
                message="native matcher unavailable",
            )
        )

    monkeypatch.setattr(psr, "simple_ira", unavailable_ira)
    config = SimpleNamespace(
        psr=SimpleNamespace(style="ira", matching_score_thr=0.1),
        ira=SimpleNamespace(kmax_factor=2.0),
    )
    system = SimpleNamespace(
        positions=np.array([[1.2, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=float),
        types=np.array(["O", "H"]),
        cell=np.eye(3) * 10.0,
    )
    dfevent = pd.Series(
        {
            "initial_positions": np.array(
                [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]],
                dtype=float,
            ),
            "types": np.array(["H", "O"]),
        }
    )
    neighbors_list = SimpleNamespace(get_neighbors=lambda _style, _idx: [0, 1])

    result = PointSetRegistration(
        config,
        system,
        dfevent,
        neighbors_list,
        central_atom_index=0,
    ).match()

    assert calls == [(["O", "H"], ["H", "O"])]
    assert result.is_ok()
    np.testing.assert_allclose(result.ok_value().translation_matrix, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result.ok_value().permutation_matrix, [1, 0])
    assert result.ok_value().matching_score == 0.0
