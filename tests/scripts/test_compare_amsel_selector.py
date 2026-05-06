import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest


def _load_script():
    script = Path(__file__).resolve().parents[2] / "scripts" / "compare_amsel_selector.py"
    spec = importlib.util.spec_from_file_location("compare_amsel_selector", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cycle_rng_replays_probability_draws():
    script = _load_script()
    rng = script.CycleRng([0.25, 0.75])

    assert [rng.random(), rng.random(), rng.random()] == [0.25, 0.75, 0.25]


def test_run_selector_report_captures_legacy_success():
    script = _load_script()
    table = script.single_exit_connectivity(rate=2.0)

    report = script.run_selector(
        case_name="single-exit",
        selector_name="legacy-fpta",
        selector_factory=script.legacy_fpta_selector,
        table=table,
        draws=[0.25, 0.0],
    )

    assert report["case"] == "single-exit"
    assert report["selector"] == "legacy-fpta"
    assert report["ok"] is True
    assert report["clock_semantics"] == "sampled-quantile"
    assert report["time_draw"] == pytest.approx(0.25)
    assert report["outlet_draw"] == pytest.approx(0.0)
    assert report["stable_absorption_probability"] == pytest.approx(0.25, rel=1e-3)
    assert report["time_draw_absorption_error"] == pytest.approx(0.0, abs=1e-3)
    assert report["master_equation_absorption_probability"] == pytest.approx(
        0.25, rel=1e-3
    )
    assert report["reduced_master_equation_absorption_probability"] == pytest.approx(
        0.25, rel=1e-3
    )
    assert report["master_reduced_absorption_error"] == pytest.approx(0.0, abs=1e-12)
    assert report["master_equation_outlet_quantile_hit"] is True
    assert math.isfinite(report["elapsed_ns"])
    assert report["exit_state"] == 1
    assert report["t_exit"] > 0.0


def test_master_equation_report_matches_sympy_two_exit_solution():
    sp = pytest.importorskip("sympy")
    script = _load_script()
    table = script.StatesConnectivity()
    table.df = pd.DataFrame(
        {
            "state": [0, 0],
            "state_connexion": [1, 2],
            "k_forward": [1.0, 3.0],
        }
    )

    a, b, t = sp.symbols("a b t", positive=True)
    generator = sp.Matrix(
        [
            [a + b, 0, 0],
            [-a, 0, 0],
            [-b, 0, 0],
        ]
    )
    p0 = sp.Matrix([1, 0, 0])
    p_t = sp.simplify((-generator * t).exp() * p0)
    subs = {a: sp.Integer(1), b: sp.Integer(3), t: sp.log(2) / 4}
    expected_absorbed = float(sp.simplify((p_t[1] + p_t[2]).subs(subs)))
    expected_exit_probability = float(
        sp.simplify((p_t[2] / (p_t[1] + p_t[2])).subs(subs))
    )

    report = script.master_equation_report(
        table=table,
        t_exit=float(math.log(2.0) / 4.0),
        time_draw=0.5,
        outlet_draw=0.75,
        exit_state=2,
    )

    assert report["master_equation_absorption_probability"] == pytest.approx(
        expected_absorbed
    )
    assert report["reduced_master_equation_absorption_probability"] == pytest.approx(
        expected_absorbed
    )
    assert report["master_reduced_absorption_error"] == pytest.approx(0.0)
    assert report["time_draw_master_equation_error"] == pytest.approx(0.0)
    assert report["master_equation_probability_sum"] == pytest.approx(1.0)
    assert report["master_equation_selected_exit_probability"] == pytest.approx(
        expected_exit_probability
    )
    assert report["master_equation_outlet_cdf_lower"] == pytest.approx(0.25)
    assert report["master_equation_outlet_cdf_upper"] == pytest.approx(1.0)
    assert report["master_equation_outlet_quantile_hit"] is True


def test_run_selector_report_labels_mean_clock_as_mfpt():
    pytest.importorskip("amsel")
    script = _load_script()
    table = script.single_exit_connectivity(rate=2.0)

    report = script.run_selector(
        case_name="single-exit",
        selector_name="amsel-mean",
        selector_factory=script.amsel_mean_selector,
        table=table,
        draws=[0.5, 0.0],
    )

    assert report["ok"] is True
    assert report["clock_mode"] == "mean"
    assert report["clock_semantics"] == "deterministic-mfpt"
    assert report["stable_absorption_probability"] is None
    assert report["time_draw_absorption_error"] is None
    assert report["t_exit"] == pytest.approx(0.5)


def test_feature_report_exposes_independent_amsel_diagnostics():
    pytest.importorskip("amsel")
    script = _load_script()
    table = script.single_exit_connectivity(rate=2.0)

    report = script.amsel_feature_report(table)

    assert report["ok"] is True
    assert report["mrm_moments"]["ok"] is True
    assert report["reduced_kinetics"]["ok"] is True
    assert report["ngt_outlets"][0]["ok"] is True
    assert report["reduced_kinetics"]["slow_subspace_rank"] == 1


def test_clock_reference_explains_rank1_mean_and_quantile_clocks():
    pytest.importorskip("amsel")
    script = _load_script()
    table = script.single_exit_connectivity(rate=2.0)
    features = script.amsel_feature_report(table)

    report = script.rank1_clock_reference(features=features, draws=[0.5, 0.0])

    assert report["ok"] is True
    assert report["effective_rate"] == pytest.approx(2.0)
    assert report["sampled_quantile_time"] == pytest.approx(math.log(2.0) / 2.0)
    assert report["mfpt"] == pytest.approx(0.5)
    assert report["mfpt_over_sampled_quantile_time"] == pytest.approx(1.0 / math.log(2.0))


def test_clock_reference_rejects_rate_inconsistent_with_mfpt():
    script = _load_script()

    report = script.rank1_clock_reference(
        features={
            "mrm_moments": {"ok": True, "mean": 10.0},
            "reduced_kinetics": {
                "ok": True,
                "effective_rate": 1.0e-30,
            },
        },
        draws=[0.5, 0.0],
    )

    assert report == {
        "ok": False,
        "error": "reduced effective rate is inconsistent with MFPT",
    }


def test_selector_payload_marks_live_source_and_table_size():
    pytest.importorskip("amsel")
    script = _load_script()
    table = script.single_exit_connectivity(rate=2.0)

    payload = script.selector_payload(
        table=table,
        case_name="live-cu",
        draws=[0.5, 0.0],
        entry=0,
        source="live-lammps-mpi",
    )

    assert payload["source"] == "live-lammps-mpi"
    assert payload["draws"] == [0.5, 0.0]
    assert payload["connectivity"]["rows"] == 1
    assert [item["selector"] for item in payload["selectors"]] == [
        "legacy-fpta",
        "amsel-sampled",
        "amsel-mean",
        "amsel-adaptive",
    ]
    assert [item["clock_semantics"] for item in payload["selectors"]] == [
        "sampled-quantile",
        "sampled-quantile",
        "deterministic-mfpt",
        "sampled-quantile",
    ]
    assert payload["selectors"][-1]["clock_mode"] == "reduced-sampled"
    assert payload["amsel_features"]["ok"] is True
    assert payload["clock_reference"]["ok"] is True
