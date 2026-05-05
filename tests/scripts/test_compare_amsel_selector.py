import importlib.util
import math
from pathlib import Path


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
    assert math.isfinite(report["elapsed_ns"])
    assert report["exit_state"] == 1
    assert report["t_exit"] > 0.0
