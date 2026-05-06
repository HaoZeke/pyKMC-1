import importlib.util
from pathlib import Path


def _load_script():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "measure_recombination_kinetics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "measure_recombination_kinetics",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detect_recombination_from_log_marks_crystal_termination():
    script = _load_script()

    result = script.detect_recombination_from_log(
        "Step : 4\n:=> Only atoms with cristalline environment\n:=> End of simulation\n"
    )

    assert result == {
        "recombined": True,
        "detector_reason": "all-crystal-environments",
    }


def test_detect_recombination_from_log_marks_censored_run():
    script = _load_script()

    result = script.detect_recombination_from_log("Step : 4\n:=> End of simulation\n")

    assert result == {
        "recombined": False,
        "detector_reason": "censored",
    }
