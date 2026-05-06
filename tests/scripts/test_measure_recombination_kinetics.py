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


def test_parse_pykmc_out_reads_numeric_rows(tmp_path):
    script = _load_script()
    path = tmp_path / "pykmc.out"
    path.write_text(
        "# Simulation Progress Tracking File\n"
        "Step       dT(s)          T(s)           Ref event      Ea(eV)        k_evt(ps-1)   k_tot(ps-1)   E(eV)         Cpu time(s)    Wall time(s)\n"
        "--------------------------------------------------------------------------------------------------------------\n"
        "1          1.000000e-12   1.000000e-12   3              4.000000e-01   2.000000e+00   5.000000e+00   -1.000000e+03  1.000000e-01   2.000000e-01\n"
    )

    assert script.parse_pykmc_out(path) == [
        {
            "step": 1,
            "dt_s": 1.0e-12,
            "time_s": 1.0e-12,
            "ref_event": 3,
            "ea_ev": 0.4,
            "k_evt_ps": 2.0,
            "k_tot_ps": 5.0,
            "energy_ev": -1000.0,
        }
    ]


def test_trial_row_uses_recombination_or_censor_time(tmp_path):
    script = _load_script()
    out_path = tmp_path / "pykmc.out"
    out_path.write_text(
        "1 1.0e-12 1.0e-12 3 0.4 2.0 5.0 -1000.0 0.1 0.2\n"
        "2 2.0e-12 3.0e-12 4 0.5 1.0 3.0 -999.0 0.2 0.3\n"
    )
    log_path = tmp_path / "pykmc.log"
    log_path.write_text(":=> Only atoms with cristalline environment\n")

    row = script.trial_row_from_outputs(
        case="ni-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["recombined"] is True
    assert row["t_recombination_s"] == 3.0e-12
    assert row["censored_time_s"] is None
    assert row["kmc_steps"] == 2


def test_survival_rows_are_plot_ready():
    script = _load_script()
    trials = [
        {
            "selector": "legacy",
            "recombined": True,
            "t_recombination_s": 1.0,
            "censored_time_s": None,
        },
        {
            "selector": "legacy",
            "recombined": False,
            "t_recombination_s": None,
            "censored_time_s": 2.0,
        },
        {
            "selector": "legacy",
            "recombined": True,
            "t_recombination_s": 3.0,
            "censored_time_s": None,
        },
    ]

    rows = script.survival_rows(trials)

    assert rows == [
        {
            "selector": "legacy",
            "time_s": 1.0,
            "n_at_risk": 3,
            "n_events": 1,
            "survival": 2 / 3,
        },
        {
            "selector": "legacy",
            "time_s": 3.0,
            "n_at_risk": 1,
            "n_events": 1,
            "survival": 0.0,
        },
    ]
