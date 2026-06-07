import configparser
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


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


def test_detect_recombination_from_log_marks_zero_event_discovery():
    script = _load_script()

    result = script.detect_recombination_from_log(
        "Step : 1\n"
        "No events have been found, empty reference events table.\n"
        ":=> End of simulation\n"
    )

    assert result == {
        "recombined": False,
        "detector_reason": "zero-event-discovery",
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
            "cpu_time_s": 0.1,
            "wall_time_s": 0.2,
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
    assert row["cpu_time_s"] == 0.2
    assert row["wall_time_s"] == 0.3
    assert row["total_cpu_time_s"] == pytest.approx(0.3)
    assert row["total_wall_time_s"] == pytest.approx(0.5)
    assert row["failed_refinements"] == 0
    assert row["failed_refinement_committor"] == 0.0
    assert row["usable_resolved_committor"] is None
    assert row["kinetic_claim_ok"] is True


def test_trial_row_uses_trajectory_all_crystal_when_log_is_censored(
    tmp_path, monkeypatch
):
    script = _load_script()
    (tmp_path / "pykmc.out").write_text(
        "1 1.0e-12 1.0e-12 3 0.4 2.0 5.0 -1000.0 0.1 0.2\n"
    )
    (tmp_path / "pykmc.log").write_text("Step : 1\n:=> End of simulation\n")
    (tmp_path / "trajkmc.xyz").write_text("trajectory placeholder\n")
    monkeypatch.setattr(script, "trajectory_noncrystal_counts", lambda path: [28, 0])

    row = script.trial_row_from_outputs(
        case="ni-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["recombined"] is True
    assert row["t_recombination_s"] == 1.0e-12
    assert row["censored_time_s"] is None
    assert row["detector_reason"] == "trajectory-all-crystal"
    assert row["final_noncrystal_atoms"] == 0
    assert row["min_noncrystal_atoms"] == 0
    assert row["trajectory_recombination_frame"] == 1
    assert row["kinetic_claim_ok"] is True


def test_trial_row_records_temperature_and_box_volume(tmp_path, monkeypatch):
    script = _load_script()
    initial_config = tmp_path / "initial.xyz"
    initial_config.write_text("placeholder\n")
    (tmp_path / "input.in").write_text(
        "[Control]\n"
        f"initial_config = {initial_config}\n"
        "[RateConstant]\n"
        "T = 650.0\n"
    )
    (tmp_path / "pykmc.out").write_text(
        "1 1.0e-12 1.0e-12 3 0.4 2.0 5.0 -1000.0 0.1 0.2\n"
    )
    (tmp_path / "pykmc.log").write_text("Step : 1\n:=> End of simulation\n")
    monkeypatch.setattr(script, "structure_volume_A3", lambda path: 128.0)

    row = script.trial_row_from_outputs(
        case="cu-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["temperature_K"] == 650.0
    assert row["box_volume_A3"] == 128.0


def test_trial_row_records_rate_model_provenance(tmp_path, monkeypatch):
    script = _load_script()
    initial_config = tmp_path / "initial.xyz"
    initial_config.write_text("placeholder\n")
    (tmp_path / "input.in").write_text(
        "[Control]\n"
        f"initial_config = {initial_config}\n"
        "[RateConstant]\n"
        "style = amsel-vtst\n"
        "T = 500.0\n"
        "prefactor = 6.0e12\n"
        "saddle_freq_invcm = 120.0\n"
        "friction_inv_s = 0.0\n"
        "barrier_omega_rad_per_s = 2.0e13\n"
    )
    (tmp_path / "pykmc.out").write_text(
        "1 1.0e-12 1.0e-12 3 0.4 2.0 5.0 -1000.0 0.1 0.2\n"
    )
    (tmp_path / "pykmc.log").write_text("Step : 1\n:=> End of simulation\n")
    monkeypatch.setattr(script, "structure_volume_A3", lambda path: 128.0)

    row = script.trial_row_from_outputs(
        case="cu-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["rate_style"] == "amsel-vtst"
    assert row["rate_prefactor_inv_s"] == pytest.approx(6.0e12)
    assert row["rate_prefactor_source"] == "rateconstant.prefactor"
    assert row["rate_anharmonic_corrections_active"] is True
    assert row["rate_model_ok"] is False
    assert row["rate_model_reason"] == "configured-prefactor"


def test_trial_row_accepts_event_vineyard_rate_model(tmp_path, monkeypatch):
    script = _load_script()
    initial_config = tmp_path / "initial.xyz"
    initial_config.write_text("placeholder\n")
    (tmp_path / "input.in").write_text(
        "[Control]\n"
        f"initial_config = {initial_config}\n"
        "[RateConstant]\n"
        "style = amsel-vtst\n"
        "T = 500.0\n"
        "prefactor = 6.0e12\n"
        "compute_vineyard_prefactor = True\n"
    )
    (tmp_path / "pykmc.out").write_text(
        "1 1.0e-12 1.0e-12 3 0.4 2.0 5.0 -1000.0 0.1 0.2\n"
    )
    (tmp_path / "pykmc.log").write_text("Step : 1\n:=> End of simulation\n")
    monkeypatch.setattr(script, "structure_volume_A3", lambda path: 128.0)

    row = script.trial_row_from_outputs(
        case="cu-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["rate_style"] == "amsel-vtst"
    assert row["rate_prefactor_inv_s"] == pytest.approx(6.0e12)
    assert row["rate_prefactor_source"] == "vineyard-finite-difference"
    assert row["rate_anharmonic_corrections_active"] is True
    assert row["rate_model_ok"] is True
    assert row["rate_model_reason"] == "event-prefactor-vtst"


def test_trial_row_records_artn_curvature_prefactor_from_log(tmp_path, monkeypatch):
    script = _load_script()
    initial_config = tmp_path / "initial.xyz"
    initial_config.write_text("placeholder\n")
    (tmp_path / "input.in").write_text(
        "[Control]\n"
        f"initial_config = {initial_config}\n"
        "[RateConstant]\n"
        "style = amsel-vtst\n"
        "T = 300.0\n"
        "prefactor = 5.0e12\n"
        "compute_vineyard_prefactor = True\n"
    )
    (tmp_path / "pykmc.out").write_text(
        "1 6.5e-13 6.5e-13 0 0.1 0.10 0.58 -1000.0 0.1 0.2\n"
    )
    (tmp_path / "pykmc.log").write_text(
        "Step : 1\n"
        "ARTn saddle curvature prefactor for event at atom 8787\n"
        "Vineyard prefactor event at atom 8787 complete: "
        "forward=6.250985e+12/s backward=6.250985e+12/s "
        "saddle_freq=4.556903e+01/cm\n"
        ":=> End of simulation\n"
    )
    monkeypatch.setattr(script, "structure_volume_A3", lambda path: 128.0)

    row = script.trial_row_from_outputs(
        case="cu-vac-sia",
        selector="amsel",
        trial=0,
        seed=4100,
        output_dir=tmp_path,
    )

    assert row["rate_prefactor_inv_s"] == pytest.approx(6.250985e12)
    assert row["rate_prefactor_source"] == "thermal-tst-artn-curvature"
    assert row["rate_anharmonic_corrections_active"] is True
    assert row["rate_model_ok"] is True
    assert row["rate_model_reason"] == "thermal-tst-artn-curvature"


def test_trial_row_rejects_failed_event_vineyard_rate_model(tmp_path, monkeypatch):
    script = _load_script()
    initial_config = tmp_path / "initial.xyz"
    initial_config.write_text("placeholder\n")
    (tmp_path / "input.in").write_text(
        "[Control]\n"
        f"initial_config = {initial_config}\n"
        "[RateConstant]\n"
        "style = amsel-vtst\n"
        "T = 500.0\n"
        "prefactor = 6.0e12\n"
        "compute_vineyard_prefactor = True\n"
    )
    (tmp_path / "pykmc.out").write_text(
        "1 1.0e-12 1.0e-12 3 0.4 2.0 5.0 -1000.0 0.1 0.2\n"
    )
    (tmp_path / "pykmc.log").write_text(
        "Step : 1\n"
        "Vineyard prefactor failed for event 0: singular active Hessian\n"
        ":=> End of simulation\n"
    )
    monkeypatch.setattr(script, "structure_volume_A3", lambda path: 128.0)

    row = script.trial_row_from_outputs(
        case="cu-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["rate_prefactor_source"] == "vineyard-finite-difference"
    assert row["rate_anharmonic_corrections_active"] is True
    assert row["rate_model_ok"] is False
    assert row["rate_model_reason"] == "vineyard-prefactor-failed"
    assert row["kinetic_claim_ok"] is False


def test_trial_row_rejects_missing_event_vineyard_prefactor(tmp_path, monkeypatch):
    script = _load_script()
    initial_config = tmp_path / "initial.xyz"
    initial_config.write_text("placeholder\n")
    (tmp_path / "input.in").write_text(
        "[Control]\n"
        f"initial_config = {initial_config}\n"
        "[RateConstant]\n"
        "style = amsel-vtst\n"
        "T = 500.0\n"
        "prefactor = 6.0e12\n"
        "compute_vineyard_prefactor = True\n"
    )
    (tmp_path / "pykmc.out").write_text("")
    (tmp_path / "pykmc.log").write_text(
        "Step : 1\n"
        "Event search failed: No event found\n"
        "No events have been found, empty reference events table.\n"
        ":=> End of simulation\n"
    )
    monkeypatch.setattr(script, "structure_volume_A3", lambda path: 128.0)

    row = script.trial_row_from_outputs(
        case="cu-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["event_discovery_status"] == "zero-events"
    assert row["rate_prefactor_source"] == "vineyard-finite-difference"
    assert row["rate_anharmonic_corrections_active"] is True
    assert row["rate_model_ok"] is False
    assert row["rate_model_reason"] == "event-prefactor-missing"
    assert row["kinetic_claim_ok"] is False


def test_trial_row_rejects_zero_step_censored_run(tmp_path):
    script = _load_script()
    (tmp_path / "pykmc.out").write_text("")
    (tmp_path / "pykmc.log").write_text(
        "No events have been found, empty reference events table.\n"
        ":=> End of simulation\n"
    )

    row = script.trial_row_from_outputs(
        case="ni-vac-sia",
        selector="amsel",
        trial=0,
        seed=11,
        output_dir=tmp_path,
    )

    assert row["recombined"] is False
    assert row["detector_reason"] == "zero-event-discovery"
    assert row["kmc_steps"] == 0
    assert row["event_discovery_status"] == "zero-events"
    assert row["event_searches"] is None
    assert row["kinetic_claim_ok"] is False


def test_survival_rows_are_plot_ready():
    script = _load_script()
    trials = [
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 300.0,
            "recombined": True,
            "t_recombination_s": 1.0,
            "censored_time_s": None,
        },
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 300.0,
            "recombined": False,
            "t_recombination_s": None,
            "censored_time_s": 2.0,
        },
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 300.0,
            "recombined": True,
            "t_recombination_s": 3.0,
            "censored_time_s": None,
        },
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 600.0,
            "recombined": True,
            "t_recombination_s": 4.0,
            "censored_time_s": None,
        },
    ]

    rows = script.survival_rows(trials)

    assert rows == [
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 300.0,
            "time_s": 1.0,
            "n_at_risk": 3,
            "n_events": 1,
            "survival": 2 / 3,
        },
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 300.0,
            "time_s": 3.0,
            "n_at_risk": 1,
            "n_events": 1,
            "survival": 0.0,
        },
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 600.0,
            "time_s": 4.0,
            "n_at_risk": 1,
            "n_events": 1,
            "survival": 0.0,
        },
    ]


def test_recombination_volume_rows_use_survival_exposure_for_cu():
    script = _load_script()
    trials = [
        {
            "case": "cu-vac-sia",
            "selector": "amsel",
            "temperature_K": 300.0,
            "box_volume_A3": 100.0,
            "recombined": True,
            "t_recombination_s": 1.0e-12,
            "censored_time_s": None,
            "kinetic_claim_ok": True,
            "rate_model_ok": True,
            "rate_model_reason": "event-prefactor-vtst",
            "rate_prefactor_source": "event-prefactor",
            "rate_anharmonic_corrections_active": True,
        },
        {
            "case": "cu-vac-sia",
            "selector": "amsel",
            "temperature_K": 300.0,
            "box_volume_A3": 100.0,
            "recombined": False,
            "t_recombination_s": None,
            "censored_time_s": 2.0e-12,
            "kinetic_claim_ok": True,
            "rate_model_ok": False,
            "rate_model_reason": "configured-prefactor",
            "rate_prefactor_source": "rateconstant.prefactor",
            "rate_anharmonic_corrections_active": True,
        },
        {
            "case": "cu-vac-sia",
            "selector": "legacy",
            "temperature_K": 600.0,
            "box_volume_A3": 100.0,
            "recombined": True,
            "t_recombination_s": 0.5e-12,
            "censored_time_s": None,
            "kinetic_claim_ok": False,
            "rate_model_ok": False,
            "rate_model_reason": "legacy-rate-model",
            "rate_prefactor_source": "rateconstant.k0",
            "rate_anharmonic_corrections_active": False,
        },
    ]

    rows = script.recombination_volume_rows(trials)

    assert rows[0]["case"] == "cu-vac-sia"
    assert rows[0]["selector"] == "amsel"
    assert rows[0]["temperature_K"] == 300.0
    assert rows[0]["n_trials"] == 2
    assert rows[0]["n_recombined"] == 1
    assert rows[0]["n_censored"] == 1
    assert rows[0]["kinetic_claim_ok_trials"] == 2
    assert rows[0]["rate_model_ok_trials"] == 1
    assert rows[0]["physical_kinetic_claim_ok_trials"] == 1
    assert rows[0]["rate_anharmonic_corrections_active_trials"] == 2
    assert rows[0]["rate_prefactor_sources"] == (
        "event-prefactor;rateconstant.prefactor"
    )
    assert rows[0]["rate_model_reasons"] == (
        "configured-prefactor;event-prefactor-vtst"
    )
    assert rows[0]["exposure_time_ps"] == pytest.approx(3.0)
    assert rows[0]["event_rate_ps_inv"] == pytest.approx(1.0 / 3.0)
    assert rows[0]["rate_coefficient_A3_per_ps"] == pytest.approx(100.0 / 3.0)
    assert rows[0]["lattice_parameter_A"] == pytest.approx(3.631)
    assert rows[0]["diffusivity_A2_per_ps"] == pytest.approx(0.143)
    expected_viv_A3 = (100.0 / 3.0) * (2.0 / 3.0) * 3.631**2 / 0.143
    assert rows[0]["recombination_volume_A3"] == pytest.approx(expected_viv_A3)
    assert rows[0]["recombination_volume_atomic"] == pytest.approx(
        expected_viv_A3 / (3.631**3 / 4.0)
    )

    assert rows[1]["case"] == "cu-vac-sia"
    assert rows[1]["selector"] == "legacy"
    assert rows[1]["temperature_K"] == 600.0
    assert rows[1]["n_trials"] == 1
    assert rows[1]["n_recombined"] == 1
    assert rows[1]["kinetic_claim_ok_trials"] == 0
    assert rows[1]["rate_model_ok_trials"] == 0
    assert rows[1]["physical_kinetic_claim_ok_trials"] == 0


def test_seed_schedule_is_paired_by_trial():
    script = _load_script()

    assert script.seed_schedule(base_seed=100, trials=3, priorities=["legacy", "amsel"]) == [
        {"priority": "legacy", "trial": 0, "seed": 100},
        {"priority": "amsel", "trial": 0, "seed": 100},
        {"priority": "legacy", "trial": 1, "seed": 101},
        {"priority": "amsel", "trial": 1, "seed": 101},
        {"priority": "legacy", "trial": 2, "seed": 102},
        {"priority": "amsel", "trial": 2, "seed": 102},
    ]


def test_cli_dry_run_writes_manifest_and_commands(tmp_path):
    script = _load_script()
    out = tmp_path / "out"
    template = tmp_path / "input.in"
    template.write_text(
        "[Control]\n"
        "initial_config = ./old.xyz\n"
        "n_steps = 1\n"
        "[pARTn]\n"
        "path_artnso = ./old.so\n"
        "zseed = 0\n"
        "[EventSearch]\n"
        "nsearch = 20\n"
        "[Lammps]\n"
        "pair_coeff = * * ./Ni.eam Ni\n"
        "[BASIN]\n"
        "energy_thr = 0.5\n"
    )
    (tmp_path / "Ni.eam").write_text("potential")

    code = script.main(
        [
            "--case",
            "ni-vac-sia",
            "--template-input",
            str(template),
            "--initial-config",
            str(tmp_path / "initial_config.xyz"),
            "--allow-preloaded-catalog",
            "--reference-table",
            str(tmp_path / "reference_table.pickle"),
            "--visited-environments",
            str(tmp_path / "visited_environments.pickle"),
            "--partn-path",
            str(tmp_path / "libartn-lmp.so"),
            "--priority",
            "legacy",
            "--priority",
            "amsel",
            "--trials",
            "2",
            "--seed",
            "10",
            "--max-steps",
            "5",
            "--event-searches",
            "3",
            "--partn-search-evals",
            "41",
            "--refine-thr",
            "0.0",
            "--basin-energy-thr",
            "0.5",
            "--basin-max-closed-states",
            "2",
            "--basin-max-absorbing-refinements",
            "1",
            "--basin-frontier-committor-tol",
            "0.01",
            "--amsel-selector",
            "amsel-adaptive",
            "--work-budget",
            "closed-states:2",
            "--dry-run",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    assert (out / "manifest.json").exists()
    assert (out / "commands.json").exists()
    assert (out / "trials.csv").read_text().splitlines()[0] == ",".join(
        script.TRIAL_FIELDS
    )
    assert (out / "survival.csv").read_text().splitlines()[0] == ",".join(
        script.SURVIVAL_FIELDS
    )
    assert (out / "recombination_volumes.csv").read_text().splitlines()[0] == ",".join(
        script.RECOMBINATION_VOLUME_FIELDS
    )
    assert (out / "basin_confidence.csv").read_text().splitlines()[0] == ",".join(
        script.BASIN_CONFIDENCE_FIELDS
    )
    assert (out / "basin_trace.csv").read_text().splitlines()[0] == ",".join(
        script.BASIN_TRACE_FIELDS
    )
    assert (out / "events.jsonl").exists()

    commands = json.loads((out / "commands.json").read_text())
    assert len(commands) == 4
    assert commands[0]["priority"] == "legacy"
    assert commands[0]["event_searches"] == 3
    assert commands[1]["priority"] == "amsel"

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(out / "amsel" / "trial-0" / "input.in")
    assert config["Control"]["initial_config"] == str(tmp_path / "initial_config.xyz")
    assert config["Control"]["reference_table"] == str(
        tmp_path / "reference_table.pickle"
    )
    assert config["Control"]["visited_environments"] == str(
        tmp_path / "visited_environments.pickle"
    )
    assert config["Control"]["n_steps"] == "5"
    assert config["Control"]["random_seed"] == "10"
    assert config["Control"]["refine_thr"] == "0.0"
    assert config["EventSearch"]["nsearch"] == "3"
    assert config["pARTn"]["path_artnso"] == str(tmp_path / "libartn-lmp.so")
    assert config["pARTn"]["zseed"] == "10"
    assert config["pARTn"]["nevalf_max"] == "41"
    assert config["Lammps"]["pair_coeff"] == f"* * {tmp_path / 'Ni.eam'} Ni"
    assert config["BASIN"]["selector"] == "amsel-adaptive"
    assert config["BASIN"]["exploration_priority"] == "amsel"
    assert config["BASIN"]["energy_thr"] == "0.5"
    assert config["BASIN"]["max_closed_states"] == "2"
    assert config["BASIN"]["max_absorbing_refinements"] == "1"
    assert config["BASIN"]["frontier_committor_tol"] == "0.01"

    legacy = configparser.ConfigParser()
    legacy.optionxform = str
    legacy.read(out / "legacy" / "trial-0" / "input.in")
    assert legacy["BASIN"]["selector"] == "legacy-fpta"
    assert legacy["BASIN"]["exploration_priority"] == "legacy"
    assert legacy["Control"]["disable_coverage_resampling"] == "True"
    assert "disable_coverage_resampling" not in config["Control"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["partn_search_evals"] == 41


def test_cli_dry_run_writes_temperature_sweep_inputs(tmp_path):
    script = _load_script()
    out = tmp_path / "out"
    template = tmp_path / "input.in"
    template.write_text(
        "[Control]\n"
        "initial_config = ./old.xyz\n"
        "[RateConstant]\n"
        "T = 300.0\n"
        "[pARTn]\n"
        "path_artnso = ./old.so\n"
        "[Lammps]\n"
        "pair_coeff = * * ./Cu.eam Cu\n"
        "[BASIN]\n"
    )
    (tmp_path / "Cu.eam").write_text("potential")

    code = script.main(
        [
            "--case",
            "cu-vac-sia",
            "--template-input",
            str(template),
            "--initial-config",
            str(tmp_path / "initial_config.xyz"),
            "--partn-path",
            str(tmp_path / "libartn-lmp.so"),
            "--priority",
            "amsel",
            "--trials",
            "1",
            "--seed",
            "10",
            "--max-steps",
            "1",
            "--temperature",
            "300",
            "--temperature",
            "600",
            "--dry-run",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    commands = json.loads((out / "commands.json").read_text())
    assert [command["temperature_K"] for command in commands] == [300.0, 600.0]
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(out / "amsel" / "T600" / "trial-0" / "input.in")
    assert config["RateConstant"]["T"] == "600.0"


def test_cli_rejects_preloaded_catalog_without_explicit_opt_in(tmp_path):
    script = _load_script()
    template = tmp_path / "input.in"
    template.write_text("[Control]\n[pARTn]\n[Lammps]\n[BASIN]\n")

    with pytest.raises(SystemExit) as exc:
        script.main(
            [
                "--case",
                "ni-vac-sia",
                "--template-input",
                str(template),
                "--initial-config",
                str(tmp_path / "initial_config.xyz"),
                "--reference-table",
                str(tmp_path / "reference_table.pickle"),
                "--partn-path",
                str(tmp_path / "libartn-lmp.so"),
                "--priority",
                "amsel",
                "--trials",
                "1",
                "--seed",
                "10",
                "--max-steps",
                "1",
                "--dry-run",
                "--out",
                str(tmp_path / "out"),
            ]
        )

    assert exc.value.code == 2


def test_cli_default_inputs_do_not_preload_catalog(tmp_path):
    script = _load_script()
    out = tmp_path / "out"
    template = tmp_path / "input.in"
    template.write_text(
        "[Control]\n"
        "initial_config = ./old.xyz\n"
        "[pARTn]\n"
        "path_artnso = ./old.so\n"
        "[Lammps]\n"
        "pair_coeff = * * ./Cu.eam Cu\n"
        "[BASIN]\n"
    )
    (tmp_path / "Cu.eam").write_text("potential")

    code = script.main(
        [
            "--case",
            "cu-vac-sia",
            "--template-input",
            str(template),
            "--initial-config",
            str(tmp_path / "initial_config.xyz"),
            "--partn-path",
            str(tmp_path / "libartn-lmp.so"),
            "--priority",
            "amsel",
            "--trials",
            "1",
            "--seed",
            "10",
            "--max-steps",
            "1",
            "--dry-run",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(out / "amsel" / "trial-0" / "input.in")
    assert "reference_table" not in config["Control"]
    assert "visited_environments" not in config["Control"]


def test_cli_generic_case_requires_explicit_inputs(tmp_path):
    script = _load_script()

    with pytest.raises(SystemExit) as exc:
        script.main(
            [
                "--case",
                "generic-defect",
                "--partn-path",
                str(tmp_path / "libartn-lmp.so"),
                "--priority",
                "amsel",
                "--trials",
                "1",
                "--seed",
                "10",
                "--max-steps",
                "1",
                "--dry-run",
                "--out",
                str(tmp_path / "out"),
            ]
        )

    assert exc.value.code == 2


def test_cli_generic_case_uses_explicit_inputs_without_preload(tmp_path):
    script = _load_script()
    out = tmp_path / "out"
    template = tmp_path / "input.in"
    template.write_text(
        "[Control]\n"
        "initial_config = ./old.xyz\n"
        "[pARTn]\n"
        "path_artnso = ./old.so\n"
        "[Lammps]\n"
        "pair_coeff = * * ./Generic.eam X\n"
        "[BASIN]\n"
    )
    (tmp_path / "Generic.eam").write_text("potential")

    code = script.main(
        [
            "--case",
            "generic-defect",
            "--template-input",
            str(template),
            "--initial-config",
            str(tmp_path / "initial_config.xyz"),
            "--partn-path",
            str(tmp_path / "libartn-lmp.so"),
            "--priority",
            "amsel",
            "--trials",
            "1",
            "--seed",
            "10",
            "--max-steps",
            "1",
            "--dry-run",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["case"] == "generic-defect"
    assert manifest["reference_table"] is None
    assert manifest["visited_environments"] is None

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(out / "amsel" / "trial-0" / "input.in")
    assert config["Control"]["initial_config"] == str(tmp_path / "initial_config.xyz")
    assert "reference_table" not in config["Control"]
    assert "visited_environments" not in config["Control"]


def test_render_trial_input_can_request_diverse_amsel_exploration(tmp_path):
    script = _load_script()

    text = script.render_trial_input(
        template_text="[Control]\n[pARTn]\n[BASIN]\n",
        template_dir=tmp_path,
        initial_config=tmp_path / "initial.xyz",
        reference_table=None,
        visited_environments=None,
        max_steps=1,
        event_searches=None,
        refine_thr=None,
        priority="amsel",
        partn_path=tmp_path / "libartn-lmp.so",
        seed=1000,
        basin_energy_thr=None,
        basin_max_expansions=None,
        basin_max_closed_states=None,
        basin_max_absorbing_refinements=None,
        basin_frontier_committor_tol=None,
        amsel_selector="amsel-adaptive",
        amsel_exploration_priority="amsel-diverse",
    )

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(text)
    assert config["BASIN"]["selector"] == "amsel-adaptive"
    assert config["BASIN"]["exploration_priority"] == "amsel-diverse"


def test_render_trial_input_enables_vineyard_prefactors_for_amsel_priority(tmp_path):
    script = _load_script()

    text = script.render_trial_input(
        template_text="[Control]\n[pARTn]\n[BASIN]\n[RateConstant]\nstyle = amsel-vtst\n",
        template_dir=tmp_path,
        initial_config=tmp_path / "initial.xyz",
        reference_table=None,
        visited_environments=None,
        max_steps=1,
        event_searches=None,
        refine_thr=None,
        priority="amsel",
        partn_path=tmp_path / "libartn-lmp.so",
        seed=1000,
        basin_energy_thr=None,
        basin_max_expansions=None,
        basin_max_closed_states=None,
        basin_max_absorbing_refinements=None,
        basin_frontier_committor_tol=None,
        amsel_selector="amsel-adaptive",
        amsel_exploration_priority="amsel-diverse",
    )

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(text)
    assert config["RateConstant"]["compute_vineyard_prefactor"] == "True"


def test_render_trial_input_uses_committor_tolerance_without_default_absorbing_cap(
    tmp_path,
):
    script = _load_script()

    text = script.render_trial_input(
        template_text=(
            "[Control]\n"
            "[pARTn]\n"
            "[BASIN]\n"
            "max_absorbing_refinements = 4\n"
            "frontier_committor_tol = 0.1\n"
            "[RateConstant]\n"
            "style = amsel-vtst\n"
        ),
        template_dir=tmp_path,
        initial_config=tmp_path / "initial.xyz",
        reference_table=None,
        visited_environments=None,
        max_steps=1,
        event_searches=None,
        refine_thr=None,
        priority="amsel",
        partn_path=tmp_path / "libartn-lmp.so",
        seed=1000,
        basin_energy_thr=None,
        basin_max_expansions=None,
        basin_max_closed_states=None,
        basin_max_absorbing_refinements=None,
        basin_frontier_committor_tol=None,
        amsel_selector="amsel-adaptive",
        amsel_exploration_priority="amsel-diverse",
    )

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(text)
    assert "max_absorbing_refinements" not in config["BASIN"]
    assert config["BASIN"]["frontier_committor_tol"] == "0.1"


def test_render_trial_input_preserves_explicit_absorbing_refinement_cap(tmp_path):
    script = _load_script()

    text = script.render_trial_input(
        template_text=(
            "[Control]\n"
            "[pARTn]\n"
            "[BASIN]\n"
            "max_absorbing_refinements = 4\n"
            "frontier_committor_tol = 0.1\n"
        ),
        template_dir=tmp_path,
        initial_config=tmp_path / "initial.xyz",
        reference_table=None,
        visited_environments=None,
        max_steps=1,
        event_searches=None,
        refine_thr=None,
        priority="amsel",
        partn_path=tmp_path / "libartn-lmp.so",
        seed=1000,
        basin_energy_thr=None,
        basin_max_expansions=None,
        basin_max_closed_states=None,
        basin_max_absorbing_refinements=2,
        basin_frontier_committor_tol=None,
        amsel_selector="amsel-adaptive",
        amsel_exploration_priority="amsel-diverse",
    )

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(text)
    assert config["BASIN"]["max_absorbing_refinements"] == "2"


def test_render_trial_input_enables_amsel_recombination_capture(tmp_path):
    script = _load_script()

    text = script.render_trial_input(
        template_text="[Control]\n[pARTn]\n[BASIN]\n",
        template_dir=tmp_path,
        initial_config=tmp_path / "initial.xyz",
        reference_table=None,
        visited_environments=None,
        max_steps=1,
        event_searches=None,
        refine_thr=None,
        priority="amsel",
        partn_path=tmp_path / "libartn-lmp.so",
        seed=1000,
        basin_energy_thr=None,
        basin_max_expansions=None,
        basin_max_closed_states=None,
        basin_max_absorbing_refinements=None,
        basin_frontier_committor_tol=None,
        amsel_selector="amsel-adaptive",
        amsel_exploration_priority="amsel-diverse",
    )

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(text)
    assert config["Control"]["amsel_recomb_inject"] == "True"
    assert config["pARTn"]["amsel_recomb_seed"] == "True"


def test_render_trial_input_keeps_legacy_recombination_capture_disabled(tmp_path):
    script = _load_script()

    text = script.render_trial_input(
        template_text="[Control]\n[pARTn]\n[BASIN]\n",
        template_dir=tmp_path,
        initial_config=tmp_path / "initial.xyz",
        reference_table=None,
        visited_environments=None,
        max_steps=1,
        event_searches=None,
        refine_thr=None,
        priority="legacy",
        partn_path=tmp_path / "libartn-lmp.so",
        seed=1000,
        basin_energy_thr=None,
        basin_max_expansions=None,
        basin_max_closed_states=None,
        basin_max_absorbing_refinements=None,
        basin_frontier_committor_tol=None,
        amsel_selector="amsel-adaptive",
        amsel_exploration_priority="amsel-diverse",
    )

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(text)
    assert "amsel_recomb_inject" not in config["Control"]
    assert "amsel_recomb_seed" not in config["pARTn"]


def test_render_trial_input_keeps_legacy_prefactors_configured(tmp_path):
    script = _load_script()

    text = script.render_trial_input(
        template_text="[Control]\n[pARTn]\n[BASIN]\n[RateConstant]\nstyle = amsel-vtst\n",
        template_dir=tmp_path,
        initial_config=tmp_path / "initial.xyz",
        reference_table=None,
        visited_environments=None,
        max_steps=1,
        event_searches=None,
        refine_thr=None,
        priority="legacy",
        partn_path=tmp_path / "libartn-lmp.so",
        seed=1000,
        basin_energy_thr=None,
        basin_max_expansions=None,
        basin_max_closed_states=None,
        basin_max_absorbing_refinements=None,
        basin_frontier_committor_tol=None,
        amsel_selector="amsel-adaptive",
        amsel_exploration_priority="amsel-diverse",
    )

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read_string(text)
    assert "compute_vineyard_prefactor" not in config["RateConstant"]


def test_basin_confidence_rows_parse_frontier_and_absorbing_markers():
    script = _load_script()

    rows = script.basin_confidence_rows_from_log(
        case="ni-vac-sia",
        selector="amsel",
        trial=0,
        seed=1000,
        log_text=(
            "Step : 4\n"
            "\t :=> Basin exploration budget left 8 frontier states; "
            "unresolved_committor=1.000000e+00; unresolved_rate=1.524846e-02\n"
            "\t :=> Basin absorbing refinement skipped 12 exits; "
            "unresolved_committor=4.937930e-17; unresolved_rate=7.529583e-19\n"
            "Step : 5\n"
            "\t :=> Basin exploration budget left 3 frontier states; "
            "unresolved_committor=2.500000e-01; unresolved_rate=7.000000e-03\n"
        ),
    )

    assert rows == [
        {
            "case": "ni-vac-sia",
            "selector": "amsel",
            "trial": 0,
            "seed": 1000,
            "step": 4,
            "frontier_states": 8,
            "frontier_committor": 1.0,
            "frontier_rate": 1.524846e-02,
            "skipped_absorbing_exits": 12,
            "skipped_absorbing_committor": 4.937930e-17,
            "skipped_absorbing_rate": 7.529583e-19,
        },
        {
            "case": "ni-vac-sia",
            "selector": "amsel",
            "trial": 0,
            "seed": 1000,
            "step": 5,
            "frontier_states": 3,
            "frontier_committor": 0.25,
            "frontier_rate": 7.0e-03,
            "skipped_absorbing_exits": 0,
            "skipped_absorbing_committor": 0.0,
            "skipped_absorbing_rate": 0.0,
        },
    ]


def test_basin_confidence_rows_parse_absorbed_frontier_boundary():
    script = _load_script()

    rows = script.basin_confidence_rows_from_log(
        case="ni-vac-sia",
        selector="amsel",
        trial=0,
        seed=1000,
        log_text=(
            "Step : 4\n"
            "\t :=> Basin frontier boundary absorbed 8 states; "
            "boundary_committor=7.500000e-01; boundary_rate=1.524846e-02\n"
        ),
    )

    assert rows == [
        {
            "case": "ni-vac-sia",
            "selector": "amsel",
            "trial": 0,
            "seed": 1000,
            "step": 4,
            "frontier_states": 8,
            "frontier_committor": 0.75,
            "frontier_rate": 1.524846e-02,
            "skipped_absorbing_exits": 0,
            "skipped_absorbing_committor": 0.0,
            "skipped_absorbing_rate": 0.0,
        }
    ]


def test_collect_basin_confidence_rows_reads_trial_logs(tmp_path):
    script = _load_script()
    workdir = tmp_path / "legacy" / "trial-0"
    workdir.mkdir(parents=True)
    (workdir / "pykmc.log").write_text(
        "Step : 1\n"
        "\t :=> Basin exploration budget left 2 frontier states; "
        "unresolved_committor=5.000000e-01; unresolved_rate=1.000000e-03\n"
    )

    rows = script.collect_basin_confidence_rows(
        [
            {
                "case": "ni-vac-sia",
                "priority": "legacy",
                "trial": 0,
                "seed": 1000,
                "workdir": str(workdir),
            }
        ]
    )

    assert rows == [
        {
            "case": "ni-vac-sia",
            "selector": "legacy",
            "trial": 0,
            "seed": 1000,
            "step": 1,
            "frontier_states": 2,
            "frontier_committor": 0.5,
            "frontier_rate": 1.0e-03,
            "skipped_absorbing_exits": 0,
            "skipped_absorbing_committor": 0.0,
            "skipped_absorbing_rate": 0.0,
        }
    ]


def test_basin_trace_rows_parse_order_queue_and_guidance():
    script = _load_script()

    rows = script.basin_trace_rows_from_log(
        case="ni-vac-sia",
        selector="amsel",
        trial=0,
        seed=1000,
        log_text=(
            "Step : 7\n"
            "\t :=> Basin exploration trace order=0,13; "
            "queue=14,1; guidance=14:3.750000e-01,1:2.500000e-01; "
            "closed_events=0:NA,13:1; "
            "closed_processes=0:NA,13:1:3330; "
            "closed_guidance=0:0.000000e+00,13:3.750000e-01\n"
        ),
    )

    assert rows == [
        {
            "case": "ni-vac-sia",
            "selector": "amsel",
            "trial": 0,
            "seed": 1000,
            "step": 7,
            "order": "0 13",
            "queue": "14 1",
            "guidance": "14:3.750000e-01 1:2.500000e-01",
            "order_count": 2,
            "queue_count": 2,
            "top_queue_state": 14,
            "top_queue_guidance": 0.375,
            "closed_events": "0:NA 13:1",
            "closed_event_families": "1",
            "closed_event_family_count": 1,
            "closed_processes": "0:NA 13:1:3330",
            "closed_process_signatures": "1:3330",
            "closed_process_signature_count": 1,
            "closed_process_singleton_count": 1,
            "process_completeness_observations": 1,
            "process_completeness_unique": 1,
            "process_missing_mass_estimate": 1.0,
            "closed_guidance": "0:0.000000e+00 13:3.750000e-01",
            "closed_guidance_sum": 0.375,
            "closed_top_guidance": 0.375,
        }
    ]


def test_collect_basin_trace_rows_reads_trial_logs(tmp_path):
    script = _load_script()
    workdir = tmp_path / "amsel" / "trial-0"
    workdir.mkdir(parents=True)
    (workdir / "pykmc.log").write_text(
        "Step : 2\n"
        "\t :=> Basin exploration trace order=0; "
        "queue=13,14; guidance=13:5.000000e-01,14:2.500000e-01; "
        "closed_events=0:NA; closed_guidance=0:0.000000e+00\n"
    )

    rows = script.collect_basin_trace_rows(
        [
            {
                "case": "ni-vac-sia",
                "priority": "amsel",
                "trial": 0,
                "seed": 1000,
                "workdir": str(workdir),
            }
        ]
    )

    assert rows == [
        {
            "case": "ni-vac-sia",
            "selector": "amsel",
            "trial": 0,
            "seed": 1000,
            "step": 2,
            "order": "0",
            "queue": "13 14",
            "guidance": "13:5.000000e-01 14:2.500000e-01",
            "order_count": 1,
            "queue_count": 2,
            "top_queue_state": 13,
            "top_queue_guidance": 0.5,
            "closed_events": "0:NA",
            "closed_event_families": "",
            "closed_event_family_count": 0,
            "closed_processes": "",
            "closed_process_signatures": "",
            "closed_process_signature_count": 0,
            "closed_process_singleton_count": 0,
            "process_completeness_observations": 0,
            "process_completeness_unique": 0,
            "process_missing_mass_estimate": 1.0,
            "closed_guidance": "0:0.000000e+00",
            "closed_guidance_sum": 0.0,
            "closed_top_guidance": 0.0,
        }
    ]


def test_kinetic_guard_rejects_failed_refinement_mass():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {"selector": "amsel", "kinetic_claim_ok": True},
        diagnostics={
            "failed_refinement_committor": 0.125,
            "usable_resolved_committor": 3.6e-9,
        },
        log_text="",
    )

    assert guarded["kinetic_claim_ok"] is False
    assert guarded["failed_refinements"] == 0
    assert guarded["failed_refinement_committor"] == 0.125
    assert guarded["usable_resolved_committor"] == 3.6e-9


def test_kinetic_guard_rejects_unresolved_basin_failure():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {"selector": "amsel", "kinetic_claim_ok": True},
        diagnostics=None,
        log_text=(
            "Basin fails with error : "
            "ErrorInfo(type=<ErrorType.EVENT_NOT_FOUND: 1>, message='missing')"
        ),
    )

    assert guarded["kinetic_claim_ok"] is False
    assert guarded["failed_refinements"] == 1
    assert guarded["failed_refinement_committor"] == 0.0


def test_kinetic_guard_rejects_unresolved_absorbing_refinement_budget():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {"selector": "amsel", "kinetic_claim_ok": True},
        diagnostics=None,
        log_text=(
            "Basin absorbing refinement skipped 2 exits; "
            "unresolved_committor=3.250000e-01; unresolved_rate=1.0"
        ),
    )

    assert guarded["kinetic_claim_ok"] is False
    assert guarded["failed_refinements"] == 2
    assert guarded["failed_refinement_committor"] == 0.325


def test_kinetic_guard_accepts_zero_mass_absorbing_refinement_budget():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {
            "selector": "amsel",
            "recombined": False,
            "kmc_steps": 1,
            "kinetic_claim_ok": True,
        },
        diagnostics=None,
        log_text=(
            "Basin absorbing refinement skipped 7 exits; "
            "unresolved_committor=3.600574e-18; unresolved_rate=1.0"
        ),
    )

    assert guarded["kinetic_claim_ok"] is True
    assert guarded["failed_refinements"] == 7
    assert guarded["failed_refinement_committor"] == pytest.approx(3.600574e-18)


def test_kinetic_guard_rejects_unresolved_frontier_budget():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {"selector": "amsel", "kinetic_claim_ok": True},
        diagnostics=None,
        log_text=(
            "Basin exploration budget left 3 frontier states; "
            "unresolved_committor=8.750000e-01; unresolved_rate=2.0"
        ),
    )

    assert guarded["kinetic_claim_ok"] is False
    assert guarded["failed_refinements"] == 3
    assert guarded["failed_refinement_committor"] == 0.875


def test_kinetic_guard_accepts_absorbed_frontier_boundary():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {
            "selector": "amsel",
            "recombined": False,
            "kmc_steps": 2,
            "kinetic_claim_ok": True,
        },
        diagnostics=None,
        log_text=(
            "Basin frontier boundary absorbed 3 states; "
            "boundary_committor=8.750000e-01; boundary_rate=2.0"
        ),
    )

    assert guarded["kinetic_claim_ok"] is True
    assert guarded["failed_refinements"] == 0
    assert guarded["failed_refinement_committor"] == 0.0


def test_kinetic_guard_uses_latest_process_coverage_certificate():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {
            "selector": "amsel",
            "recombined": False,
            "kmc_steps": 1,
            "kinetic_claim_ok": True,
        },
        diagnostics=None,
        log_text=(
            "AMSEL process coverage env=env-a; attempts=1; observations=1; "
            "unique_processes=1; singleton_processes=1; "
            "missing_process_mass=7.142857e-01; "
            "missing_rate_mass=1.593874e-03; needs_more_search=True\n"
            "AMSEL process coverage env=env-a; attempts=15; observations=15; "
            "unique_processes=1; singleton_processes=0; "
            "missing_process_mass=4.983389e-02; "
            "missing_rate_mass=1.668008e-03; needs_more_search=False"
        ),
    )

    assert guarded["kinetic_claim_ok"] is True
    assert guarded["coverage_envs_observed"] == 1
    assert guarded["coverage_total_attempts"] == 15
    assert guarded["coverage_total_observations"] == 15
    assert guarded["coverage_max_missing_process_mass"] == pytest.approx(0.04983389)
    assert guarded["coverage_max_missing_rate_mass"] == pytest.approx(1.668008e-03)
    assert guarded["coverage_needs_more_search"] is False


def test_kinetic_guard_ignores_process_coverage_for_legacy_selector():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {
            "selector": "legacy",
            "recombined": False,
            "kmc_steps": 1,
            "kinetic_claim_ok": True,
        },
        diagnostics=None,
        log_text=(
            "AMSEL process coverage env=env-a; attempts=1; observations=1; "
            "unique_processes=1; singleton_processes=1; "
            "missing_process_mass=7.142857e-01; "
            "missing_rate_mass=1.593874e-03; needs_more_search=True"
        ),
    )

    assert guarded["kinetic_claim_ok"] is True
    assert guarded["coverage_needs_more_search"] is True


def test_kinetic_guard_ignores_process_coverage_when_resampling_disabled():
    script = _load_script()

    guarded = script.apply_kinetic_guard(
        {
            "selector": "amsel",
            "recombined": False,
            "kmc_steps": 1,
            "kinetic_claim_ok": True,
            "coverage_resampling_disabled": True,
        },
        diagnostics=None,
        log_text=(
            "AMSEL process coverage env=env-a; attempts=1; observations=1; "
            "unique_processes=1; singleton_processes=1; "
            "missing_process_mass=7.142857e-01; "
            "missing_rate_mass=1.593874e-03; needs_more_search=True"
        ),
    )

    assert guarded["kinetic_claim_ok"] is True
    assert guarded["coverage_needs_more_search"] is True


def test_execute_trials_records_timeout_as_unusable_kinetics(tmp_path, monkeypatch):
    script = _load_script()
    workdir = tmp_path / "legacy" / "trial-0"
    workdir.mkdir(parents=True)
    (workdir / "pykmc.out").write_text(
        "1 1.0e-12 1.0e-12 3 0.4 2.0 5.0 -1000.0 0.1 0.2\n"
        "2 2.0e-12 3.0e-12 3 0.4 2.0 5.0 -1000.0 0.3 0.4\n"
    )
    (workdir / "pykmc.log").write_text("Step : 2\n")
    (workdir / "trajkmc.xyz").write_text("trajectory placeholder\n")

    class FakeTimedOutProcess:
        pid = 4321
        returncode = None

        def __init__(self):
            self.communicate_calls = 0

        def communicate(self, timeout=None):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(
                    cmd=["python", "-m", "pykmc"],
                    timeout=timeout,
                    output="partial stdout",
                )
            self.returncode = -15
            return ("", None)

    def fake_popen(*_args, **_kwargs):
        return FakeTimedOutProcess()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or args[0],
            timeout=7.5,
            output="partial stdout",
        )

    kill_calls = []
    monkeypatch.setattr(script.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        script,
        "os",
        SimpleNamespace(
            getpgid=lambda pid: pid,
            killpg=lambda pgid, sig: kill_calls.append((pgid, sig)),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "signal",
        SimpleNamespace(SIGTERM=15),
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "trajectory_noncrystal_counts",
        lambda path: [28, 27, 25],
    )

    rows = script.execute_trials(
        [
            {
                "case": "ni-vac-sia",
                "priority": "legacy",
                "trial": 0,
                "seed": 1000,
                "event_searches": 2,
                "workdir": str(workdir),
                "command": ["python", "-m", "pykmc"],
            }
        ],
        tmp_path / "events.jsonl",
        trial_timeout_s=7.5,
    )

    assert rows == [
        {
            "case": "ni-vac-sia",
            "selector": "legacy",
            "trial": 0,
            "seed": 1000,
            "temperature_K": None,
            "box_volume_A3": None,
            "recombined": False,
            "t_recombination_s": None,
            "censored_time_s": 3.0e-12,
            "kmc_steps": 2,
            "cpu_time_s": 0.3,
            "wall_time_s": 0.4,
            "total_cpu_time_s": pytest.approx(0.4),
            "total_wall_time_s": pytest.approx(0.6),
            "detector_reason": "timeout-7.5s",
            "event_discovery_status": "not-zero-event",
            "event_searches": 2,
            "final_noncrystal_atoms": 25,
            "min_noncrystal_atoms": 25,
            "trajectory_recombination_frame": None,
            "coverage_resampling_disabled": False,
            "failed_refinements": 0,
            "failed_refinement_committor": 0.0,
            "usable_resolved_committor": None,
            "kinetic_claim_ok": False,
            "output_dir": str(workdir),
        }
    ]
    assert kill_calls == [(4321, 15)]
    assert "partial stdout" in (workdir / "harness.log").read_text()
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert events == [
        {
            "case": "ni-vac-sia",
            "priority": "legacy",
            "returncode": None,
            "seed": 1000,
            "temperature_K": None,
            "timed_out": True,
            "timeout_s": 7.5,
            "trial": 0,
        }
    ]


def test_execute_trials_rejects_timeout_vineyard_rate_model(tmp_path, monkeypatch):
    script = _load_script()
    workdir = tmp_path / "amsel" / "trial-0"
    workdir.mkdir(parents=True)
    (workdir / "input.in").write_text(
        "[RateConstant]\n"
        "style = amsel-vtst\n"
        "prefactor = 5.0e12\n"
        "compute_vineyard_prefactor = True\n"
    )
    (workdir / "pykmc.out").write_text("")
    (workdir / "pykmc.log").write_text("Step : 1\n")

    class FakeTimedOutProcess:
        pid = 4321
        returncode = None

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(
                cmd=["python", "-m", "pykmc"],
                timeout=timeout,
                output="partial stdout",
            )

    def fake_popen(*_args, **_kwargs):
        return FakeTimedOutProcess()

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or args[0],
            timeout=5.0,
            output="partial stdout",
        )

    monkeypatch.setattr(script.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(
        script,
        "os",
        SimpleNamespace(
            getpgid=lambda pid: pid,
            killpg=lambda _pgid, _sig: None,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        script,
        "signal",
        SimpleNamespace(SIGTERM=15),
        raising=False,
    )

    rows = script.execute_trials(
        [
            {
                "case": "cu-vac-sia",
                "priority": "amsel",
                "trial": 0,
                "seed": 1000,
                "event_searches": 3,
                "workdir": str(workdir),
                "command": ["python", "-m", "pykmc"],
            }
        ],
        tmp_path / "events.jsonl",
        trial_timeout_s=5.0,
    )

    assert rows[0]["detector_reason"] == "timeout-5.0s"
    assert rows[0]["kinetic_claim_ok"] is False
    assert rows[0]["rate_prefactor_source"] == "vineyard-finite-difference"
    assert rows[0]["rate_model_ok"] is False
    assert rows[0]["rate_model_reason"] == "vineyard-prefactor-timeout"


def test_execute_trials_runs_child_in_isolated_session(tmp_path, monkeypatch):
    script = _load_script()
    workdir = tmp_path / "amsel" / "trial-0"
    workdir.mkdir(parents=True)
    observed = {}

    class FakeProcess:
        returncode = 125

        def communicate(self, timeout=None):
            return ("child aborted", None)

    def fake_popen(*args, **kwargs):
        observed.update(kwargs)
        return FakeProcess()

    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["python", "-m", "pykmc"],
            returncode=125,
            stdout="child aborted",
        )

    monkeypatch.setattr(script.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(script.subprocess, "run", fake_run)

    rows = script.execute_trials(
        [
            {
                "case": "ni-vac-sia",
                "priority": "amsel",
                "trial": 0,
                "seed": 1000,
                "event_searches": 2,
                "workdir": str(workdir),
                "command": ["python", "-m", "pykmc"],
            }
        ],
        tmp_path / "events.jsonl",
    )

    assert observed["start_new_session"] is True
    assert (workdir / "harness.log").read_text() == "child aborted"
    assert rows[0]["detector_reason"] == "returncode-125"
