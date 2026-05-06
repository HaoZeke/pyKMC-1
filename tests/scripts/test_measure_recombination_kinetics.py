import configparser
import importlib.util
import json
import subprocess
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
    assert row["failed_refinements"] == 0
    assert row["failed_refinement_committor"] == 0.0
    assert row["usable_resolved_committor"] is None
    assert row["kinetic_claim_ok"] is True


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
            "--refine-thr",
            "0.0",
            "--basin-energy-thr",
            "0.5",
            "--basin-max-closed-states",
            "2",
            "--basin-max-absorbing-refinements",
            "1",
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
    assert config["Lammps"]["pair_coeff"] == f"* * {tmp_path / 'Ni.eam'} Ni"
    assert config["BASIN"]["selector"] == "amsel-adaptive"
    assert config["BASIN"]["exploration_priority"] == "amsel"
    assert config["BASIN"]["energy_thr"] == "0.5"
    assert config["BASIN"]["max_closed_states"] == "2"
    assert config["BASIN"]["max_absorbing_refinements"] == "1"

    legacy = configparser.ConfigParser()
    legacy.optionxform = str
    legacy.read(out / "legacy" / "trial-0" / "input.in")
    assert legacy["BASIN"]["selector"] == "legacy-fpta"
    assert legacy["BASIN"]["exploration_priority"] == "legacy"


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
            "queue=14,1; guidance=14:3.750000e-01,1:2.500000e-01\n"
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
        }
    ]


def test_collect_basin_trace_rows_reads_trial_logs(tmp_path):
    script = _load_script()
    workdir = tmp_path / "amsel" / "trial-0"
    workdir.mkdir(parents=True)
    (workdir / "pykmc.log").write_text(
        "Step : 2\n"
        "\t :=> Basin exploration trace order=0; "
        "queue=13,14; guidance=13:5.000000e-01,14:2.500000e-01\n"
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


def test_execute_trials_records_timeout_as_unusable_kinetics(tmp_path, monkeypatch):
    script = _load_script()
    workdir = tmp_path / "legacy" / "trial-0"
    workdir.mkdir(parents=True)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=kwargs.get("args") or args[0],
            timeout=7.5,
            output="partial stdout",
        )

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    rows = script.execute_trials(
        [
            {
                "case": "ni-vac-sia",
                "priority": "legacy",
                "trial": 0,
                "seed": 1000,
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
            "recombined": False,
            "t_recombination_s": None,
            "censored_time_s": 0.0,
            "kmc_steps": 0,
            "cpu_time_s": None,
            "wall_time_s": None,
            "detector_reason": "timeout-7.5s",
            "failed_refinements": 0,
            "failed_refinement_committor": 0.0,
            "usable_resolved_committor": None,
            "kinetic_claim_ok": False,
            "output_dir": str(workdir),
        }
    ]
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
            "timed_out": True,
            "timeout_s": 7.5,
            "trial": 0,
        }
    ]
