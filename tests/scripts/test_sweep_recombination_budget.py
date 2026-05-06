import csv
import importlib.util
import json
from pathlib import Path


def _load_script():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "sweep_recombination_budget.py"
    )
    spec = importlib.util.spec_from_file_location("sweep_recombination_budget", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_budget_sweep_commands_label_and_isolate_outputs(tmp_path):
    script = _load_script()

    commands = script.budget_sweep_commands(
        measure_script=tmp_path / "measure_recombination_kinetics.py",
        python="python",
        out=tmp_path / "sweep",
        closed_states=[1, 2],
        absorbing_refinements=[0],
        measure_args=["--case", "ni-vac-sia", "--priority", "amsel"],
    )

    assert commands == [
        {
            "budget_label": "closed-1_absorb-0",
            "basin_max_closed_states": 1,
            "basin_max_absorbing_refinements": 0,
            "out": str(tmp_path / "sweep" / "closed-1_absorb-0"),
            "command": [
                "python",
                str(tmp_path / "measure_recombination_kinetics.py"),
                "--case",
                "ni-vac-sia",
                "--priority",
                "amsel",
                "--basin-max-closed-states",
                "1",
                "--basin-max-absorbing-refinements",
                "0",
                "--work-budget",
                "closed-1_absorb-0",
                "--out",
                str(tmp_path / "sweep" / "closed-1_absorb-0"),
            ],
        },
        {
            "budget_label": "closed-2_absorb-0",
            "basin_max_closed_states": 2,
            "basin_max_absorbing_refinements": 0,
            "out": str(tmp_path / "sweep" / "closed-2_absorb-0"),
            "command": [
                "python",
                str(tmp_path / "measure_recombination_kinetics.py"),
                "--case",
                "ni-vac-sia",
                "--priority",
                "amsel",
                "--basin-max-closed-states",
                "2",
                "--basin-max-absorbing-refinements",
                "0",
                "--work-budget",
                "closed-2_absorb-0",
                "--out",
                str(tmp_path / "sweep" / "closed-2_absorb-0"),
            ],
        },
    ]


def test_cli_dry_run_writes_sweep_manifest_and_commands(tmp_path):
    script = _load_script()
    out = tmp_path / "sweep"

    code = script.main(
        [
            "--measure-script",
            str(tmp_path / "measure_recombination_kinetics.py"),
            "--python",
            "python",
            "--closed-states",
            "1,2",
            "--absorbing-refinements",
            "0,1",
            "--dry-run",
            "--out",
            str(out),
            "--",
            "--case",
            "ni-vac-sia",
            "--priority",
            "legacy",
            "--priority",
            "amsel",
        ]
    )

    assert code == 0
    manifest = json.loads((out / "sweep_manifest.json").read_text())
    commands = json.loads((out / "sweep_commands.json").read_text())
    assert manifest["closed_states"] == [1, 2]
    assert manifest["absorbing_refinements"] == [0, 1]
    assert len(commands) == 4
    assert commands[0]["budget_label"] == "closed-1_absorb-0"


def test_aggregate_outputs_adds_budget_columns(tmp_path):
    script = _load_script()
    child = tmp_path / "sweep" / "closed-1_absorb-0"
    child.mkdir(parents=True)
    _write_csv(
        child / "trials.csv",
        [
            {
                "selector": "amsel",
                "trial": "0",
                "kinetic_claim_ok": "False",
                "wall_time_s": "12.5",
            }
        ],
    )
    _write_csv(
        child / "basin_confidence.csv",
        [
            {
                "selector": "amsel",
                "step": "4",
                "frontier_states": "8",
                "frontier_committor": "1.0",
                "frontier_rate": "0.015",
            }
        ],
    )
    _write_csv(
        child / "basin_trace.csv",
        [
            {
                "selector": "amsel",
                "step": "4",
                "order": "0 13",
                "queue": "13 14",
                "top_queue_state": "13",
                "top_queue_guidance": "0.5",
            }
        ],
    )

    script.write_aggregate_outputs(
        tmp_path / "sweep",
        [
            {
                "budget_label": "closed-1_absorb-0",
                "basin_max_closed_states": 1,
                "basin_max_absorbing_refinements": 0,
                "out": str(child),
                "command": ["python"],
            }
        ],
    )

    trials = list(csv.DictReader((tmp_path / "sweep" / "sweep_trials.csv").open()))
    confidence = list(
        csv.DictReader((tmp_path / "sweep" / "sweep_basin_confidence.csv").open())
    )
    trace = list(csv.DictReader((tmp_path / "sweep" / "sweep_basin_trace.csv").open()))
    assert trials[0]["budget_label"] == "closed-1_absorb-0"
    assert trials[0]["basin_max_closed_states"] == "1"
    assert confidence[0]["frontier_committor"] == "1.0"
    assert trace[0]["queue"] == "13 14"
    summary = list(csv.DictReader((tmp_path / "sweep" / "sweep_summary.csv").open()))
    assert summary == [
        {
            "budget_label": "closed-1_absorb-0",
            "basin_max_closed_states": "1",
            "basin_max_absorbing_refinements": "0",
            "selector": "amsel",
            "trials": "1",
            "kinetic_claim_ok": "0",
            "recombined": "0",
            "last_wall_time_s": "12.5",
            "last_frontier_states": "8",
            "last_frontier_committor": "1.0",
            "last_frontier_rate": "0.015",
            "last_order": "0 13",
            "last_queue": "13 14",
            "last_top_queue_state": "13",
            "last_top_queue_guidance": "0.5",
        }
    ]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
