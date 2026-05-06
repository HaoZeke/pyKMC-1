import importlib.util
from pathlib import Path

import pytest


def test_basins_amsel_benchmarks_run_once():
    pytest.importorskip("amsel")
    benchmark_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "basins_amsel.py"
    )
    spec = importlib.util.spec_from_file_location("basins_amsel_bench", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    suite = module.BasinAmselSuite()
    suite.setup(8)
    suite.time_legacy_fpta_selector(8)
    suite.time_amsel_sampled_selector(8)
    suite.time_amsel_frontier_guidance(8)


def test_defective_basin_amsel_benchmarks_run_once():
    pytest.importorskip("amsel")
    benchmark_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / "basins_amsel.py"
    )
    spec = importlib.util.spec_from_file_location("basins_amsel_bench", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    suite = module.DefectiveBasinAmselSuite()
    suite.setup()
    suite.time_legacy_defective_selector()
    suite.time_amsel_sampled_defective_selector()
