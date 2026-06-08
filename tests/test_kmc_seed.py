import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pykmc.kmc as kmc_module
from pykmc.kmc import (
    EnvironmentSearchEvidence,
    KMC,
    basin_exploration_trace_line,
    environment_search_evidence_trace_lines,
    environments_with_cataloged_searches,
    event_search_attempt_evidence,
    event_search_process_evidence,
    undercovered_environments_for_search,
)
from pykmc.eventsearch import EventSearch
from pykmc.result import Err, ErrorInfo, ErrorType, EventSearchOutput, Ok


def test_kmc_seeds_python_and_numpy_rngs_from_control_config():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))

    kmc._seed_rngs()

    expected_python = random.Random(12345).random()
    expected_numpy = np.random.RandomState(12345).random_sample()
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy


def test_central_atoms_research_covers_distinct_atoms_before_resampling():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-a", "env-a", "crystal"]
    )
    random.seed(0)

    central_atoms = kmc.central_atoms_research(["env-a"], nsearch=3)

    assert sorted(central_atoms) == [0, 1, 2]


def test_central_atoms_research_uses_amsel_recombination_center(monkeypatch):
    kmc = KMC(
        SimpleNamespace(
            control=SimpleNamespace(random_seed=12345),
            partn=SimpleNamespace(amsel_recomb_seed=True),
        )
    )
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-a", "crystal", "crystal"]
    )
    kmc.system = SimpleNamespace(
        positions=np.zeros((4, 3), dtype=float),
        cell=np.eye(3),
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.recombination_search_center",
        lambda positions, cell: 3,
        raising=False,
    )
    random.seed(0)

    central_atoms = kmc.central_atoms_research(["env-a"], nsearch=1)

    assert central_atoms[0] == 3
    assert any(atom in central_atoms for atom in [0, 1])


def test_central_atoms_research_prioritizes_amsel_center_without_dropping_environments(
    monkeypatch,
):
    kmc = KMC(
        SimpleNamespace(
            control=SimpleNamespace(random_seed=12345),
            partn=SimpleNamespace(amsel_recomb_seed=True),
        )
    )
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-b", "env-b", "crystal"]
    )
    kmc.system = SimpleNamespace(
        positions=np.zeros((4, 3), dtype=float),
        cell=np.eye(3),
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.recombination_search_center",
        lambda positions, cell: 3,
        raising=False,
    )
    random.seed(0)

    central_atoms = kmc.central_atoms_research(["env-a", "env-b"], nsearch=1)

    assert central_atoms[0] == 3
    assert 0 in central_atoms
    assert any(atom in central_atoms for atom in [1, 2])


def test_rejected_amsel_capture_keeps_search_center_available(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        partn=SimpleNamespace(amsel_recomb_capture_mult=1.6),
        rateconstant=SimpleNamespace(prefactor=5.0e12),
    )
    kmc.system = SimpleNamespace(
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        cell=np.eye(3) * 20.0,
    )
    kmc.total_energy = 0.0
    kmc.manager = SimpleNamespace(
        use_global=lambda: None,
        global_minimize_with_results=lambda config, positions: SimpleNamespace(
            result=lambda: (positions, -1.0)
        ),
    )
    n_defects = iter([10, 9])
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.detect_recomb",
        lambda positions, cell, capture_mult: (1, [0.0, 0.0, 0.0]),
        raising=False,
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.build_product",
        lambda positions, cell, source, target: np.asarray(positions, dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.n_defects",
        lambda positions, cell: next(n_defects),
        raising=False,
    )

    assert kmc._try_amsel_capture() is None
    assert kmc._amsel_recomb_search_suppressed is False


def test_rejected_amsel_capture_restores_local_manager_mode(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        partn=SimpleNamespace(amsel_recomb_capture_mult=1.6),
        rateconstant=SimpleNamespace(prefactor=5.0e12),
    )
    kmc.system = SimpleNamespace(
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        cell=np.eye(3) * 20.0,
    )
    kmc.total_energy = 0.0
    calls = []
    kmc.manager = SimpleNamespace(
        use_global=lambda: calls.append("global"),
        use_local=lambda: calls.append("local"),
        global_minimize_with_results=lambda config, positions: SimpleNamespace(
            result=lambda: (positions, -1.0)
        ),
    )
    n_defects = iter([10, 9])
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.detect_recomb",
        lambda positions, cell, capture_mult: (1, [0.0, 0.0, 0.0]),
        raising=False,
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.build_product",
        lambda positions, cell, source, target: np.asarray(positions, dtype=float),
        raising=False,
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.n_defects",
        lambda positions, cell: next(n_defects),
        raising=False,
    )

    assert kmc._try_amsel_capture() is None

    assert calls == ["global", "local"]


def test_amsel_capture_uses_global_minimizer(monkeypatch):
    class FakeSystem(SimpleNamespace):
        def update_positions(self, positions):
            self.positions = np.asarray(positions, dtype=float)

    class FakeManager:
        def __init__(self):
            self.calls = []

        def use_global(self):
            self.calls.append("global")

        def use_local(self):
            self.calls.append("local")

        def minimize_with_results(self, *_args, **_kwargs):
            raise AssertionError("direct capture must use the global minimizer")

        def global_minimize_with_results(self, _config, positions=None):
            self.calls.append("global_minimize")
            return np.asarray(positions, dtype=float), -1.0

        def set_all_positions(self, positions):
            self.calls.append("set_all")
            self.positions = np.asarray(positions, dtype=float)

    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        partn=SimpleNamespace(amsel_recomb_capture_mult=1.6),
        rateconstant=SimpleNamespace(prefactor=5.0e12),
    )
    product = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float)
    kmc.system = FakeSystem(
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
        cell=np.eye(3) * 20.0,
    )
    kmc.total_energy = 0.0
    manager = FakeManager()
    kmc.manager = manager
    kmc.loggers = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    n_defects = iter([10, 0])
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.detect_recomb",
        lambda positions, cell, capture_mult: (1, [0.0, 0.0, 0.0]),
        raising=False,
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.build_product",
        lambda positions, cell, source, target: product,
        raising=False,
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.n_defects",
        lambda positions, cell: next(n_defects),
        raising=False,
    )

    dt = kmc._try_amsel_capture()

    assert np.isclose(dt, 2.0e-13)
    assert manager.calls == ["global", "global_minimize", "local", "set_all"]
    np.testing.assert_allclose(kmc.system.positions, product)


def test_suppressed_amsel_capture_skips_recombination_center(monkeypatch):
    kmc = KMC(
        SimpleNamespace(
            control=SimpleNamespace(random_seed=12345),
            partn=SimpleNamespace(amsel_recomb_seed=True),
        )
    )
    kmc._amsel_recomb_search_suppressed = True
    kmc.system = SimpleNamespace(
        positions=np.zeros((4, 3), dtype=float),
        cell=np.eye(3),
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.recombination_search_center",
        lambda positions, cell: 3,
        raising=False,
    )

    assert kmc._amsel_recomb_search_center() is None


def test_event_search_logs_failed_search_reason():
    class FinishedFuture:
        def result(self):
            return Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_FOUND,
                    message="partn_search: pARTn failed",
                )
            )

    class FakeManager:
        def partn_search(self, **_kwargs):
            return [FinishedFuture()]

    log_messages = []
    event_search = EventSearch(
        config=SimpleNamespace(control=SimpleNamespace(active_volume=False)),
        system=SimpleNamespace(
            positions=np.zeros((1, 3)),
            cell=np.eye(3),
            types=["Cu"],
        ),
        manager=FakeManager(),
        loggers=SimpleNamespace(
            info=lambda _name, message: log_messages.append(message),
            progress_bar=lambda *_args: None,
        ),
    )

    event_search.execute([0])

    assert any("partn_search: pARTn failed" in message for message in log_messages)


def test_event_search_passes_amsel_topology_hint(monkeypatch):
    class FinishedFuture:
        def result(self):
            return Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="stop"))

    captured = {}

    class FakeManager:
        def partn_search(self, **kwargs):
            captured.update(kwargs)
            return [FinishedFuture()]

    topology = (7, [1.0, 2.0, 3.0], 0.5, 1.0)
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb._nearest_recomb_topology",
        lambda positions, cell: topology,
        raising=False,
    )

    event_search = EventSearch(
        config=SimpleNamespace(
            control=SimpleNamespace(active_volume=False),
            partn=SimpleNamespace(amsel_recomb_seed=True),
        ),
        system=SimpleNamespace(
            positions=np.zeros((2, 3)),
            cell=np.eye(3),
            types=["Cu", "Cu"],
        ),
        manager=FakeManager(),
        loggers=SimpleNamespace(
            info=lambda *_args: None,
            progress_bar=lambda *_args: None,
        ),
    )

    event_search.execute([7])

    assert captured["amsel_recomb_topology"] == topology


def test_kmc_attaches_vineyard_prefactors_to_event_search_outputs(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            style="amsel-vtst",
            compute_vineyard_prefactor=True,
            vineyard_fd_step_A=1.0e-4,
        ),
        atomicenvironment=SimpleNamespace(rcut=2.0),
    )
    kmc.system = SimpleNamespace(
        types=["Cu"],
        cell=np.eye(3) * 10.0,
    )
    kmc.manager = SimpleNamespace(
        get_forces=lambda positions=None: SimpleNamespace(
            result=lambda: np.zeros_like(positions)
        )
    )
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array([[1.0, 1.0, 1.0]], dtype=float),
        saddle_positions=np.array([[1.1, 1.0, 1.0]], dtype=float),
        min2_positions=np.array([[1.2, 1.0, 1.0]], dtype=float),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
    )

    def fake_prefactors(force_fn, min1, saddle, min2, **kwargs):
        assert kwargs["active_indices"] == [0]
        assert kwargs["step_A"] == 1.0e-4
        assert kwargs["masses_amu"][0] == pytest.approx(63.546)
        return SimpleNamespace(
            forward_prefactor_inv_s=1.1e13,
            backward_prefactor_inv_s=2.2e13,
            saddle_freq_invcm=120.0,
            barrier_omega_rad_per_s=2.4e13,
        )

    monkeypatch.setattr(
        kmc_module,
        "vineyard_event_prefactors_from_forces",
        fake_prefactors,
    )

    kmc._attach_vineyard_prefactors([event])

    assert event.prefactor_inv_s == pytest.approx(1.1e13)
    assert event.product_prefactor_inv_s == pytest.approx(2.2e13)
    assert event.prefactor_source == "vineyard-finite-difference"
    assert event.saddle_freq_invcm == pytest.approx(120.0)
    assert event.barrier_omega_rad_per_s == pytest.approx(2.4e13)


def test_kmc_vineyard_prefactors_log_subspace_and_result(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            style="amsel-vtst",
            compute_vineyard_prefactor=True,
            vineyard_fd_step_A=1.0e-4,
        ),
        atomicenvironment=SimpleNamespace(rcut=3.0),
    )
    kmc.system = SimpleNamespace(
        types=["Cu", "Cu"],
        cell=np.eye(3) * 10.0,
    )
    kmc.manager = SimpleNamespace(
        get_forces=lambda positions=None: SimpleNamespace(
            result=lambda: np.zeros_like(positions)
        )
    )
    log_messages = []
    kmc.loggers = SimpleNamespace(
        info=lambda _name, message: log_messages.append(message)
    )
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        saddle_positions=np.array([[1.2, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=float),
        min2_positions=np.array([[1.4, 1.0, 1.0], [2.4, 1.0, 1.0]], dtype=float),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
    )

    def fake_prefactors(force_fn, min1, saddle, min2, **_kwargs):
        return SimpleNamespace(
            forward_prefactor_inv_s=1.1e13,
            backward_prefactor_inv_s=2.2e13,
            saddle_freq_invcm=120.0,
            barrier_omega_rad_per_s=2.4e13,
        )

    monkeypatch.setattr(
        kmc_module,
        "vineyard_event_prefactors_from_forces",
        fake_prefactors,
    )

    kmc._attach_vineyard_prefactors([event])

    assert any(
        "Vineyard prefactor event at atom 0: active_atoms=2 active_dof=6 force_evaluations=36"
        in message
        for message in log_messages
    )
    assert any(
        "Vineyard prefactor event at atom 0 complete: forward=1.100000e+13/s backward=2.200000e+13/s saddle_freq=1.200000e+02/cm"
        in message
        for message in log_messages
    )


def test_kmc_uses_projected_vineyard_for_large_active_core(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            style="amsel-vtst",
            compute_vineyard_prefactor=True,
            vineyard_fd_step_A=1.0e-4,
        ),
        atomicenvironment=SimpleNamespace(rcut=3.0),
    )
    kmc.system = SimpleNamespace(
        types=["Cu", "Cu", "Cu", "Cu", "Cu"],
        cell=np.eye(3) * 10.0,
    )
    kmc.manager = SimpleNamespace(
        get_forces=lambda positions=None: SimpleNamespace(
            result=lambda: np.zeros_like(positions)
        )
    )
    kmc.loggers = SimpleNamespace(info=lambda *_args: None)
    min1 = np.array([[float(index), 0.0, 0.0] for index in range(5)], dtype=float)
    saddle = min1.copy()
    product = min1.copy()
    saddle[:, 0] += 0.2
    product[:, 0] += 0.4
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=min1,
        saddle_positions=saddle,
        min2_positions=product,
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
    )

    def fail_full_prefactors(*_args, **_kwargs):
        raise AssertionError("full Vineyard path should not run")

    def fake_projected_prefactors(*_args, **_kwargs):
        return SimpleNamespace(
            forward_prefactor_inv_s=1.1e13,
            backward_prefactor_inv_s=2.2e13,
            saddle_freq_invcm=120.0,
            barrier_omega_rad_per_s=2.4e13,
        )

    monkeypatch.setattr(
        kmc_module,
        "vineyard_event_prefactors_from_forces",
        fail_full_prefactors,
    )
    monkeypatch.setattr(
        kmc_module,
        "vineyard_projected_event_prefactors_from_forces",
        fake_projected_prefactors,
    )

    kmc._attach_vineyard_prefactors([event])

    assert event.prefactor_inv_s == pytest.approx(1.1e13)
    assert event.product_prefactor_inv_s == pytest.approx(2.2e13)
    assert event.prefactor_source == "vineyard-projected-mode"


def test_kmc_uses_artn_saddle_curvature_without_force_calls():
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            style="amsel-vtst",
            compute_vineyard_prefactor=True,
            vineyard_fd_step_A=1.0e-4,
            T=300.0,
        ),
        atomicenvironment=SimpleNamespace(rcut=3.0),
    )
    kmc.system = SimpleNamespace(
        types=["Cu", "Cu"],
        cell=np.eye(3) * 10.0,
    )

    class FailingManager:
        def use_global(self):
            raise AssertionError("ARTn curvature path must not switch manager mode")

        def use_local(self):
            raise AssertionError("ARTn curvature path must not restore manager mode")

        def get_forces(self, positions=None):
            raise AssertionError("ARTn curvature path must not evaluate forces")

    kmc.manager = FailingManager()
    kmc.loggers = SimpleNamespace(info=lambda *_args: None)
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        saddle_positions=np.array([[1.1, 1.0, 1.0], [2.1, 1.0, 1.0]], dtype=float),
        min2_positions=np.array([[1.2, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=float),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
        saddle_eigenvalue_ev_per_A2=-1.0,
    )

    kmc._attach_vineyard_prefactors([event])

    expected_prefactor = 8.6173303e-05 * 300.0 / 4.135667e-3 * 1.0e12
    assert event.prefactor_inv_s == pytest.approx(expected_prefactor)
    assert event.product_prefactor_inv_s == pytest.approx(expected_prefactor)
    assert event.prefactor_source == "thermal-tst-artn-curvature"
    assert event.saddle_freq_invcm > 0.0
    assert event.barrier_omega_rad_per_s > 0.0


def test_kmc_vineyard_prefactors_use_global_full_system_forces(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            style="amsel-vtst",
            compute_vineyard_prefactor=True,
            vineyard_fd_step_A=1.0e-4,
        ),
        atomicenvironment=SimpleNamespace(rcut=3.0),
    )
    kmc.system = SimpleNamespace(
        types=["Cu", "Cu"],
        cell=np.eye(3) * 10.0,
    )
    calls = []
    full_forces = np.array(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
        dtype=float,
    )

    class FakeManager:
        def __init__(self):
            self.global_mode = False
            self.use_global_calls = 0
            self.use_local_calls = 0

        def use_global(self):
            self.global_mode = True
            self.use_global_calls += 1

        def use_local(self):
            self.global_mode = False
            self.use_local_calls += 1

        def get_forces(self, positions=None):
            calls.append((self.global_mode, np.asarray(positions, dtype=float).shape))
            if not self.global_mode:
                return SimpleNamespace(result=lambda: np.zeros((1, 3), dtype=float))
            return full_forces.reshape(-1)

    manager = FakeManager()
    kmc.manager = manager
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        saddle_positions=np.array([[1.1, 1.0, 1.0], [2.1, 1.0, 1.0]], dtype=float),
        min2_positions=np.array([[1.2, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=float),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
    )

    def fake_prefactors(force_fn, min1, saddle, min2, **_kwargs):
        np.testing.assert_allclose(force_fn(np.asarray(min1, dtype=float)), full_forces)
        return SimpleNamespace(
            forward_prefactor_inv_s=1.1e13,
            backward_prefactor_inv_s=2.2e13,
            saddle_freq_invcm=120.0,
            barrier_omega_rad_per_s=2.4e13,
        )

    monkeypatch.setattr(
        kmc_module,
        "vineyard_event_prefactors_from_forces",
        fake_prefactors,
    )

    kmc._attach_vineyard_prefactors([event])

    assert calls == [(True, (2, 3))]
    assert manager.use_global_calls == 1
    assert manager.use_local_calls == 1
    assert event.prefactor_source == "vineyard-finite-difference"


def test_kmc_vineyard_active_indices_follow_event_displacements():
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(vineyard_fd_step_A=1.0e-3),
        atomicenvironment=SimpleNamespace(rcut=10.0),
    )
    kmc.system = SimpleNamespace(cell=np.eye(3) * 20.0)
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [4.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        saddle_positions=np.array(
            [
                [1.0, 1.0, 1.0],
                [2.2, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [4.015, 1.0, 1.0],
            ],
            dtype=float,
        ),
        min2_positions=np.array(
            [
                [1.0, 1.0, 1.0],
                [2.4, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [4.015, 1.0, 1.0],
            ],
            dtype=float,
        ),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 20.0,
    )

    assert kmc._vineyard_active_indices(event) == [0, 1]


def test_kmc_vineyard_active_indices_drop_elastic_tail():
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(vineyard_fd_step_A=1.0e-3),
        atomicenvironment=SimpleNamespace(rcut=10.0),
    )
    kmc.system = SimpleNamespace(cell=np.eye(3) * 30.0)
    min1 = np.array([[float(index), 0.0, 0.0] for index in range(12)], dtype=float)
    saddle = min1.copy()
    product = min1.copy()
    saddle[1, 0] += 0.4
    product[1, 0] += 0.5
    for index in range(2, 12):
        saddle[index, 0] += 0.03
        product[index, 0] += 0.03
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=min1,
        saddle_positions=saddle,
        min2_positions=product,
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 30.0,
    )

    assert kmc._vineyard_active_indices(event) == [0, 1]


def test_kmc_vineyard_active_indices_keep_displacement_participation_core():
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(vineyard_fd_step_A=1.0e-3),
        atomicenvironment=SimpleNamespace(rcut=10.0),
    )
    kmc.system = SimpleNamespace(cell=np.eye(3) * 30.0)
    min1 = np.array([[float(index), 0.0, 0.0] for index in range(8)], dtype=float)
    saddle = min1.copy()
    product = min1.copy()
    saddle[1, 0] += 1.0
    product[1, 0] += 1.0
    saddle[2, 0] += 0.35
    product[2, 0] += 0.35
    for index in range(3, 8):
        saddle[index, 0] += 0.12
        product[index, 0] += 0.12
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=min1,
        saddle_positions=saddle,
        min2_positions=product,
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 30.0,
    )

    assert kmc._vineyard_active_indices(event) == [0, 1, 2]


def test_kmc_vineyard_prefactor_logs_force_shape_mismatch(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            style="amsel-vtst",
            compute_vineyard_prefactor=True,
            vineyard_fd_step_A=1.0e-4,
        ),
        atomicenvironment=SimpleNamespace(rcut=3.0),
    )
    kmc.system = SimpleNamespace(
        types=["Cu", "Cu"],
        cell=np.eye(3) * 10.0,
    )
    kmc.manager = SimpleNamespace(
        global_get_forces=lambda positions=None: np.zeros((1, 3), dtype=float),
        get_forces=lambda positions=None: np.zeros((2, 3), dtype=float),
    )
    log_messages = []
    kmc.loggers = SimpleNamespace(
        info=lambda _name, message: log_messages.append(message)
    )
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        saddle_positions=np.array([[1.1, 1.0, 1.0], [2.1, 1.0, 1.0]], dtype=float),
        min2_positions=np.array([[1.2, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=float),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
    )

    def fake_prefactors(force_fn, min1, _saddle, _min2, **_kwargs):
        force_fn(np.asarray(min1, dtype=float))

    monkeypatch.setattr(
        kmc_module,
        "vineyard_event_prefactors_from_forces",
        fake_prefactors,
    )

    kmc._attach_vineyard_prefactors([event])

    assert event.prefactor_source is None
    assert any(
        "force shape (1, 3) does not match positions shape (2, 3)" in message
        for message in log_messages
    )


def test_kmc_vineyard_prefactors_switch_global_session(monkeypatch):
    kmc = KMC.__new__(KMC)
    kmc.config = SimpleNamespace(
        rateconstant=SimpleNamespace(
            style="amsel-vtst",
            compute_vineyard_prefactor=True,
            vineyard_fd_step_A=1.0e-4,
        ),
        atomicenvironment=SimpleNamespace(rcut=3.0),
    )
    kmc.system = SimpleNamespace(
        types=["Cu", "Cu"],
        cell=np.eye(3) * 10.0,
    )
    full_forces = np.array(
        [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]],
        dtype=float,
    )

    class FakeGlobalSession:
        def __init__(self):
            self.global_mode = False
            self.use_global_calls = 0

        def use_global(self):
            self.global_mode = True
            self.use_global_calls += 1

    class FakeManager:
        def __init__(self):
            self.global_session = FakeGlobalSession()

        def get_forces(self, positions=None):
            return np.zeros((1, 3), dtype=float)

        def global_get_forces(self, positions=None):
            if not self.global_session.global_mode:
                return np.zeros((1, 3), dtype=float)
            return full_forces.reshape(-1)

    manager = FakeManager()
    kmc.manager = manager
    event = EventSearchOutput(
        central_atom_index=0,
        min1_positions=np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=float),
        saddle_positions=np.array([[1.1, 1.0, 1.0], [2.1, 1.0, 1.0]], dtype=float),
        min2_positions=np.array([[1.2, 1.0, 1.0], [2.2, 1.0, 1.0]], dtype=float),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 10.0,
    )

    def fake_prefactors(force_fn, min1, _saddle, _min2, **_kwargs):
        np.testing.assert_allclose(force_fn(np.asarray(min1, dtype=float)), full_forces)
        np.testing.assert_allclose(force_fn(np.asarray(min1, dtype=float)), full_forces)
        return SimpleNamespace(
            forward_prefactor_inv_s=1.1e13,
            backward_prefactor_inv_s=2.2e13,
            saddle_freq_invcm=120.0,
            barrier_omega_rad_per_s=2.4e13,
        )

    monkeypatch.setattr(
        kmc_module,
        "vineyard_event_prefactors_from_forces",
        fake_prefactors,
    )

    kmc._attach_vineyard_prefactors([event])

    assert manager.global_session.use_global_calls == 1
    assert event.prefactor_source == "vineyard-finite-difference"


def test_environments_with_cataloged_searches_tracks_valid_search_centers():
    event_outputs = [
        SimpleNamespace(central_atom_index=1),
        SimpleNamespace(central_atom_index=2),
    ]
    valid_results = [
        Ok(object()),
        Err(
            ErrorInfo(
                type=ErrorType.EVENT_NOT_NEW,
                message="duplicate catalog event",
            )
        ),
    ]

    assert environments_with_cataloged_searches(
        ["env-a", "env-b", "env-c"],
        event_outputs,
        valid_results,
    ) == {"env-b", "env-c"}


def test_environments_with_cataloged_searches_keeps_failed_search_centers_unvisited():
    event_outputs = [
        SimpleNamespace(central_atom_index=1),
        SimpleNamespace(central_atom_index=2),
    ]
    valid_results = [
        Err(
            ErrorInfo(
                type=ErrorType.EVENT_ENERGY_HIGHER_THAN_THRESHOLD,
                message="barrier too high",
            )
        ),
        Ok(object()),
    ]

    assert environments_with_cataloged_searches(
        ["env-a", "env-b", "env-c"],
        event_outputs,
        valid_results,
    ) == {"env-c"}


def test_event_search_process_evidence_counts_new_and_duplicate_processes():
    event_outputs = [
        SimpleNamespace(central_atom_index=1),
        SimpleNamespace(central_atom_index=1),
    ]
    process_key = (7, "env-b", "env-c")
    valid_results = [
        Ok(
            pd.DataFrame(
                [
                    {
                        "idx_ref": 7,
                        "event_id": "env-b",
                        "id_final": "env-c",
                        "k": 2.5,
                    }
                ]
            )
        ),
        Err(
            ErrorInfo(
                type=ErrorType.EVENT_NOT_NEW,
                message="duplicate catalog event",
                variables={
                    "matched_idx_ref": 7,
                    "event_id": "env-b",
                    "id_final": "env-c",
                    "k": 2.5,
                },
            )
        ),
    ]

    evidence = event_search_process_evidence(
        ["crystal", "env-b"],
        event_outputs,
        valid_results,
    )

    assert evidence["env-b"].attempts == 2
    assert evidence["env-b"].process_counts == Counter({process_key: 2})
    assert evidence["env-b"].process_rates == {process_key: 2.5}


def test_event_search_attempt_evidence_counts_failed_searches():
    process_key = (7, "env-b", "env-c")
    event_search_results = [
        Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="No event found")),
        Ok(SimpleNamespace(central_atom_index=1)),
    ]
    valid_results = [
        Err(
            ErrorInfo(
                type=ErrorType.EVENT_NOT_NEW,
                message="duplicate catalog event",
                variables={
                    "matched_idx_ref": 7,
                    "event_id": "env-b",
                    "id_final": "env-c",
                    "k": 2.5,
                },
            )
        )
    ]

    evidence = event_search_attempt_evidence(
        ["crystal", "env-b"],
        [1, 1],
        event_search_results,
        valid_results,
    )

    assert evidence["env-b"].attempts == 2
    assert evidence["env-b"].process_counts == Counter({process_key: 1})
    assert evidence["env-b"].process_rates == {process_key: 2.5}


def test_undercovered_environments_for_search_resamples_singleton_known_environment():
    evidence = EnvironmentSearchEvidence(
        attempts=1,
        process_counts=Counter({("process-0",): 1}),
    )

    assert undercovered_environments_for_search(
        current_environments=["crystal", "env-a", "env-a"],
        new_environments=[],
        visited_environments={"crystal", "env-a"},
        environment_search_evidence={"env-a": evidence},
    ) == ["env-a"]


def test_undercovered_environments_for_search_skips_duplicate_saturated_environment():
    evidence = EnvironmentSearchEvidence(
        attempts=2,
        process_counts=Counter({("process-0",): 2}),
    )

    assert undercovered_environments_for_search(
        current_environments=["env-a", "env-a", "crystal"],
        new_environments=[],
        visited_environments={"crystal", "env-a"},
        environment_search_evidence={"env-a": evidence},
    ) == []


def test_kmc_reference_search_repeats_current_environment_until_process_covered():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-a"]
    )
    kmc.visited_environments = set()
    kmc.environment_search_evidence = {}
    kmc.reference_table = SimpleNamespace(table=[object()])
    log_messages = []
    kmc.loggers = SimpleNamespace(
        info=lambda _name, message: log_messages.append(message)
    )
    kmc._close = lambda: None
    batches = []
    process_key = (7, "env-a", "env-b")

    class FakeEventSearch:
        def __init__(self, outputs):
            self.results = [Ok(output) for output in outputs]
            self._outputs = outputs

        def get_successes_results(self):
            return self._outputs

    def execute_event_searches(central_atoms):
        batches.append(list(central_atoms))
        return FakeEventSearch(
            [SimpleNamespace(central_atom_index=int(central_atoms[0]))]
        )

    def add_reference_events(_event_outputs):
        if len(batches) == 1:
            return [
                Ok(
                    pd.DataFrame(
                        [
                            {
                                "idx_ref": process_key[0],
                                "event_id": process_key[1],
                                "id_final": process_key[2],
                                "k": 2.5,
                            }
                        ]
                    )
                )
            ]
        return [
            Err(
                ErrorInfo(
                    type=ErrorType.EVENT_NOT_NEW,
                    message="duplicate catalog event",
                    variables={
                        "matched_idx_ref": process_key[0],
                        "event_id": process_key[1],
                        "id_final": process_key[2],
                        "k": 2.5,
                    },
                )
            )
        ]

    kmc.execute_event_searches = execute_event_searches
    kmc.add_reference_events = add_reference_events

    search_results, valid_results = kmc.search_reference_events_until_covered(
        ["env-a"], nsearch=1
    )

    assert len(batches) == 2
    assert len(search_results) == 2
    assert len(valid_results) == 2
    assert any("Resampling 1 undercovered atomic environments" in message for message in log_messages)
    assert any(
        "attempts=2; observations=2; unique_processes=1; "
        "singleton_processes=0; missing_process_mass=0.000000e+00" in message
        and "needs_more_search=False" in message
        for message in log_messages
    )
    assert kmc.environment_search_evidence["env-a"].process_counts == Counter(
        {process_key: 2}
    )


def test_kmc_reference_search_bounds_singleton_process_resampling():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-a"]
    )
    kmc.visited_environments = set()
    kmc.environment_search_evidence = {}
    kmc.reference_table = SimpleNamespace(table=[object()])
    kmc.loggers = SimpleNamespace(info=lambda *_args: None)
    kmc._close = lambda: None
    batches = []
    attempt_limit = 8

    class FakeEventSearch:
        def __init__(self, outputs):
            self.results = [Ok(output) for output in outputs]
            self._outputs = outputs

        def get_successes_results(self):
            return self._outputs

    def execute_event_searches(central_atoms):
        batches.append(list(central_atoms))
        if len(batches) > attempt_limit + 1:
            raise AssertionError("coverage resampling did not stop")
        return FakeEventSearch(
            [SimpleNamespace(central_atom_index=int(central_atoms[0]))]
        )

    def add_reference_events(_event_outputs):
        process_index = len(batches)
        return [
            Ok(
                pd.DataFrame(
                    [
                        {
                            "idx_ref": process_index,
                            "event_id": "env-a",
                            "id_final": f"env-{process_index}",
                            "k": 1.0,
                        }
                    ]
                )
            )
        ]

    kmc.execute_event_searches = execute_event_searches
    kmc.add_reference_events = add_reference_events

    search_results, valid_results = kmc.search_reference_events_until_covered(
        ["env-a"], nsearch=1
    )

    assert len(batches) == attempt_limit
    assert len(search_results) == attempt_limit
    assert len(valid_results) == attempt_limit
    evidence = kmc.environment_search_evidence["env-a"]
    assert evidence.attempts == attempt_limit
    assert sum(evidence.process_counts.values()) == attempt_limit
    assert environment_search_evidence_trace_lines({"env-a": evidence})[0].endswith(
        "needs_more_search=True"
    )


def test_kmc_reference_search_respects_disabled_coverage_resampling():
    kmc = KMC(
        SimpleNamespace(
            control=SimpleNamespace(
                random_seed=12345,
                disable_coverage_resampling=True,
            )
        )
    )
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-a"]
    )
    kmc.visited_environments = set()
    kmc.environment_search_evidence = {}
    kmc.reference_table = SimpleNamespace(table=[object()])
    kmc.loggers = SimpleNamespace(info=lambda *_args: None)
    kmc._close = lambda: None
    batches = []
    process_key = (7, "env-a", "env-b")

    class FakeEventSearch:
        def __init__(self, outputs):
            self.results = [Ok(output) for output in outputs]
            self._outputs = outputs

        def get_successes_results(self):
            return self._outputs

    def execute_event_searches(central_atoms):
        batches.append(list(central_atoms))
        return FakeEventSearch(
            [SimpleNamespace(central_atom_index=int(central_atoms[0]))]
        )

    def add_reference_events(_event_outputs):
        return [
            Ok(
                pd.DataFrame(
                    [
                        {
                            "idx_ref": process_key[0],
                            "event_id": process_key[1],
                            "id_final": process_key[2],
                            "k": 2.5,
                        }
                    ]
                )
            )
        ]

    kmc.execute_event_searches = execute_event_searches
    kmc.add_reference_events = add_reference_events

    search_results, valid_results = kmc.search_reference_events_until_covered(
        ["env-a"], nsearch=1
    )

    assert len(batches) == 1
    assert len(search_results) == 1
    assert len(valid_results) == 1
    assert kmc.environment_search_evidence["env-a"].process_counts == Counter(
        {process_key: 1}
    )


def test_kmc_reference_search_checks_amsel_between_search_rounds(monkeypatch):
    class CompleteCertificate:
        attempts = 1
        observations = 1
        unique_processes = 1
        singleton_processes = 0
        unseen_process_probability = 0.0
        missing_rate_mass_estimate = 0.0
        needs_more_search = False

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=lambda **_kwargs: CompleteCertificate()),
    )
    kmc_module._process_search_certificate_cache_clear()
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))
    kmc.atomic_environment = SimpleNamespace(atomic_environment_list=["env-a"])
    kmc.visited_environments = set()
    kmc.environment_search_evidence = {}
    kmc.reference_table = SimpleNamespace(table=[object()])
    kmc.loggers = SimpleNamespace(info=lambda *_args: None)
    kmc._close = lambda: None
    batches = []
    process_key = (7, "env-a", "env-b")

    class FakeEventSearch:
        results = [Ok(SimpleNamespace(central_atom_index=0))]

        @staticmethod
        def get_successes_results():
            return [SimpleNamespace(central_atom_index=0)]

    def execute_event_searches(central_atoms):
        batches.append(list(central_atoms))
        return FakeEventSearch()

    def add_reference_events(_event_outputs):
        return [
            Ok(
                pd.DataFrame(
                    [
                        {
                            "idx_ref": process_key[0],
                            "event_id": process_key[1],
                            "id_final": process_key[2],
                            "k": 2.5,
                        }
                    ]
                )
            )
        ]

    kmc.execute_event_searches = execute_event_searches
    kmc.add_reference_events = add_reference_events

    search_results, valid_results = kmc.search_reference_events_until_covered(
        ["env-a"], nsearch=2
    )

    assert batches == [[0]]
    assert len(search_results) == 1
    assert len(valid_results) == 1


def test_kmc_reference_search_counts_failed_resampling_attempts(monkeypatch):
    class FakeCertificate:
        def __init__(self, attempts):
            self.attempts = attempts
            self.observations = 1
            self.unique_processes = 1
            self.singleton_processes = 1
            self.unseen_process_probability = 1.0 if attempts < 2 else 0.0
            self.missing_rate_mass_estimate = 1.0 if attempts < 2 else 0.0
            self.needs_more_search = attempts < 2

    def event_completeness(**kwargs):
        return FakeCertificate(int(kwargs["attempts"]))

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=event_completeness),
    )
    kmc_module._process_search_certificate_cache_clear()
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))
    kmc.atomic_environment = SimpleNamespace(
        atomic_environment_list=["env-a", "env-a"]
    )
    process_key = (7, "env-a", "env-b")
    kmc.visited_environments = {"env-a"}
    kmc.environment_search_evidence = {
        "env-a": EnvironmentSearchEvidence(
            attempts=1,
            process_counts=Counter({process_key: 1}),
            process_rates={process_key: 2.5},
        )
    }
    kmc.reference_table = SimpleNamespace(table=[object()])
    kmc.loggers = SimpleNamespace(info=lambda *_args: None)
    kmc._close = lambda: None
    batches = []

    class FakeEventSearch:
        def __init__(self):
            self.results = [
                Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="No event found"))
            ]

        def get_successes_results(self):
            return []

    def execute_event_searches(central_atoms):
        batches.append(list(central_atoms))
        return FakeEventSearch()

    kmc.execute_event_searches = execute_event_searches
    kmc.add_reference_events = lambda _event_outputs: []

    search_results, valid_results = kmc.search_reference_events_until_covered(
        [], nsearch=1
    )

    assert len(batches) == 1
    assert len(search_results) == 1
    assert len(valid_results) == 0
    assert kmc.environment_search_evidence["env-a"].attempts == 2
    assert kmc.environment_search_evidence["env-a"].process_counts == Counter(
        {process_key: 1}
    )


def test_kmc_reference_search_spends_nsearch_before_zero_event_abort():
    kmc = KMC(SimpleNamespace(control=SimpleNamespace(random_seed=12345)))
    kmc.atomic_environment = SimpleNamespace(atomic_environment_list=["env-a"])
    kmc.visited_environments = set()
    kmc.environment_search_evidence = {}
    kmc.reference_table = SimpleNamespace(table=[])
    kmc.loggers = SimpleNamespace(
        info=lambda *_args: None,
        error=lambda *_args: None,
    )
    batches = []

    class FakeEventSearch:
        results = [Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="not found"))]

        @staticmethod
        def get_successes_results():
            return []

    def execute_event_searches(central_atoms):
        batches.append(list(central_atoms))
        return FakeEventSearch()

    def close():
        raise RuntimeError("closed")

    kmc.execute_event_searches = execute_event_searches
    kmc.add_reference_events = lambda _event_outputs: []
    kmc._close = close

    with pytest.raises(RuntimeError, match="closed"):
        kmc.search_reference_events_until_covered(["env-a"], nsearch=3)

    assert batches == [[0], [0], [0]]
    assert kmc.environment_search_evidence["env-a"].attempts == 3


def test_environment_search_evidence_trace_lines_report_missing_process_mass():
    evidence = {
        b"\x01\x02long-environment-signature": EnvironmentSearchEvidence(
            attempts=1,
            process_counts=Counter({("process-0",): 1}),
            process_rates={("process-0",): 4.0},
        )
    }

    assert environment_search_evidence_trace_lines(evidence) == [
        "\t :=> AMSEL process coverage env=01026c6f6e672d65; "
        "attempts=1; observations=1; unique_processes=1; "
        "singleton_processes=1; missing_process_mass=1.000000e+00; "
        "missing_rate_mass=inf; needs_more_search=True"
    ]


def test_negligible_missing_rate_mass_does_not_trigger_more_process_search(
    monkeypatch,
):
    class FakeCertificate:
        attempts = 8
        observations = 8
        unique_processes = 1
        singleton_processes = 0
        unseen_process_probability = 0.09316770
        missing_rate_mass_estimate = 5.0e-20
        needs_more_search = True

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=lambda **kwargs: FakeCertificate()),
    )
    kmc_module._process_search_certificate_cache_clear()
    evidence = {
        "slow-env": EnvironmentSearchEvidence(
            attempts=8,
            process_counts=Counter({("slow-process",): 8}),
            process_rates={("slow-process",): 5.0e-20},
        )
    }

    assert undercovered_environments_for_search(
        current_environments=["slow-env"],
        new_environments=[],
        visited_environments={"slow-env"},
        environment_search_evidence=evidence,
    ) == []
    assert "needs_more_search=False" in environment_search_evidence_trace_lines(
        evidence
    )[0]


def test_small_missing_rate_mass_does_not_resample_against_active_rate_scale(monkeypatch):
    class FakeCertificate:
        attempts = 34
        observations = 10
        unique_processes = 3
        singleton_processes = 1
        unseen_process_probability = 0.02941176
        missing_rate_mass_estimate = 1.5e-3
        needs_more_search = True

    class CompleteCertificate:
        attempts = 2
        observations = 2
        unique_processes = 1
        singleton_processes = 0
        unseen_process_probability = 0.0
        missing_rate_mass_estimate = 0.0
        needs_more_search = False

    def event_completeness(**kwargs):
        if ("fast",) in kwargs["process_counts"]:
            return CompleteCertificate()
        return FakeCertificate()

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=event_completeness),
    )
    kmc_module._process_search_certificate_cache_clear()
    evidence = {
        "slow-gap-env": EnvironmentSearchEvidence(
            attempts=34,
            process_counts=Counter(
                {
                    ("slow-a",): 8,
                    ("slow-b",): 1,
                    ("slow-c",): 1,
                }
            ),
            process_rates={
                ("slow-a",): 1.0e-3,
                ("slow-b",): 7.0e-4,
                ("slow-c",): 5.0e-4,
            },
        ),
        "fast-env": EnvironmentSearchEvidence(
            attempts=2,
            process_counts=Counter({("fast",): 2}),
            process_rates={("fast",): 1.0},
        ),
    }

    assert undercovered_environments_for_search(
        current_environments=["slow-gap-env", "fast-env", "fast-env"],
        new_environments=[],
        visited_environments={"slow-gap-env", "fast-env"},
        environment_search_evidence=evidence,
    ) == []


def test_zero_observation_environment_is_not_rate_scaled_away(monkeypatch):
    class CompleteCertificate:
        attempts = 2
        observations = 2
        unique_processes = 1
        singleton_processes = 0
        unseen_process_probability = 0.0
        missing_rate_mass_estimate = 0.0
        needs_more_search = False

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=lambda **kwargs: CompleteCertificate()),
    )
    kmc_module._process_search_certificate_cache_clear()
    evidence = {
        "zero-env": EnvironmentSearchEvidence(attempts=1),
        "known-env": EnvironmentSearchEvidence(
            attempts=2,
            process_counts=Counter({("known",): 2}),
            process_rates={("known",): 1.0},
        ),
    }

    assert undercovered_environments_for_search(
        current_environments=["zero-env", "known-env"],
        new_environments=[],
        visited_environments={"known-env"},
        environment_search_evidence=evidence,
        zero_observation_attempt_limit=10,
    ) == ["zero-env"]


def test_zero_observation_new_environment_obeys_attempt_limit(monkeypatch):
    class CompleteCertificate:
        attempts = 2
        observations = 2
        unique_processes = 1
        singleton_processes = 0
        unseen_process_probability = 0.0
        missing_rate_mass_estimate = 0.0
        needs_more_search = False

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=lambda **kwargs: CompleteCertificate()),
    )
    kmc_module._process_search_certificate_cache_clear()
    evidence = {"zero-env": EnvironmentSearchEvidence(attempts=3)}

    assert undercovered_environments_for_search(
        current_environments=["zero-env"],
        new_environments=["zero-env"],
        visited_environments={"crystal"},
        environment_search_evidence=evidence,
        zero_observation_attempt_limit=3,
    ) == []


def test_process_coverage_trace_reports_rate_scale_search_decision(monkeypatch):
    class FakeCertificate:
        attempts = 34
        observations = 10
        unique_processes = 3
        singleton_processes = 1
        unseen_process_probability = 0.02941176
        missing_rate_mass_estimate = 1.5e-3
        needs_more_search = True

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=lambda **kwargs: FakeCertificate()),
    )
    kmc_module._process_search_certificate_cache_clear()
    evidence = {
        "slow-gap-env": EnvironmentSearchEvidence(
            attempts=34,
            process_counts=Counter(
                {
                    ("slow-a",): 8,
                    ("slow-b",): 1,
                    ("slow-c",): 1,
                }
            ),
            process_rates={
                ("slow-a",): 1.0e-3,
                ("slow-b",): 7.0e-4,
                ("slow-c",): 5.0e-4,
            },
        )
    }

    assert "needs_more_search=False" in environment_search_evidence_trace_lines(
        evidence,
        known_rate_scale=2.0,
    )[0]


def test_rate_material_missing_mass_still_resamples_environment(monkeypatch):
    class FakeCertificate:
        attempts = 4
        observations = 2
        unique_processes = 1
        singleton_processes = 1
        unseen_process_probability = 0.25
        missing_rate_mass_estimate = 2.0e-1
        needs_more_search = True

    class CompleteCertificate:
        attempts = 2
        observations = 2
        unique_processes = 1
        singleton_processes = 0
        unseen_process_probability = 0.0
        missing_rate_mass_estimate = 0.0
        needs_more_search = False

    def event_completeness(**kwargs):
        if ("known",) in kwargs["process_counts"]:
            return CompleteCertificate()
        return FakeCertificate()

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=event_completeness),
    )
    kmc_module._process_search_certificate_cache_clear()
    evidence = {
        "gap-env": EnvironmentSearchEvidence(
            attempts=4,
            process_counts=Counter({("gap",): 1}),
            process_rates={("gap",): 0.2},
        ),
        "known-env": EnvironmentSearchEvidence(
            attempts=2,
            process_counts=Counter({("known",): 2}),
            process_rates={("known",): 1.0},
        ),
    }

    assert undercovered_environments_for_search(
        current_environments=["gap-env", "known-env"],
        new_environments=[],
        visited_environments={"gap-env", "known-env"},
        environment_search_evidence=evidence,
    ) == ["gap-env"]


def test_process_search_certificate_uses_finite_catalog_estimator(monkeypatch):
    captured = {}

    class FakeCertificate:
        attempts = 2
        observations = 2
        unique_processes = 1
        singleton_processes = 0
        unseen_process_probability = 0.0
        missing_rate_mass_estimate = 0.0
        needs_more_search = False

    def event_completeness(**kwargs):
        captured.update(kwargs)
        return FakeCertificate()

    monkeypatch.setattr(
        kmc_module,
        "_amsel",
        SimpleNamespace(event_completeness=event_completeness),
    )
    kmc_module._process_search_certificate_cache_clear()

    evidence = EnvironmentSearchEvidence(
        attempts=2,
        process_counts=Counter({("hop",): 2}),
        process_rates={("hop",): 1.0},
    )

    assert kmc_module._process_search_certificate(evidence)["needs_more_search"] is False
    assert captured["use_py_heavy_tail"] is False


def test_basin_exploration_trace_line_reports_order_queue_and_guidance():
    basin = SimpleNamespace(
        exploration_order=[0, 13],
        states_to_explore=[14, 1],
        last_exploration_guidance={14: 0.375, 1: 0.25},
        exploration_decisions=[
            {
                "state": 0,
                "event_family": None,
                "process_signature": None,
                "guidance": 0.0,
            },
            {
                "state": 13,
                "event_family": 1,
                "process_signature": "1:3330",
                "guidance": 0.375,
            },
        ],
    )

    assert basin_exploration_trace_line(basin) == (
        "\t :=> Basin exploration trace order=0,13; "
        "queue=14,1; guidance=14:3.750000e-01,1:2.500000e-01; "
        "closed_events=0:NA,13:1; "
        "closed_processes=0:NA,13:1:3330; "
        "closed_guidance=0:0.000000e+00,13:3.750000e-01"
    )


def test_basin_exploration_trace_line_skips_empty_trace():
    assert basin_exploration_trace_line(SimpleNamespace()) is None
