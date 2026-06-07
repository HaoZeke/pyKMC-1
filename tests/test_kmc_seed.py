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
        def get_forces(self, positions=None):
            calls.append(("local", np.asarray(positions, dtype=float).shape))
            return SimpleNamespace(result=lambda: np.zeros((1, 3), dtype=float))

        def global_get_forces(self, positions=None):
            calls.append(("global", np.asarray(positions, dtype=float).shape))
            return full_forces.reshape(-1)

    kmc.manager = FakeManager()
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

    assert calls == [("global", (2, 3))]
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
            ],
            dtype=float,
        ),
        saddle_positions=np.array(
            [
                [1.0, 1.0, 1.0],
                [2.2, 1.0, 1.0],
                [3.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        min2_positions=np.array(
            [
                [1.0, 1.0, 1.0],
                [2.4, 1.0, 1.0],
                [3.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        dE_forward=0.2,
        dE_backward=0.3,
        move_atom_index=0,
        cell=np.eye(3) * 20.0,
    )

    assert kmc._vineyard_active_indices(event) == [0, 1]


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


def test_missing_process_mass_resamples_environment_even_when_rate_mass_is_small(monkeypatch):
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
    ) == ["slow-gap-env"]


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


def test_process_coverage_trace_reports_strict_search_decision(monkeypatch):
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

    assert "needs_more_search=True" in environment_search_evidence_trace_lines(
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
