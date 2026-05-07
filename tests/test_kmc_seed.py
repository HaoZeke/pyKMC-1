import random
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd

from pykmc.kmc import (
    EnvironmentSearchEvidence,
    KMC,
    basin_exploration_trace_line,
    environment_search_evidence_trace_lines,
    environments_with_cataloged_searches,
    event_search_process_evidence,
    undercovered_environments_for_search,
)
from pykmc.result import Err, ErrorInfo, ErrorType, Ok


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
    assert kmc.environment_search_evidence["env-a"].process_counts == Counter(
        {process_key: 2}
    )


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
