import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from pykmc import System
from pykmc.basins import (
    AmselFPTASelector,
    BasinStatesConnectivity,
    BasinsGenericEvents,
    FPTASelector,
    StateData,
)
import logging
from pykmc.enginemanager.lmpi.pool import ManagerFactory
import pykmc.basins.basin as basin_module
from pykmc.result import BasinSelectorOutput, Err, ErrorInfo, ErrorType, Ok

logger = logging.getLogger("tests")


def test_frontier_state_searches_unknown_environments(monkeypatch):
    known = "known-env"
    unknown = "unknown-env"
    event_output = SimpleNamespace(central_atom_index=1)

    class FakeEventSearch:
        def __init__(self, config, system, manager, loggers):
            self.config = config
            self.system = system
            self.manager = manager
            self.loggers = loggers
            self.results = [Ok(event_output)]

        def execute(self, central_atom_research_list):
            self.central_atom_research_list = list(central_atom_research_list)

        def get_successes_results(self):
            return [event_output]

    class FakeReferenceTable:
        def __init__(self):
            self.added_events = None

        def add_events(self, events):
            self.added_events = list(events)
            return [Ok(pd.DataFrame({"idx_ref": [0]}))]

    prefactor_events = []
    manager = SimpleNamespace(use_global=lambda: None)
    config = SimpleNamespace(
        basin=SimpleNamespace(frontier_event_searches=1),
        partn=SimpleNamespace(amsel_recomb_seed=False),
    )
    state = StateData(
        system=System(
            positions=np.zeros((2, 3)),
            types=np.array(["Cu", "Cu"]),
            cell=np.eye(3) * 10.0,
            pbc=True,
            index=np.arange(2),
        ),
        environment=SimpleNamespace(atomic_environment_list=[known, unknown]),
        neighbors_list=None,
        transient=True,
    )
    reference_table = FakeReferenceTable()
    basin = BasinsGenericEvents(
        config=config,
        reference_table=reference_table,
        known_environments={known},
        manager=manager,
        prefactor_attacher=lambda events: prefactor_events.extend(events),
    )

    monkeypatch.setattr(basin_module, "EventSearch", FakeEventSearch)

    searched = basin._try_search_unknown_state_environments(state)

    assert searched is True
    assert reference_table.added_events == [event_output]
    assert prefactor_events == [event_output]
    assert basin.known_environments == {known, unknown}


def test_frontier_state_search_caps_and_restores_partn_evals(monkeypatch):
    known = "known-env"
    unknown = "unknown-env"
    event_output = SimpleNamespace(central_atom_index=1)
    seen_limits = []

    class FakeEventSearch:
        def __init__(self, config, system, manager, loggers):
            self.config = config
            self.results = [Ok(event_output)]

        def execute(self, central_atom_research_list):
            seen_limits.append(
                (
                    int(self.config.partn.nevalf_max),
                    int(self.config.partn.evalf_max),
                )
            )

        def get_successes_results(self):
            return [event_output]

    class FakeReferenceTable:
        def add_events(self, events):
            return [Ok(pd.DataFrame({"idx_ref": [0]}))]

    config = SimpleNamespace(
        basin=SimpleNamespace(
            frontier_event_searches=1,
            frontier_search_nevalf_max=80,
        ),
        partn=SimpleNamespace(
            amsel_recomb_seed=False,
            nevalf_max=1200,
            evalf_max=2400,
        ),
    )
    state = StateData(
        system=System(
            positions=np.zeros((2, 3)),
            types=np.array(["Cu", "Cu"]),
            cell=np.eye(3) * 10.0,
            pbc=True,
            index=np.arange(2),
        ),
        environment=SimpleNamespace(atomic_environment_list=[known, unknown]),
        neighbors_list=None,
        transient=True,
    )
    log_messages = []
    basin = BasinsGenericEvents(
        config=config,
        reference_table=FakeReferenceTable(),
        known_environments={known},
        manager=SimpleNamespace(use_global=lambda: None),
        loggers=SimpleNamespace(
            info=lambda _name, message: log_messages.append(message)
        ),
    )

    monkeypatch.setattr(basin_module, "EventSearch", FakeEventSearch)

    assert basin._try_search_unknown_state_environments(state) is True
    assert seen_limits == [(80, 80)]
    assert config.partn.nevalf_max == 1200
    assert config.partn.evalf_max == 2400
    assert any(
        "AMSEL frontier pARTn force-evaluation limits nevalf_max=80 evalf_max=80"
        in message
        for message in log_messages
    )


def test_frontier_recomb_search_center_uses_configured_capture_radius(monkeypatch):
    seen = {}
    config = SimpleNamespace(
        partn=SimpleNamespace(
            amsel_recomb_seed=True,
            amsel_recomb_capture_mult=1.25,
        )
    )
    basin = BasinsGenericEvents(
        config=config,
        reference_table=None,
        known_environments=set(),
        manager=None,
    )
    state = StateData(
        system=System(
            positions=np.zeros((2, 3)),
            types=np.array(["Cu", "Cu"]),
            cell=np.eye(3) * 10.0,
            pbc=True,
            index=np.arange(2),
        ),
        environment=None,
        neighbors_list=None,
        transient=True,
    )
    monkeypatch.setattr(
        "pykmc.basins.amsel_recomb.recombination_search_center",
        lambda positions, cell, capture_mult=None: seen.setdefault(
            "capture_mult", capture_mult
        ),
        raising=False,
    )

    assert basin._frontier_recomb_search_center(state) == 1.25


def test_frontier_state_search_uses_local_pool_and_restores_global(monkeypatch):
    known = "known-env"
    unknown = "unknown-env"
    event_output = SimpleNamespace(central_atom_index=1)

    class FakeManager:
        def __init__(self):
            self.using_global = True
            self.calls = []

        def use_local(self):
            self.calls.append("use_local")
            self.using_global = False

        def use_global(self):
            self.calls.append("use_global")
            self.using_global = True

    manager = FakeManager()

    class FakeEventSearch:
        def __init__(self, config, system, manager, loggers):
            self.config = config
            self.manager = manager
            self.results = [Ok(event_output)]

        def execute(self, central_atom_research_list):
            assert self.manager.using_global is False

        def get_successes_results(self):
            return [event_output]

    class FakeReferenceTable:
        def add_events(self, events):
            return [Ok(pd.DataFrame({"idx_ref": [0]}))]

    config = SimpleNamespace(
        basin=SimpleNamespace(
            frontier_event_searches=1,
            frontier_search_nevalf_max=80,
        ),
        partn=SimpleNamespace(
            amsel_recomb_seed=False,
            nevalf_max=1200,
            evalf_max=2400,
        ),
    )
    state = StateData(
        system=System(
            positions=np.zeros((2, 3)),
            types=np.array(["Cu", "Cu"]),
            cell=np.eye(3) * 10.0,
            pbc=True,
            index=np.arange(2),
        ),
        environment=SimpleNamespace(atomic_environment_list=[known, unknown]),
        neighbors_list=None,
        transient=True,
    )
    basin = BasinsGenericEvents(
        config=config,
        reference_table=FakeReferenceTable(),
        known_environments={known},
        manager=manager,
    )

    monkeypatch.setattr(basin_module, "EventSearch", FakeEventSearch)

    assert basin._try_search_unknown_state_environments(state) is True
    assert manager.calls == ["use_local", "use_global"]
    assert manager.using_global is True


def test_frontier_state_search_suppresses_duplicate_failed_environments(monkeypatch):
    unknown = "unknown-env"
    executions = []

    class FakeEventSearch:
        def __init__(self, config, system, manager, loggers):
            self.config = config

        def execute(self, central_atom_research_list):
            executions.append(list(central_atom_research_list))

        def get_successes_results(self):
            return []

    class FakeReferenceTable:
        def add_events(self, events):
            return []

    config = SimpleNamespace(
        basin=SimpleNamespace(
            frontier_event_searches=1,
            frontier_search_nevalf_max=80,
        ),
        partn=SimpleNamespace(
            amsel_recomb_seed=False,
            nevalf_max=1200,
            evalf_max=2400,
        ),
    )

    def make_state():
        return StateData(
            system=System(
                positions=np.zeros((1, 3)),
                types=np.array(["Cu"]),
                cell=np.eye(3) * 10.0,
                pbc=True,
                index=np.arange(1),
            ),
            environment=SimpleNamespace(atomic_environment_list=[unknown]),
            neighbors_list=None,
            transient=True,
        )

    basin = BasinsGenericEvents(
        config=config,
        reference_table=FakeReferenceTable(),
        known_environments=set(),
        manager=SimpleNamespace(use_local=lambda: None, use_global=lambda: None),
    )

    monkeypatch.setattr(basin_module, "EventSearch", FakeEventSearch)

    assert (
        basin._try_search_unknown_state_environments(make_state(), state_index=1)
        is False
    )
    assert (
        basin._try_search_unknown_state_environments(make_state(), state_index=2)
        is False
    )
    assert executions == [[0]]


def test_frontier_state_search_budget_limits_total_search_centers(monkeypatch):
    executions = []

    class FakeEventSearch:
        def __init__(self, config, system, manager, loggers):
            self.config = config

        def execute(self, central_atom_research_list):
            executions.append(list(central_atom_research_list))

        def get_successes_results(self):
            return []

    class FakeReferenceTable:
        def add_events(self, events):
            return []

    config = SimpleNamespace(
        basin=SimpleNamespace(
            frontier_event_searches=1,
            frontier_search_nevalf_max=80,
        ),
        partn=SimpleNamespace(
            amsel_recomb_seed=False,
            nevalf_max=1200,
            evalf_max=2400,
        ),
    )
    state = StateData(
        system=System(
            positions=np.zeros((3, 3)),
            types=np.array(["Cu", "Cu", "Cu"]),
            cell=np.eye(3) * 10.0,
            pbc=True,
            index=np.arange(3),
        ),
        environment=SimpleNamespace(
            atomic_environment_list=["unknown-a", "unknown-b", "unknown-c"]
        ),
        neighbors_list=None,
        transient=True,
    )
    basin = BasinsGenericEvents(
        config=config,
        reference_table=FakeReferenceTable(),
        known_environments=set(),
        manager=SimpleNamespace(use_local=lambda: None, use_global=lambda: None),
    )

    monkeypatch.setattr(basin_module, "EventSearch", FakeEventSearch)

    assert (
        basin._try_search_unknown_state_environments(state, state_index=1) is False
    )
    assert executions == [[0]]


class TestBasin : 

    def test_connectivity_table_construction(self, test_logger, config_Cu, reference_table_Cu_fake, system_Cu, visited_environments_Cu) : 
        
        #Create Manager
        factory = ManagerFactory(n_sessions=config_Cu.control.n_sessions, use_rank_0=True)
        manager = factory.launch()

        if manager is not None: #On rank 0
            manager.initialize_sessions(config_Cu, system_Cu)

            self.basin = BasinsGenericEvents(config=config_Cu, reference_table=reference_table_Cu_fake, known_environments=visited_environments_Cu, manager = None)
            self.basin.manager = manager

            result = self.basin.execute(system=system_Cu)
            if result.is_ok() : 
                test_logger.debug("Find Exit State : ")
                test_logger.debug("Exit time t_exit = {}ps".format(result.ok_value().t_exit))
                test_logger.debug("Exit state n : {}".format(result.ok_value().exit_state))
            else : 
                test_logger.debug("Error: {}".format(result.err_value()))
            
            manager.close_all()

    def test_basin_initializes_configured_legacy_selector(
        self, config_Cu, reference_table_Cu_fake, visited_environments_Cu, system_Cu
    ):
        config_Cu.basin.selector = "legacy-fpta"
        basin = BasinsGenericEvents(
            config=config_Cu,
            reference_table=reference_table_Cu_fake,
            known_environments=visited_environments_Cu,
            manager=None,
        )

        basin._initialize(system_Cu)

        assert isinstance(basin.selector, FPTASelector)

    def test_basin_initializes_configured_amsel_sampled_selector(
        self, config_Cu, reference_table_Cu_fake, visited_environments_Cu, system_Cu
    ):
        pytest.importorskip("amsel")
        config_Cu.basin.selector = "amsel-sampled"
        basin = BasinsGenericEvents(
            config=config_Cu,
            reference_table=reference_table_Cu_fake,
            known_environments=visited_environments_Cu,
            manager=None,
        )

        basin._initialize(system_Cu)

        assert isinstance(basin.selector, AmselFPTASelector)
        assert basin.selector.clock_mode == "sampled"

    def test_basin_auto_uses_sampled_amsel_selector_when_available(
        self, config_Cu, reference_table_Cu_fake, visited_environments_Cu, system_Cu
    ):
        pytest.importorskip("amsel")
        config_Cu.basin.selector = "auto"
        basin = BasinsGenericEvents(
            config=config_Cu,
            reference_table=reference_table_Cu_fake,
            known_environments=visited_environments_Cu,
            manager=None,
        )

        basin._initialize(system_Cu)

        assert isinstance(basin.selector, AmselFPTASelector)
        assert basin.selector.clock_mode == "sampled"

    def test_basin_auto_falls_back_to_legacy_selector_without_amsel(
        self,
        monkeypatch,
        config_Cu,
        reference_table_Cu_fake,
        visited_environments_Cu,
        system_Cu,
    ):
        monkeypatch.setattr(basin_module, "_AMSEL_AVAILABLE", False)
        config_Cu.basin.selector = "auto"
        basin = BasinsGenericEvents(
            config=config_Cu,
            reference_table=reference_table_Cu_fake,
            known_environments=visited_environments_Cu,
            manager=None,
        )

        basin._initialize(system_Cu)

        assert isinstance(basin.selector, FPTASelector)

    def test_update_to_explore_prioritizes_amsel_frontier_states(self):
        pytest.importorskip("amsel")
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 13,
                    "event_connexion": 1,
                    "central_atom": 3330,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 14,
                    "event_connexion": 1,
                    "central_atom": 3330,
                    "sym": 2,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 0,
                    "central_atom": 9,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 2.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(exploration_priority="amsel")
        )
        basin.connectivity_table = table
        basin.explored_states = [0]

        basin.update_to_explore()

        assert basin.states_to_explore[:2] == [13, 14]
        assert basin.last_exploration_guidance[13] == pytest.approx(0.375)
        assert basin.last_exploration_guidance[14] == pytest.approx(0.375)
        assert basin.last_exploration_guidance[1] == pytest.approx(0.25)

    def test_update_to_explore_uses_process_signature_novelty_as_tiebreaker(
        self, monkeypatch
    ):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 13,
                    "event_connexion": 1,
                    "central_atom": 3330,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 14,
                    "event_connexion": 1,
                    "central_atom": 4444,
                    "sym": 2,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 33,
                    "event_connexion": 2,
                    "central_atom": 910,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 0.5,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 13,
                    "state_connexion": 40,
                    "event_connexion": 3,
                    "central_atom": 912,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 0.25,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )
        monkeypatch.setattr(
            basin_module,
            "amsel_state_guidance_scores",
            lambda connectivity_table, entry=0: {14: 0.9, 33: 0.9, 40: 0.1},
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(exploration_priority="amsel-diverse")
        )
        basin.connectivity_table = table
        basin.explored_states = [0, 13]

        basin.update_to_explore()

        assert basin.states_to_explore == [14, 33, 40]
        assert basin.last_exploration_guidance[14] == pytest.approx(0.9)

    def test_update_to_explore_keeps_guidance_before_event_family_novelty(
        self, monkeypatch
    ):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 14,
                    "event_connexion": 1,
                    "central_atom": 3330,
                    "sym": 2,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 33,
                    "event_connexion": 2,
                    "central_atom": 910,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 0.5,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 13,
                    "state_connexion": 40,
                    "event_connexion": 3,
                    "central_atom": 912,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 0.25,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )
        monkeypatch.setattr(
            basin_module,
            "amsel_state_guidance_scores",
            lambda connectivity_table, entry=0: {14: 0.9, 33: 0.2, 40: 0.1},
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(exploration_priority="amsel-diverse")
        )
        basin.connectivity_table = table
        basin.explored_states = [0, 13]

        basin.update_to_explore()

        assert basin.states_to_explore == [14, 33, 40]

    def test_record_exploration_decision_captures_event_family_and_guidance(self):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 13,
                    "event_connexion": 7,
                    "central_atom": 3330,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.connectivity_table = table
        basin.last_exploration_guidance = {13: 0.375}
        basin.exploration_decisions = []

        basin._record_exploration_decision(13)

        assert basin.exploration_decisions == [
            {
                "state": 13,
                "event_family": 7,
                "process_signature": "7:3330",
                "guidance": pytest.approx(0.375),
            }
        ]

    def test_construct_connexion_table_can_stop_after_expansion_budget(self):
        class FakeState:
            def ensure_full_state(self, config):
                pass

            def release_heavy_objects(self):
                pass

        class FakeExplorer:
            def __init__(self):
                self.connectivity_table = BasinStatesConnectivity()
                self.connectivity_table.df = pd.DataFrame(
                    [
                        {
                            "state": 0,
                            "state_connexion": 1,
                            "event_connexion": 1,
                            "central_atom": 0,
                            "sym": 0,
                            "transient": True,
                            "dE_forward": 0.0,
                            "k_forward": 3.0,
                            "dE_backward": 0.0,
                            "k_backward": 0.0,
                        }
                    ]
                )
                self.explored = []

            def explore(self, state, state_index, start_index):
                self.explored.append(state_index)

            def clear(self):
                pass

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(exploration_priority="legacy")
        )
        basin.current_state = 0
        basin.states_to_explore = [0]
        basin.explored_states = []
        basin.states = {0: FakeState()}
        basin.connectivity_table = BasinStatesConnectivity()
        basin.explorer = FakeExplorer()

        result = basin.construct_connexion_table(
            max_expansions=10,
            max_closed_states=1,
        )

        assert result.is_ok()
        assert basin.exploration_order == [0]
        assert basin.explorer.explored == [0]
        assert basin.states_to_explore == [1]

    def test_absorbing_refinement_budget_uses_amsel_outlet_committor(
        self, monkeypatch
    ):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 10.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 2,
                    "event_connexion": 2,
                    "central_atom": 20,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 3,
                    "event_connexion": 3,
                    "central_atom": 30,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 5.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )

        class FakeSelector:
            def diagnose_connectivity(self, connectivity_table, entry=0):
                return {
                    "ok": True,
                    "ngt_outlets": [
                        {"ok": True, "absorbing_state": 1, "committor": 0.20},
                        {"ok": True, "absorbing_state": 2, "committor": 0.70},
                        {"ok": True, "absorbing_state": 3, "committor": 0.10},
                    ],
                }

        monkeypatch.setattr(basin_module, "AmselFPTASelector", FakeSelector)
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(max_absorbing_refinements=2)
        )
        basin.connectivity_table = table

        rows = basin._absorbing_refinement_rows()

        assert rows == [1, 0]
        assert basin.absorbing_refinement_diagnostics == {
            "total": 3,
            "refined": 2,
            "skipped": 1,
            "unresolved_committor": pytest.approx(0.10),
            "unresolved_rate": pytest.approx(5.0),
        }

    def test_absorbing_refinement_uses_committor_tolerance(self, monkeypatch):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 10.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 2,
                    "event_connexion": 2,
                    "central_atom": 20,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 3,
                    "event_connexion": 3,
                    "central_atom": 30,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 5.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )

        class FakeSelector:
            def diagnose_connectivity(self, connectivity_table, entry=0):
                return {
                    "ok": True,
                    "ngt_outlets": [
                        {"ok": True, "absorbing_state": 1, "committor": 0.60},
                        {"ok": True, "absorbing_state": 2, "committor": 0.30},
                        {"ok": True, "absorbing_state": 3, "committor": 0.10},
                    ],
                }

        monkeypatch.setattr(basin_module, "AmselFPTASelector", FakeSelector)
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(
                max_absorbing_refinements=None,
                frontier_committor_tol=0.10,
            )
        )
        basin.connectivity_table = table

        rows = basin._absorbing_refinement_rows()

        assert rows == [0, 1]
        assert basin.absorbing_refinement_diagnostics == {
            "total": 3,
            "refined": 2,
            "skipped": 1,
            "unresolved_committor": pytest.approx(0.10),
            "unresolved_rate": pytest.approx(5.0),
        }

    def test_absorbing_refinement_uses_rate_fraction_fallback(self, monkeypatch):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 10.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 2,
                    "event_connexion": 2,
                    "central_atom": 20,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 3,
                    "event_connexion": 3,
                    "central_atom": 30,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 5.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )

        class EmptyOutletSelector:
            def diagnose_connectivity(self, connectivity_table, entry=0):
                return {"ok": True, "ngt_outlets": []}

        monkeypatch.setattr(basin_module, "AmselFPTASelector", EmptyOutletSelector)
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(
                max_absorbing_refinements=None,
                frontier_committor_tol=0.10,
            )
        )
        basin.connectivity_table = table

        rows = basin._absorbing_refinement_rows()

        assert rows == [0, 2]
        assert basin.absorbing_refinement_diagnostics == {
            "total": 3,
            "refined": 2,
            "skipped": 1,
            "unresolved_committor": pytest.approx(1.0 / 16.0),
            "unresolved_rate": pytest.approx(1.0),
        }

    def test_absorbing_refinement_diagnostics_refresh_after_rate_updates(
        self, monkeypatch
    ):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 2,
                    "event_connexion": 2,
                    "central_atom": 20,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )

        class FakeSelector:
            def diagnose_connectivity(self, connectivity_table, entry=0):
                df = connectivity_table.df
                total_rate = float(df["k_forward"].sum())
                return {
                    "ok": True,
                    "ngt_outlets": [
                        {
                            "ok": True,
                            "absorbing_state": int(row["state_connexion"]),
                            "committor": float(row["k_forward"]) / total_rate,
                        }
                        for _, row in df.iterrows()
                    ],
                }

        monkeypatch.setattr(basin_module, "AmselFPTASelector", FakeSelector)
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(max_absorbing_refinements=1)
        )
        basin.connectivity_table = table

        rows = basin._absorbing_refinement_rows()
        assert rows == [0]
        assert basin.absorbing_refinement_diagnostics[
            "unresolved_committor"
        ] == pytest.approx(0.5)

        table.df.loc[0, "k_forward"] = 99.0
        basin._refresh_absorbing_refinement_diagnostics()

        assert basin.absorbing_refinement_diagnostics == {
            "total": 2,
            "refined": 1,
            "skipped": 1,
            "unresolved_committor": pytest.approx(0.01),
            "unresolved_rate": pytest.approx(1.0),
        }

    def test_absorbing_refinement_continues_after_rate_update_committor_drift(
        self, monkeypatch
    ):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": state_connexion,
                    "event_connexion": state_connexion,
                    "central_atom": central_atom,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                }
                for state_connexion, central_atom in ((1, 10), (2, 20), (3, 30))
            ]
        )

        class FakeSelector:
            def diagnose_connectivity(self, connectivity_table, entry=0):
                df = connectivity_table.df
                total_rate = float(df["k_forward"].sum())
                return {
                    "ok": True,
                    "ngt_outlets": [
                        {
                            "ok": True,
                            "absorbing_state": int(row["state_connexion"]),
                            "committor": float(row["k_forward"]) / total_rate,
                        }
                        for _, row in df.iterrows()
                    ],
                }

        class FakeFuture:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        class FakeManager:
            def __init__(self):
                self.refined_centers = []

            def get_total_energy(self, **_kwargs):
                return FakeFuture(0.0)

            def partn_refine(self, _config, central_atom, *_args):
                self.refined_centers.append(int(central_atom))
                saddle_energy = 0.01 if int(central_atom) in {10, 20} else 1.0
                return FakeFuture(
                    Ok(
                        SimpleNamespace(
                            E_saddle=saddle_energy,
                            saddle_positions=np.array([[0.5, 0.0, 0.0]]),
                        )
                    )
                )

        class FakeNeighbors:
            def get_neighbors(self, *_args):
                return np.array([0])

        class FakeState:
            def __init__(self):
                self.system = System(
                    positions=np.array([[0.0, 0.0, 0.0]]),
                    types=np.array(["Cu"]),
                    cell=np.eye(3) * 10.0,
                    pbc=True,
                    index=np.array([0]),
                )
                self.neighbors_list = FakeNeighbors()

            def ensure_full_state(self, _config):
                return None

            def release_heavy_objects(self):
                return None

        class FakeRegistration:
            def __init__(self, *_args, **_kwargs):
                pass

            def match(self):
                return Ok(
                    SimpleNamespace(
                        rotation_matrix=np.eye(3),
                        translation_matrix=np.zeros(3),
                        permutation_matrix=np.array([0]),
                    )
                )

        monkeypatch.setattr(basin_module, "AmselFPTASelector", FakeSelector)
        monkeypatch.setattr(basin_module, "PointSetRegistration", FakeRegistration)
        monkeypatch.setattr(basin_module, "check_match", lambda result, _thr: result)
        monkeypatch.setattr(
            basin_module,
            "compute_rate_Eyring",
            lambda dE, _config: float(dE),
        )

        reference_table = SimpleNamespace(
            table=pd.DataFrame(
                [
                    {
                        "idx_ref": idx_ref,
                        "saddle_positions": np.array([[0.5, 0.0, 0.0]]),
                    }
                    for idx_ref in (1, 2, 3)
                ]
            )
        )
        manager = FakeManager()
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(
                max_absorbing_refinements=None,
                frontier_committor_tol=0.40,
            ),
            control=SimpleNamespace(active_volume=True),
            psr=SimpleNamespace(matching_score_thr=0.4),
        )
        basin.connectivity_table = table
        basin.reference_table = reference_table
        basin.manager = manager
        basin.states = {0: FakeState()}
        basin.absorbing_saddle_positions = {}
        basin.basin_search_registry = None

        result = basin.refine_absorbing(system=object())

        assert result.is_ok(), result.err_value() if not result.is_ok() else None
        assert manager.refined_centers == [10, 20, 30]
        assert basin.absorbing_refinement_diagnostics["unresolved_committor"] == 0.0

    def test_frontier_budget_records_unresolved_committor(self, monkeypatch):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 2,
                    "event_connexion": 2,
                    "central_atom": 20,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )
        monkeypatch.setattr(
            basin_module,
            "amsel_state_guidance_scores",
            lambda connectivity_table, entry=0: {1: 0.8, 2: 0.2},
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.connectivity_table = table

        basin._record_unresolved_frontier_diagnostics()

        assert basin.unresolved_frontier_diagnostics == {
            "total": 1,
            "unresolved_committor": pytest.approx(0.8),
            "unresolved_rate": pytest.approx(3.0),
        }

    def test_frontier_boundary_marks_unexpanded_states_absorbing(self, monkeypatch):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
                {
                    "state": 0,
                    "state_connexion": 2,
                    "event_connexion": 2,
                    "central_atom": 20,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                },
            ]
        )
        monkeypatch.setattr(
            basin_module,
            "amsel_state_guidance_scores",
            lambda connectivity_table, entry=0: {1: 0.75, 2: 0.25},
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.connectivity_table = table

        basin._absorb_unexpanded_frontier()

        assert not bool(basin.connectivity_table.df.loc[0, "transient"])
        assert basin.frontier_boundary_diagnostics == {
            "total": 1,
            "boundary_committor": pytest.approx(0.75),
            "boundary_rate": pytest.approx(3.0),
        }

    def test_execute_absorbs_frontier_boundary_before_refinement(
        self, monkeypatch
    ):
        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                }
            ]
        )

        def fake_initialize(self, system):
            self.states = {0: object()}
            self.connectivity_table = table

        def stop_refine(self, system):
            assert not bool(self.connectivity_table.df.loc[0, "transient"])
            return Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="stop"))

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            lambda self, max_expansions=None, max_closed_states=None: Ok(None),
        )
        monkeypatch.setattr(BasinsGenericEvents, "refine_absorbing", stop_refine)
        monkeypatch.setattr(
            basin_module,
            "amsel_state_guidance_scores",
            lambda connectivity_table, entry=0: {1: 1.0},
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(
                max_expansions=None,
                max_closed_states=None,
                frontier_committor_tol=0.0,
            )
        )
        basin.manager = SimpleNamespace(use_local=lambda: None)

        result = basin.execute(system=object())

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.EVENT_NOT_FOUND
        assert basin.unresolved_frontier_diagnostics == {
            "total": 1,
            "unresolved_committor": pytest.approx(1.0),
            "unresolved_rate": pytest.approx(3.0),
        }
        assert basin.frontier_boundary_diagnostics == {
            "total": 1,
            "boundary_committor": pytest.approx(1.0),
            "boundary_rate": pytest.approx(3.0),
        }

    def test_construct_marks_all_crystal_state_as_absorbing_terminal(
        self, monkeypatch
    ):
        class FakeEnvironment:
            atomic_environment_list = ["crystal", "crystal", "crystal"]

        class FakeState:
            def __init__(self, system=None):
                self.system = system
                self.environment = FakeEnvironment()
                self.transient = True

            def ensure_full_state(self, config):
                return None

            def release_heavy_objects(self):
                return None

        class FailingExplorer:
            connectivity_table = BasinStatesConnectivity()

            def explore(self, *args, **kwargs):
                raise AssertionError("terminal all-crystal states must not be explored")

            def clear(self):
                return None

        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                }
            ]
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace()
        basin.states = {0: FakeState(system=object())}
        basin.known_environments = {"crystal"}
        basin.states_to_explore = [1]
        basin.explored_states = [0]
        basin.current_state = 0
        basin.connectivity_table = table
        basin.explorer = FailingExplorer()
        basin.exploration_order = []
        basin.exploration_decisions = []
        basin.last_exploration_guidance = {}
        monkeypatch.setattr(
            basin,
            "system_from_state",
            lambda *args, **kwargs: Ok(object()),
        )
        monkeypatch.setattr(basin, "is_new_state", lambda system: -1)
        monkeypatch.setattr(
            basin,
            "_add_state",
            lambda state_index, system, transient=True: basin.states.update(
                {state_index: FakeState(system=system)}
            ),
        )

        result = basin.construct_connexion_table()

        assert result.is_ok()
        assert not bool(basin.connectivity_table.df.loc[0, "transient"])
        assert basin.states[1].transient is False
        assert basin.states_to_explore == []

    def test_execute_materializes_frontier_exit_with_global_manager(
        self, monkeypatch
    ):
        class FakeManager:
            def __init__(self):
                self.mode = "global"

            def use_local(self):
                self.mode = "local"

            def use_global(self):
                self.mode = "global"

        class FakeNeighbors:
            def get_neighbors(self, *_args):
                return np.array([0])

        class FakeState:
            def __init__(self):
                self.system = SimpleNamespace(positions=np.array([[0.0, 0.0, 0.0]]))
                self.neighbors_list = FakeNeighbors()

            def ensure_full_state(self, config):
                return None

        class FakeSelector:
            def select_from_connectivity(self, connectivity_table):
                return Ok(BasinSelectorOutput(t_exit=1.0, exit_state=1))

        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 1,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.0,
                    "k_forward": 3.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                }
            ]
        )
        manager = FakeManager()

        def fake_initialize(self, system):
            self.states = {0: FakeState()}
            self.connectivity_table = table
            self.selector = FakeSelector()
            self.absorbing_saddle_positions = {1: np.array([[0.0, 0.0, 0.0]])}

        def fake_refine(self, system):
            assert self.manager.mode == "local"
            return Ok(None)

        def fake_system_from_state(self, *args):
            assert self.manager.mode == "global"
            return Ok(SimpleNamespace(positions=np.array([[1.0, 0.0, 0.0]])))

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            lambda self, max_expansions=None, max_closed_states=None: Ok(None),
        )
        monkeypatch.setattr(BasinsGenericEvents, "refine_absorbing", fake_refine)
        monkeypatch.setattr(BasinsGenericEvents, "system_from_state", fake_system_from_state)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "_add_state",
            lambda self, state_index, system, transient=True: self.states.update(
                {state_index: SimpleNamespace(system=system, transient=transient)}
            ),
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(basin=SimpleNamespace())
        basin.manager = manager

        result = basin.execute(system=object())

        assert result.is_ok()
        assert result.ok_value().exit_state == 1

    def test_execute_falls_back_to_legacy_selector_when_amsel_selector_fails(
        self, monkeypatch
    ):
        class FakeManager:
            def use_local(self):
                return None

            def use_global(self):
                return None

        class FakeNeighbors:
            def get_neighbors(self, *_args):
                return np.array([0])

        class FakeState:
            def __init__(self):
                self.system = SimpleNamespace(positions=np.array([[0.0, 0.0, 0.0]]))
                self.neighbors_list = FakeNeighbors()

            def ensure_full_state(self, config):
                return None

        class FailingSelector:
            def select_from_connectivity(self, connectivity_table):
                return Err(
                    ErrorInfo(
                        type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                        message="amsel rejected graph",
                    )
                )

        class LegacySelector:
            def select_from_connectivity(self, connectivity_table):
                return Ok(BasinSelectorOutput(t_exit=2.0, exit_state=1))

        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": 1,
                    "event_connexion": 3,
                    "central_atom": 0,
                    "sym": 0,
                    "transient": True,
                    "dE_forward": 0.2,
                    "k_forward": 4.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                }
            ]
        )

        def fake_initialize(self, system):
            self.states = {0: FakeState()}
            self.connectivity_table = table
            self.selector = FailingSelector()
            self.selector_fallback = LegacySelector()
            self.absorbing_saddle_positions = {1: np.array([[0.0, 0.0, 0.0]])}

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            lambda self, max_expansions=None, max_closed_states=None: Ok(None),
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "refine_absorbing",
            lambda self, system: Ok(None),
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "system_from_state",
            lambda self, *args: Ok(
                SimpleNamespace(positions=np.array([[1.0, 0.0, 0.0]]))
            ),
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "_add_state",
            lambda self, state_index, system, transient=True: self.states.update(
                {state_index: SimpleNamespace(system=system, transient=transient)}
            ),
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(basin=SimpleNamespace())
        basin.manager = FakeManager()

        result = basin.execute(system=object())

        assert result.is_ok()
        assert result.ok_value().t_exit == 2.0
        assert result.ok_value().num_reference_event == 3

    def test_execute_supplies_catalog_saddle_for_unrefined_exit(
        self, monkeypatch
    ):
        class FakeManager:
            def __init__(self):
                self.mode = "global"

            def use_local(self):
                self.mode = "local"

            def use_global(self):
                self.mode = "global"

        class FakeNeighbors:
            def get_neighbors(self, *_args):
                return np.array([0])

        class FakeState:
            def __init__(self):
                self.system = SimpleNamespace(positions=np.array([[0.0, 0.0, 0.0]]))
                self.neighbors_list = FakeNeighbors()

            def ensure_full_state(self, config):
                return None

        class FakeSelector:
            def select_from_connectivity(self, connectivity_table):
                return Ok(BasinSelectorOutput(t_exit=1.0, exit_state=5))

        table = BasinStatesConnectivity()
        table.df = pd.DataFrame(
            [
                {
                    "state": 0,
                    "state_connexion": connexion,
                    "event_connexion": connexion,
                    "central_atom": 10,
                    "sym": 0,
                    "transient": False,
                    "dE_forward": 0.0,
                    "k_forward": 1.0,
                    "dE_backward": 0.0,
                    "k_backward": 0.0,
                }
                for connexion in (1, 2, 3, 4, 5)
            ]
        )

        catalog_saddle = np.array([[0.5, 0.0, 0.0]])
        ensure_calls = {"count": 0}

        def fake_initialize(self, system):
            self.states = {0: FakeState()}
            self.connectivity_table = table
            self.selector = FakeSelector()
            self.selector_fallback = None
            # Refinement budget covered only state 1 and state 2.
            self.absorbing_saddle_positions = {
                1: np.array([[0.0, 0.0, 0.0]]),
                2: np.array([[0.0, 0.0, 0.0]]),
            }

        def fake_ensure(self, exit_state, from_state, event_idx, central_atom, sym_idx):
            ensure_calls["count"] += 1
            ensure_calls["exit_state"] = int(exit_state)
            self.absorbing_saddle_positions[int(exit_state)] = catalog_saddle
            return Ok(None)

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            lambda self, max_expansions=None, max_closed_states=None: Ok(None),
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "refine_absorbing",
            lambda self, system: Ok(None),
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "_ensure_catalog_saddle_for_exit",
            fake_ensure,
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "system_from_state",
            lambda self, *args: Ok(
                SimpleNamespace(positions=np.array([[1.0, 0.0, 0.0]]))
            ),
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "_add_state",
            lambda self, state_index, system, transient=True: self.states.update(
                {state_index: SimpleNamespace(system=system, transient=transient)}
            ),
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(basin=SimpleNamespace())
        basin.manager = FakeManager()

        result = basin.execute(system=object())

        assert result.is_ok(), result.err_value() if not result.is_ok() else None
        assert result.ok_value().exit_state == 5
        assert ensure_calls == {"count": 1, "exit_state": 5}
        np.testing.assert_array_equal(
            result.ok_value().saddle_positions,
            catalog_saddle,
        )

    def test_execute_uses_configured_basin_exploration_budgets(self, monkeypatch):
        calls = {}

        def fake_initialize(self, system):
            self.states = {}
            self.connectivity_table = SimpleNamespace(
                reorder_states_index=lambda: {},
            )

        def fake_construct(self, max_expansions=None, max_closed_states=None):
            calls["max_expansions"] = max_expansions
            calls["max_closed_states"] = max_closed_states
            return Ok(None)

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            fake_construct,
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "refine_absorbing",
            lambda self, system: Err(
                ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="stop")
            ),
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(max_expansions=3, max_closed_states=2)
        )
        basin.manager = SimpleNamespace(use_local=lambda: None)

        result = basin.execute(system=object())

        assert not result.is_ok()
        assert calls == {"max_expansions": 3, "max_closed_states": 2}

    def test_execute_uses_finite_amsel_basin_exploration_budget(self, monkeypatch):
        calls = {}

        def fake_initialize(self, system):
            self.states = {}
            self.connectivity_table = SimpleNamespace(
                reorder_states_index=lambda: {},
            )

        def fake_construct(self, max_expansions=None, max_closed_states=None):
            calls["max_expansions"] = max_expansions
            calls["max_closed_states"] = max_closed_states
            return Ok(None)

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            fake_construct,
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "refine_absorbing",
            lambda self, system: Err(
                ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="stop")
            ),
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(
                max_expansions=None,
                max_closed_states=None,
                selector="amsel-adaptive",
            )
        )
        basin.manager = SimpleNamespace(use_local=lambda: None)

        result = basin.execute(system=object())

        assert not result.is_ok()
        assert calls == {"max_expansions": None, "max_closed_states": 8}

    def test_execute_keeps_legacy_basin_exploration_unbounded(self, monkeypatch):
        calls = {}

        def fake_initialize(self, system):
            self.states = {}
            self.connectivity_table = SimpleNamespace(
                reorder_states_index=lambda: {},
            )

        def fake_construct(self, max_expansions=None, max_closed_states=None):
            calls["max_expansions"] = max_expansions
            calls["max_closed_states"] = max_closed_states
            return Ok(None)

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            fake_construct,
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "refine_absorbing",
            lambda self, system: Err(
                ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="stop")
            ),
        )

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(
            basin=SimpleNamespace(
                max_expansions=None,
                max_closed_states=None,
                selector="legacy-fpta",
            )
        )
        basin.manager = SimpleNamespace(use_local=lambda: None)

        result = basin.execute(system=object())

        assert not result.is_ok()
        assert calls == {"max_expansions": None, "max_closed_states": None}

    def test_execute_drops_states_removed_by_connectivity_reorder(self, monkeypatch):
        kept_state = object()

        def fake_initialize(self, system):
            self.states = {0: object(), 13: kept_state}
            self.connectivity_table = SimpleNamespace(
                reorder_states_index=lambda: {13: 0},
            )

        def fake_refine(self, system):
            assert self.states == {0: kept_state}
            return Err(ErrorInfo(type=ErrorType.EVENT_NOT_FOUND, message="stop"))

        monkeypatch.setattr(BasinsGenericEvents, "_initialize", fake_initialize)
        monkeypatch.setattr(
            BasinsGenericEvents,
            "construct_connexion_table",
            lambda self, max_expansions=None, max_closed_states=None: Ok(None),
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "_record_unresolved_frontier_diagnostics",
            lambda self: None,
        )
        monkeypatch.setattr(
            BasinsGenericEvents,
            "_absorb_unexpanded_frontier",
            lambda self: None,
        )
        monkeypatch.setattr(BasinsGenericEvents, "refine_absorbing", fake_refine)

        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.config = SimpleNamespace(basin=SimpleNamespace())
        basin.manager = SimpleNamespace(use_local=lambda: None)

        result = basin.execute(system=object())

        assert not result.is_ok()
        assert result.err_value().type == ErrorType.EVENT_NOT_FOUND

    def test_state_equivalence_reuses_cached_neighbor_tree(self, monkeypatch):
        real_tree = basin_module.cKDTree
        builds = 0

        def counting_tree(*args, **kwargs):
            nonlocal builds
            builds += 1
            return real_tree(*args, **kwargs)

        monkeypatch.setattr(basin_module, "cKDTree", counting_tree)

        cell = np.diag([10.0, 10.0, 10.0])
        stored = System(
            positions=np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            ),
            types=["X", "X", "X"],
            cell=cell,
            pbc=np.array([True, True, True]),
            index=np.arange(3),
        )
        candidate = System(
            positions=np.array(
                [[0.0, 0.0, 0.0], [6.0, 6.0, 6.0], [0.0, 2.0, 0.0]]
            ),
            types=["X", "X", "X"],
            cell=cell,
            pbc=np.array([True, True, True]),
            index=np.arange(3),
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.states = {0: StateData(system=stored, environment=None, neighbors_list=None)}

        assert basin.is_new_state(candidate) == -1
        assert basin.is_new_state(candidate) == -1
        assert builds == 1

    def test_state_equivalence_rebuilds_tree_after_position_mutation(self, monkeypatch):
        real_tree = basin_module.cKDTree
        builds = 0

        def counting_tree(*args, **kwargs):
            nonlocal builds
            builds += 1
            return real_tree(*args, **kwargs)

        monkeypatch.setattr(basin_module, "cKDTree", counting_tree)

        cell = np.diag([10.0, 10.0, 10.0])
        stored = System(
            positions=np.array(
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
            ),
            types=["X", "X", "X"],
            cell=cell,
            pbc=np.array([True, True, True]),
            index=np.arange(3),
        )
        candidate = System(
            positions=np.array(
                [[0.0, 0.0, 0.0], [6.0, 6.0, 6.0], [0.0, 2.0, 0.0]]
            ),
            types=["X", "X", "X"],
            cell=cell,
            pbc=np.array([True, True, True]),
            index=np.arange(3),
        )
        basin = BasinsGenericEvents.__new__(BasinsGenericEvents)
        basin.states = {0: StateData(system=stored, environment=None, neighbors_list=None)}

        assert basin.is_new_state(candidate) == -1
        stored.positions[1] = [3.0, 0.0, 0.0]
        assert basin.is_new_state(candidate) == -1
        assert builds == 2

    def test_refinement_error_context_records_failed_connectivity_row(self):
        error = ErrorInfo(
            type=ErrorType.EVENT_NOT_FOUND,
            message="no event found",
            details=(0, ""),
        )
        row = pd.Series(
            {
                "state": 0,
                "state_connexion": 13,
                "event_connexion": 1,
                "central_atom": 3330,
                "sym": 0,
                "transient": False,
            }
        )

        contextual = basin_module._refinement_error_with_row_context(
            error,
            row_index=7,
            row=row,
        )

        assert contextual.type is ErrorType.EVENT_NOT_FOUND
        assert contextual.message == "no event found"
        assert contextual.details == (0, "")
        assert contextual.variables == {
            "refinement_row": {
                "row_index": 7,
                "state": 0,
                "state_connexion": 13,
                "event_connexion": 1,
                "central_atom": 3330,
                "sym": 0,
                "transient": False,
            }
        }
