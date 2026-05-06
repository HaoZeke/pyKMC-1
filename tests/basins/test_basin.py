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
from pykmc.result import Err, ErrorInfo, ErrorType, Ok

logger = logging.getLogger("tests")

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
