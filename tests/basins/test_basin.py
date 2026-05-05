import numpy as np

from pykmc import System
from pykmc.basins import BasinsGenericEvents, StateData
import logging
from pykmc.enginemanager.lmpi.pool import ManagerFactory
import pykmc.basins.basin as basin_module

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
