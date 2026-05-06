from .detection import Detector
from .exploration import Explorer, BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FPTASelector
from .amsel_selection import AmselFPTASelector, _AMSEL_AVAILABLE
from .amsel_guidance import amsel_rank_basin_frontier, amsel_state_guidance_scores
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pykmc import System, Config, NeighborsList, AtomicEnvironment, ReferenceEventTable, PointSetRegistration, check_match, Reconstruction
from typing import Optional
from ..utils import geometry
from ..rate_constant import compute_rate_Eyring
from pykmc.result import Err, ErrorInfo, ErrorType, Ok, BasinOutput
import hashlib
import pandas as pd
import copy
import numpy as np
from scipy.spatial import cKDTree


REFINEMENT_CONTEXT_COLUMNS = (
    "state",
    "state_connexion",
    "event_connexion",
    "central_atom",
    "sym",
    "transient",
)


def _refinement_error_with_row_context(
    error: ErrorInfo,
    *,
    row_index: int,
    row: pd.Series,
) -> ErrorInfo:
    variables = dict(error.variables or {})
    variables["refinement_row"] = _refinement_row_payload(
        row_index=row_index,
        row=row,
    )
    return ErrorInfo(
        type=error.type,
        message=error.message,
        details=error.details,
        variables=variables,
    )


def _refinement_row_payload(*, row_index: int, row: pd.Series) -> dict[str, object]:
    payload: dict[str, object] = {"row_index": int(row_index)}
    for column in REFINEMENT_CONTEXT_COLUMNS:
        if column in row.index:
            payload[column] = _json_scalar(row[column])
    return payload


def _json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)

#TODO: StateDate is here to handle state informations, when State Object will be creates, need to remove
#TODO: For the moment Basin uses EnergyThresholdDetector, BasinGenericEventExplorer, FPTASelector, need to deal with possible multiple implementation with builder.
#TODO: Think about parallized exploration 
#TODO: Could think of refining transient -> absorbing event when exploring
#TODO : Exit if state 0 leads to all absorbing states because all unknown environments, here FTPA fails but because only have 1 transient state (0), should be a different ERROR.TYPE
#TODO should also check if we apply same event to different central atoms but same saddle position meaning that it s a duplicate event, so remove.

@dataclass
class StateData:
    system: Optional[System]
    environment: Optional[AtomicEnvironment]
    neighbors_list: Optional[NeighborsList] 
    transient: bool = False
    visited: bool = False
    _equivalence_tree: object | None = field(default=None, init=False, repr=False)
    _equivalence_tree_signature: tuple[tuple[float, float, float], tuple[int, ...], bytes] | None = field(
        default=None, init=False, repr=False
    )

    def release_heavy_objects(self) -> None : 
        """Release heavy objects"""
        self.neighbors_list = None 
        self.environment = None
    
    def ensure_full_state(self, config: Config) -> None : 
        if self.system is not None : 
            if self.neighbors_list is None : 
                self.neighbors_list = NeighborsList(self.system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)  
            if self.environment is None : 
                self.environment = AtomicEnvironment(config.atomicenvironment.style, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'], config.atomicenvironment.neighbors_add)

    def _equivalence_signature(self, cell):
        box = tuple(np.diag(cell).astype(float))
        positions = np.ascontiguousarray(self.system.positions, dtype=np.float64)
        position_bytes = positions.view(np.uint8).reshape(-1)
        digest = hashlib.blake2b(position_bytes, digest_size=16).digest()
        return box, positions.shape, digest

    def equivalence_tree(self, cell):
        signature = self._equivalence_signature(cell)
        if self._equivalence_tree is None or self._equivalence_tree_signature != signature:
            box = signature[0]
            self._equivalence_tree = cKDTree(self.system.positions, boxsize=list(box))
            self._equivalence_tree_signature = signature
        return self._equivalence_tree


class BasinsGenericEvents() : 

    def __init__(self, config: Config, reference_table,known_environments, manager ) -> None :  
        self.config = config #Config object with basins parameters
        self.explorer = None #object to explore a state in the basin 
        self.reference_table = reference_table #Object with reference generic events
        self.manager = manager #object to do external task (minimize, refine)

        self.connectivity_table = None #Dataframe of basin connexion state
        self.selected_event = None #The selected event after basin exploration
        self.current_state = None #Current state where we're at 
        self.states_to_explore = None #List of state to explore 
        self.explored_states = None #List of state that we already explored
        self.states: dict[int, StateData] = {}  #Dictionnary of StateDate
        self.known_environments = known_environments 
        self.absorbing_saddle_positions: dict[int, np.ndarray] = {}
        self.absorbing_refinement_diagnostics: dict[str, object] = {}
        self.unresolved_frontier_diagnostics: dict[str, object] = {}
        self.frontier_boundary_diagnostics: dict[str, object] = {}
        self.last_exploration_guidance: dict[int, float] = {}
        self.exploration_order: list[int] = []
        self.exploration_decisions: list[dict[str, object]] = []

    def detection(self, params) -> bool : 
        """Utility method."""
        return self.detector.detection(**params) 
    
    def execute(self, system) : 
        """ 
        run the basin exploration and select an event from a system, corresponding to the first state in the basin, it is assumed that this state is transient.
        """
        #initialize the basin
        self._initialize(system)
        #explore the basin
        basin_config = getattr(self.config, "basin", None)
        result = self.construct_connexion_table(
            max_expansions=getattr(basin_config, "max_expansions", None),
            max_closed_states=getattr(basin_config, "max_closed_states", None),
        )
        if not result.is_ok() : 
            return result
        #reorder states index 
        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.states.items()}
        self._record_unresolved_frontier_diagnostics()
        self._absorb_unexpanded_frontier()
        #Refine absorbing states
        self.manager.use_local()
        result =self.refine_absorbing(system)
        if not result.is_ok() : 
            return result
        #apply selector algorithm to find t_exit and exit_state
        result = self.selector.select_from_connectivity(self.connectivity_table)
        if not result.is_ok() : 
            return result
        #Construct output KMC needs 
        t_exit = result.ok_value().t_exit
        exit_state = result.ok_value().exit_state
        if exit_state not in self.absorbing_saddle_positions:
            return Err(
                ErrorInfo(
                    type=ErrorType.BASIN_TEXIT_NOT_FOUND,
                    message="selected basin exit was not refined",
                    variables={
                        "exit_state": int(exit_state),
                        "absorbing_refinement": self.absorbing_refinement_diagnostics,
                    },
                )
            )

        from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=exit_state)
        if exit_state not in self.states:
            self.manager.use_global()
            result_state = self.system_from_state(
                from_state,
                event_idx,
                central_atom,
                sym_idx,
            )
            if not result_state.is_ok():
                return result_state
            self._add_state(
                state_index=exit_state,
                system=result_state.ok_value(),
                transient=False,
            )
        #Ensure from_state is state are full 
        self.states[from_state].ensure_full_state(self.config)

        neighbors = self.states[from_state].neighbors_list.get_neighbors("rcut", central_atom)
        return Ok(BasinOutput(initial_system_positions=self.states[from_state].system.positions, 
                              central_atom=central_atom, 
                              saddle_positions=self.absorbing_saddle_positions[exit_state], 
                              final_positions=self.states[exit_state].system.positions[neighbors], 
                              neighbors=neighbors,
                              energy_barrier= self.connectivity_table.df[(self.connectivity_table.df["state"] == from_state) & (self.connectivity_table.df["state_connexion"] == exit_state)].iloc[0]["dE_forward"], 
                              k_tot = self.connectivity_table.df.loc[self.connectivity_table.df["transient"] == False, "k_forward"].sum(),
                              t_exit = t_exit,
                              exit_state = exit_state, 
                              from_state = from_state,
                              num_reference_event= event_idx))
        

    def _initialize(self, system) -> None: 
        """ 
        Initialize necessary component after entering in basin. We always enter in state == 0.
        """
        self.current_state = 0
        self.states_to_explore = [0] 
        self.explored_states = [] 
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(config=self.config, reference_table=self.reference_table)
        self.selector = self._make_selector()
        self.exploration_order = []
        self.exploration_decisions = []
        self.absorbing_refinement_diagnostics = {}
        self.unresolved_frontier_diagnostics = {}
        self.frontier_boundary_diagnostics = {}
        new_system = System(positions=system.positions.copy(), types=system.types.copy(), cell=system.cell.copy(), pbc=system.pbc.copy(), index=np.arange(len(system.types)))
        self._add_state(state_index=0, system=new_system)  #add current state 0 to self.states

    def _make_selector(self):
        selector = getattr(self.config.basin, "selector", "auto")
        if selector == "auto":
            if _AMSEL_AVAILABLE:
                return AmselFPTASelector(clock_mode="sampled")
            return FPTASelector()
        if selector == "legacy-fpta":
            return FPTASelector()
        clock_mode = selector.removeprefix("amsel-")
        return AmselFPTASelector(clock_mode=clock_mode)


    def construct_connexion_table(
        self,
        max_expansions: int | None = None,
        max_closed_states: int | None = None,
    ) : 
        """ 
        explore the basin and construct the connextion table
        """
        if not hasattr(self, "exploration_order"):
            self.exploration_order = []
        if not hasattr(self, "exploration_decisions"):
            self.exploration_decisions = []
        expanded_states = 0
        closed_states = 0
        #Loop over state to explore 
        while len(self.states_to_explore) != 0 :
            if max_expansions is not None and expanded_states >= max_expansions:
                break
            if max_closed_states is not None and closed_states >= max_closed_states:
                break
            #next state to explore : 
            to_explore = self.states_to_explore[0]

            if to_explore not in self.states : #always true except at the start (to_explore = 0) 
                #We need to create the state 
                    #find a state and an event from which we go to the state that we want to create
                from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=to_explore)

                    #Create new system by applying (reconstruction) the generic event to the from_state
                result = self.system_from_state(from_state, event_idx, central_atom, sym_idx) 
                if not result.is_ok() : 
                    return result
                new_system = result.ok_value()

                    #Check if it is a new_system or already in states 
                is_new_state = self.is_new_state(new_system) 
                if is_new_state != -1 : #It already exists 
                    #update table
                    self.connectivity_table.change_state_index(current_index=to_explore, new_index=is_new_state)
                    self.explored_states.append(to_explore)
                    closed_states += 1
                    self.states_to_explore.remove(to_explore)

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    continue #Skip the rest

                #add state
                self._add_state(state_index=to_explore, system=new_system, transient=is_transient)

                #ENSURE FULL STATE TO EXPLORE 
                self.states[to_explore].ensure_full_state(self.config)
                # The basin graph uses the same terminal boundary as the KMC loop.
                if self.is_state_terminal_environment(self.states[to_explore]):
                    self.connectivity_table.change_state_to_absorbing(to_explore)
                    self.states[to_explore].transient = False
                    is_transient = False
                #Check if unknown atomic environments
                elif self.is_states_has_unknown_environments(self.states[to_explore]) : 
                    #We consider that this state is an absorbing one because we need to search new events (in main KMC loop) 
                    #Need to update the connectivity table 
                    self.connectivity_table.change_state_to_absorbing(to_explore) 
                    self.states[to_explore].transient = False
                    is_transient = False
                
                if not is_transient : 
                    self.states_to_explore.remove(to_explore)
                    self.explored_states.append(to_explore)
                    closed_states += 1

                    #Cleaning
                    self.states[from_state].release_heavy_objects()
                    self.states[to_explore].release_heavy_objects()


                    continue #We dont explore/skip the rest

                #Release heavy objet memory
                self.states[from_state].release_heavy_objects()

            


            #Explore state 
            self.current_state = to_explore
            last_state_connectivity = self.get_last_state_index()
            self.exploration_order.append(int(to_explore))
            self._record_exploration_decision(int(to_explore))

            #Ensure full state to explore 
            self.states[to_explore].ensure_full_state(self.config)
            self.explorer.explore(state=self.states[to_explore], state_index=self.current_state, start_index=last_state_connectivity)
            expanded_states += 1
            
            #to_explore has been explored : 
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)
            closed_states += 1

            #Merge state connectivity table to basin connectivity table 
            self.connectivity_table.merge(self.explorer.connectivity_table)
            #Clrean explorer connectivity table
            self.explorer.clear()
            self.update_to_explore()
            #Clean heaby state object : 
            self.states[to_explore].release_heavy_objects()
            
            if to_explore == 1 : #TESTST 
                self.connectivity_table.save("test.pickle")

        return Ok(None)

    def select_event(self) : 
        """ 
        select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self) : 
        """ 
        convinient method
        """
        pass

    def get_last_state_index(self) : 
        if self.current_state == 0 : #connextion table is empty
            new_state_connexion = 1 
        else : #last state connexion +1
            new_state_connexion = int(self.connectivity_table.get_table()['state_connexion'].iloc[-1]+1)
        return new_state_connexion
    
    def update_to_explore(self) : 
        #Find all state index in the connexion table : 
        unique_states = set(self.connectivity_table.get_table()['state']).union(set(self.connectivity_table.get_table()['state_connexion']))
        remaining_states = unique_states.difference(set(self.explored_states))
        self.states_to_explore = self._order_states_to_explore(remaining_states)

    def _order_states_to_explore(self, candidate_states) -> list[int]:
        priority = getattr(getattr(self.config, "basin", None), "exploration_priority", "auto")
        self.last_exploration_guidance = {}
        if priority in {"legacy", "fifo"}:
            return list(candidate_states)
        if priority not in {"auto", "amsel", "amsel-diverse"}:
            return list(candidate_states)
        if priority == "auto" and not _AMSEL_AVAILABLE:
            return list(candidate_states)

        scores = amsel_state_guidance_scores(self.connectivity_table, entry=0)
        self.last_exploration_guidance = scores
        if not scores:
            return list(candidate_states)

        if priority == "amsel-diverse":
            seen_processes = self._explored_process_signatures()
            ranked = amsel_rank_basin_frontier(
                candidate_states=candidate_states,
                guidance=scores,
                event_families={
                    int(state): self._incoming_diversity_signature(int(state))
                    for state in candidate_states
                },
                closed_event_families=seen_processes,
                duplicate_family_penalty=float(
                    getattr(
                        getattr(self.config, "basin", None),
                        "exploration_duplicate_family_penalty",
                        1.0,
                    )
                ),
                min_guidance=float(
                    getattr(
                        getattr(self.config, "basin", None),
                        "exploration_min_guidance",
                        0.0,
                    )
                ),
            )
            if ranked is not None:
                return ranked
            return sorted(
                candidate_states,
                key=lambda state: (
                    -float(scores.get(int(state), 0.0)),
                    self._incoming_diversity_signature(int(state)) in seen_processes,
                    self._incoming_diversity_signature_sort_key(int(state)),
                    int(state),
                ),
            )

        return sorted(
            candidate_states,
            key=lambda state: (-float(scores.get(int(state), 0.0)), int(state)),
        )

    def _explored_event_families(self) -> set[int]:
        return {
            event
            for state in self.explored_states
            for event in [self._incoming_event_family(int(state))]
            if event is not None
        }

    def _incoming_event_family(self, state: int) -> int | None:
        df = getattr(self.connectivity_table, "df", None)
        if not isinstance(df, pd.DataFrame) or "event_connexion" not in df.columns:
            return None
        rows = df.loc[df["state_connexion"] == int(state)]
        if rows.empty:
            return None
        return int(rows.iloc[0]["event_connexion"])

    def _incoming_event_family_sort_key(self, state: int) -> int:
        event = self._incoming_event_family(state)
        if event is None:
            return -1
        return int(event)

    def _explored_process_signatures(self) -> set[object]:
        return {
            process
            for state in self.explored_states
            for process in [self._incoming_diversity_signature(int(state))]
            if process is not None
        }

    def _incoming_process_signature(self, state: int) -> str | None:
        df = getattr(self.connectivity_table, "df", None)
        required = {"event_connexion", "central_atom"}
        if not isinstance(df, pd.DataFrame) or not required.issubset(df.columns):
            return None
        rows = df.loc[df["state_connexion"] == int(state)]
        if rows.empty:
            return None
        row = rows.iloc[0]
        if pd.isna(row["event_connexion"]) or pd.isna(row["central_atom"]):
            return None
        return f"{int(row['event_connexion'])}:{int(row['central_atom'])}"

    def _incoming_diversity_signature(self, state: int) -> object | None:
        process = self._incoming_process_signature(state)
        if process is not None:
            return process
        return self._incoming_event_family(state)

    def _incoming_diversity_signature_sort_key(self, state: int) -> str:
        process = self._incoming_diversity_signature(state)
        if process is None:
            return ""
        return str(process)

    def _record_exploration_decision(self, state: int) -> None:
        if not hasattr(self, "exploration_decisions"):
            self.exploration_decisions = []
        state = int(state)
        self.exploration_decisions.append(
            {
                "state": state,
                "event_family": self._incoming_event_family(state),
                "process_signature": self._incoming_process_signature(state),
                "guidance": float(
                    (getattr(self, "last_exploration_guidance", {}) or {}).get(
                        state,
                        0.0,
                    )
                ),
            }
        )


    def system_from_state(self, from_state, event_idx, central_atom, sym_idx) : 
        """ Reconstruct the generic event to generate new state from state
        """

        ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == event_idx] #event where event_idx == idx_ref
        if ref_event.empty:
            raise ValueError(f"idx_ref={event_idx} not found in reference table")
        ref_event = ref_event.iloc[0].copy()
#        ref_event = self.reference_table.table.iloc[event_idx].copy()

        #supposed_initial_positions = ref_event["initial_positions"].copy()
        #supposed_final_positions = ref_event["final_positions"].copy()
        #saddle_positions = ref_event['saddle_positions'].copy()

        supposed_initial_positions = np.array(ref_event["initial_positions"], copy=True)
        supposed_final_positions = np.array(ref_event["final_positions"], copy=True)
        saddle_positions = np.array(ref_event['saddle_positions'], copy=True)

        #Apply the generic event to the current state 

        #ENSURE FULL STATE FOR FROM STATE 
        self.states[from_state].ensure_full_state(self.config)

            #We start from the from_state
        new_system = System(positions=self.states[from_state].system.positions.copy(), types=self.states[from_state].system.types, cell=self.states[from_state].system.cell, pbc=True, index=np.arange(len(self.states[from_state].system.types)))
        #new_system = copy.deepcopy(self.states[from_state].system)

            #Apply PSR between event initial position and environment positions of the central_atoms
        result = PointSetRegistration(self.config, new_system, ref_event , self.states[from_state].neighbors_list, central_atom).match()
        if not result.is_ok(): #PSR Err
            return result
            # Check if PointSetRegistration match is valid 
        result = check_match(result, self.config.psr.matching_score_thr)
        if not result.is_ok() : #PSR matching score not valid : 
            return result
        else : 
            psr_output = result.ok_value() #get psr results
            
        # Apply PSR to generic event to move 
            
        # Apply symmetry matrix if sym != 0
        if sym_idx != 0 :
            sym_matrices = ref_event['sym_matrix']
            sym_matrix = sym_matrices[sym_idx]
            supposed_initial_positions = geometry.transform_positions(supposed_initial_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
            saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
            supposed_final_positions = geometry.transform_positions(supposed_final_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
        supposed_initial_positions = geometry.transform_positions(supposed_initial_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
        saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
        supposed_final_positions= geometry.transform_positions(supposed_final_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)

        # Move system do saddle positions
        neighbors = self.states[from_state].neighbors_list.get_neighbors('rcut', central_atom)
        new_system.update_positions(saddle_positions, atom_idx = neighbors)

        #Reconstruct the event
        #future = self.manager.minimize_with_results(self.config, positions=new_system.positions)
        #min_pos, _ = future.result()

        result = Reconstruction(self.config, self.manager).reconstruct(
            supposed_initial_positions,
            supposed_final_positions,
            new_system.positions,
            new_system.cell,
            self.config.psr.matching_score_thr,
            neighbors,
            minimize_command=self.config.basin.reconstruction_minimize,
        )
        if not result.is_ok() :
            return result
        new_system.update_positions(result.ok_value().min2_positions)

        return Ok(new_system)

    def refine_absorbing(self, system) :
        """When connectivity table is build, and that we have dict of states, we refine the energy barrier and k_forward of the transient -> absorbing event"""
        #compute the energy of the state 
        #for all row in connectivity table where we need to refine
        futures_context = {} #idx → { "min": f_min, "saddle": f_sad }
        for idx, row in self.connectivity_table.df.loc[
            self._absorbing_refinement_rows()
        ].iterrows() :
            if row['transient']  == False : #need to refine
                #tmp_system = copy.deepcopy(self.states[row["state"]].system)
                tmp_system = System(positions=self.states[row["state"]].system.positions.copy(), types=self.states[row["state"]].system.types, cell=self.states[row["state"]].system.cell, pbc=True, index=np.arange(len(self.states[row["state"]].system.types)))
                #get tmp_system energy 
                future1 = self.manager.get_total_energy(positions=tmp_system.positions.copy()) #Send copy not reference
                #move to generic saddle positions 
                ref_event = self.reference_table.table[self.reference_table.table["idx_ref"] == row["event_connexion"]] 
                if ref_event.empty:
                    raise ValueError(f"idx_ref={row['event_connexion']} not found in reference table")
                ref_event = ref_event.iloc[0].copy()
                #ref_event = self.reference_table.table.iloc[row["event_connexion"]].copy()
                saddle_positions = ref_event['saddle_positions'].copy()
                #Apply PSR between event initial position and environment positions of the central_atoms


                #ENSURE "STATE" FULL 
                self.states[row["state"]].ensure_full_state(self.config)

                result = PointSetRegistration(self.config, tmp_system, ref_event , self.states[row["state"]].neighbors_list, row["central_atom"]).match()
                if not result.is_ok(): #PSR Err
                    return result
                    # Check if PointSetRegistration match is valid 
                result = check_match(result, self.config.psr.matching_score_thr)
                if not result.is_ok() : #PSR matching score not valid : 
                    return result
                else : 
                    psr_output = result.ok_value() #get psr results

                # Apply symmetry matrix if sym != 0
                if row["sym"] != 0 :
                    sym_matrices = ref_event['sym_matrix']
                    sym_matrix = sym_matrices[row["sym"]]
                    saddle_positions = geometry.transform_positions(saddle_positions, sym_matrix,0, ref_event["sym_perm"][row["sym"]])
                saddle_positions = geometry.transform_positions(saddle_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)
                neighbors = self.states[row["state"]].neighbors_list.get_neighbors('rcut', row["central_atom"])

                if self.config.control.active_volume==True:
                    # add a job to manager queue
                    future2 = self.manager.partn_refine(self.config, row["central_atom"],
                                                  tmp_system.positions.copy(),
                                                  tmp_system.cell,
                                                  tmp_system.types.copy(),
                                                  neighbors.copy(),
                                                  saddle_positions.copy())
                # Move system do saddle positions
                else:
                    tmp_system.update_positions(saddle_positions, atom_idx = neighbors)
                    #refine
                    future2 = self.manager.partn_refine(self.config, row["central_atom"], tmp_system.positions.copy()) #send copy not reference !
                
                #save future in context : 
                futures_context[idx] = {
            "min": future1,
            "saddle": future2, 
            "neighbors": neighbors}
                
                #RELEASE MEMORY : 
                self.states[row["state"]].release_heavy_objects()

        #modify connectivity table entry future1 hold min energy, future2 holds E_saddle
        for idx, ctx in futures_context.items():
            E_min    = ctx["min"].result()
            result_sad = ctx["saddle"].result()
            if not result_sad.is_ok() : 
                row = self.connectivity_table.df.loc[idx]
                return Err(
                    _refinement_error_with_row_context(
                        result_sad.err_value(),
                        row_index=idx,
                        row=row,
                    )
                )
            E_sad = result_sad.ok_value().E_saddle
            if self.config.control.active_volume==True:
                dE = E_sad
            else:
                dE = E_sad - E_min
            k = compute_rate_Eyring(dE, self.config)

            #also save saddle positions refined 
            idx_state = self.connectivity_table.df.loc[idx].at['state_connexion']
            central_atom = self.connectivity_table.df.loc[idx].at['central_atom']
            #self.absorbing_saddle_positions[idx_state] = result.ok_value().saddle_positions[self.states[idx_state].neighbors_list.get_neighbors("rcut", central_atom)]
            self.absorbing_saddle_positions[idx_state] = result_sad.ok_value().saddle_positions[ctx["neighbors"]]
            # update connectivity table row
            self.connectivity_table.df.loc[idx, "dE_forward"] = dE
            self.connectivity_table.df.loc[idx, "k_forward"] = k
        self._refresh_absorbing_refinement_diagnostics()
        return Ok(None)

    def _absorbing_refinement_rows(self) -> list[int]:
        df = self.connectivity_table.df
        absorbing_rows = [
            int(idx) for idx, row in df.iterrows() if not bool(row["transient"])
        ]
        max_refinements = getattr(
            getattr(self.config, "basin", None),
            "max_absorbing_refinements",
            None,
        )
        scores = self._absorbing_refinement_scores()
        ordered_rows = sorted(
            absorbing_rows,
            key=lambda idx: (
                -float(scores.get(int(df.loc[idx, "state_connexion"]), 0.0)),
                -float(df.loc[idx, "k_forward"]),
                int(idx),
            ),
        )
        if max_refinements is None:
            selected_rows = self._rows_until_absorbing_committor_tolerance(
                ordered_rows,
                scores,
            )
        else:
            selected_rows = ordered_rows[: max(0, int(max_refinements))]

        selected_set = set(selected_rows)
        skipped_rows = [idx for idx in ordered_rows if idx not in selected_set]
        self._absorbing_refinement_selected_rows = selected_rows
        self._absorbing_refinement_skipped_rows = skipped_rows
        self._absorbing_refinement_total_rows = len(absorbing_rows)
        self._refresh_absorbing_refinement_diagnostics()
        return selected_rows

    def _rows_until_absorbing_committor_tolerance(
        self,
        ordered_rows: list[int],
        scores: dict[int, float],
    ) -> list[int]:
        if not ordered_rows:
            return []
        if not scores:
            return ordered_rows

        df = self.connectivity_table.df
        score_by_row = {
            idx: float(scores.get(int(df.loc[idx, "state_connexion"]), 0.0))
            for idx in ordered_rows
        }
        unresolved = float(sum(score_by_row.values()))
        if unresolved <= 0.0:
            return ordered_rows

        committor_tol = float(
            getattr(
                getattr(self.config, "basin", None),
                "frontier_committor_tol",
                0.0,
            )
            or 0.0
        )
        if committor_tol <= 0.0:
            return ordered_rows

        selected_rows: list[int] = []
        slack = max(1.0e-15, np.finfo(float).eps * max(unresolved, 1.0) * 16.0)
        for idx in ordered_rows:
            if unresolved <= committor_tol + slack:
                break
            selected_rows.append(idx)
            unresolved -= score_by_row[idx]
        return selected_rows

    def _refresh_absorbing_refinement_diagnostics(self) -> None:
        selected_rows = list(
            getattr(self, "_absorbing_refinement_selected_rows", [])
        )
        skipped_rows = list(getattr(self, "_absorbing_refinement_skipped_rows", []))
        total = int(
            getattr(
                self,
                "_absorbing_refinement_total_rows",
                len(selected_rows) + len(skipped_rows),
            )
        )
        self._record_absorbing_refinement_diagnostics(
            total=total,
            selected_rows=selected_rows,
            skipped_rows=skipped_rows,
        )

    def _record_absorbing_refinement_diagnostics(
        self,
        *,
        total: int,
        selected_rows: list[int],
        skipped_rows: list[int],
    ) -> None:
        df = self.connectivity_table.df
        scores = self._absorbing_refinement_scores()
        self.absorbing_refinement_diagnostics = {
            "total": int(total),
            "refined": len(selected_rows),
            "skipped": len(skipped_rows),
            "unresolved_committor": float(
                sum(
                    float(scores.get(int(df.loc[idx, "state_connexion"]), 0.0))
                    for idx in skipped_rows
                )
            ),
            "unresolved_rate": float(
                sum(float(df.loc[idx, "k_forward"]) for idx in skipped_rows)
            ),
        }

    def _absorbing_refinement_scores(self) -> dict[int, float]:
        try:
            report = AmselFPTASelector().diagnose_connectivity(
                self.connectivity_table,
                entry=0,
            )
        except Exception:  # noqa: BLE001 - refinement ordering falls back to rates.
            return {}
        scores: dict[int, float] = {}
        for outlet in report.get("ngt_outlets", []):
            if not isinstance(outlet, dict) or not outlet.get("ok", False):
                continue
            scores[int(outlet["absorbing_state"])] = float(outlet["committor"])
        return scores

    def _record_unresolved_frontier_diagnostics(self) -> None:
        df = getattr(self.connectivity_table, "df", None)
        if not isinstance(df, pd.DataFrame):
            self.unresolved_frontier_diagnostics = {
                "total": 0,
                "unresolved_committor": 0.0,
                "unresolved_rate": 0.0,
            }
            return
        transient_sources = {int(value) for value in df["state"].to_numpy()}
        frontier_states = sorted(
            {
                int(row["state_connexion"])
                for _, row in df.iterrows()
                if bool(row["transient"])
                and int(row["state_connexion"]) not in transient_sources
            }
        )
        scores = amsel_state_guidance_scores(self.connectivity_table, entry=0)
        self.unresolved_frontier_diagnostics = {
            "total": len(frontier_states),
            "unresolved_committor": float(
                sum(float(scores.get(state, 0.0)) for state in frontier_states)
            ),
            "unresolved_rate": float(
                sum(
                    float(row["k_forward"])
                    for _, row in df.iterrows()
                    if bool(row["transient"])
                    and int(row["state_connexion"]) in frontier_states
                )
            ),
        }

    def _absorb_unexpanded_frontier(self) -> None:
        df = getattr(self.connectivity_table, "df", None)
        if not isinstance(df, pd.DataFrame):
            self.frontier_boundary_diagnostics = {
                "total": 0,
                "boundary_committor": 0.0,
                "boundary_rate": 0.0,
            }
            return
        transient_sources = {int(value) for value in df["state"].to_numpy()}
        frontier_states = sorted(
            {
                int(row["state_connexion"])
                for _, row in df.iterrows()
                if bool(row["transient"])
                and int(row["state_connexion"]) not in transient_sources
            }
        )
        scores = amsel_state_guidance_scores(self.connectivity_table, entry=0)
        self.frontier_boundary_diagnostics = {
            "total": len(frontier_states),
            "boundary_committor": float(
                sum(float(scores.get(state, 0.0)) for state in frontier_states)
            ),
            "boundary_rate": float(
                sum(
                    float(row["k_forward"])
                    for _, row in df.iterrows()
                    if bool(row["transient"])
                    and int(row["state_connexion"]) in frontier_states
                )
            ),
        }
        for state in frontier_states:
            self.connectivity_table.change_state_to_absorbing(state)


    def is_new_state(self, system) : 
        #Loop over all other system in self.states to see if system is already known

        for state_index, state_data in self.states.items():
            are_equivalent = self.are_structures_equivalent(
                system.positions,
                state_data.system.positions,
                cell=system.cell,
                tree2=state_data.equivalence_tree(system.cell),
            )
            if are_equivalent : 
                return state_index
        return -1 


    def are_structures_equivalent(self, pos1, pos2, cell, tol=0.3, tree2=None):

        if len(pos1) != len(pos2):
            return False

        box = np.diag(cell).astype(float)
        delta = pos1 - pos2
        delta -= np.rint(delta / box) * box
        if np.all(np.linalg.norm(delta, axis=1) < tol):
            return True

        box = box.tolist()
        if tree2 is None:
            tree2 = cKDTree(pos2, boxsize=box)
        distances, _ = tree2.query(pos1, k=1, distance_upper_bound=tol)

        return np.all(distances < tol)

    def is_states_has_unknown_environments(self, state: StateData) : 
        if set(state.environment.atomic_environment_list).difference(self.known_environments) != set() :
            return True 
        else : 
            return False

    def is_state_terminal_environment(self, state: StateData) -> bool:
        return set(state.environment.atomic_environment_list) == {"crystal"}

    def _add_state(self, state_index, system=None, transient=True, applicable_events=None, visited=False, full=False ) :
        """Add a new state in the `self.states` dictionnary."""
        #to fit typing 
        neighbors_list  = []
        atomic_environment = []

        if full == True : 
            neighbors_list = NeighborsList(system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut)  
            atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add)
        else : 
            neighbors_list = None 
            atomic_environment = None 
        new_state =  StateData(system=system, environment=atomic_environment, neighbors_list=neighbors_list, transient=transient,  visited=visited)

        self.states[state_index]= new_state
