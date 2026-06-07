"""Module for executing off-lattice kinetic Monte Carlo (KMC) simulations.

This module defines the `KMC` class.
"""

from pykmc import NeighborsList, AtomicEnvironment, ActiveEventTable, Config, Reconstruction
from collections import Counter
from dataclasses import dataclass, field
import random
from .result import (
    EventSearchOutput,
    KMCLoopInfo,
    ErrorInfo,
    ErrorType,
    Result,
    AtomicEnvironmentInfo,
    ReferenceEventSearchInfo,
    ReferenceValidEventsInfo,
    RefinementsInfo,
    EventRefinementOutput,
    ReconstructionOutput,
    Err,
    Ok
)
import numpy as np
from ase.io import write
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from .algorithms import rejection_free
import sys
import pandas as pd
import pickle
from .initializer import Initializer
from .info_simulation import (
    info_atomic_environments,
    info_reference_event_searches,
    info_is_valid_reference_events,
    info_refinements,
    info_active_events,
    info_basin_events
)
from .eventsearch import EventSearch
from .refinement import Refinement
from .rate_constant import vineyard_event_prefactors_from_forces
from .log import Colors
import time
from .utils import push_towards, compute_delr
import copy
import math
from .basins.detection import DetectorThreshold
from .basins import BasinsGenericEvents

try:
    import amsel as _amsel
except ImportError:  # pragma: no cover - exercised without AMSEL installed.
    _amsel = None


@dataclass
class EnvironmentSearchEvidence:
    """Process-discovery evidence accumulated for one atomic environment."""

    attempts: int = 0
    process_counts: Counter = field(default_factory=Counter)
    process_rates: dict[object, float] = field(default_factory=dict)


def basin_exploration_trace_line(basin) -> str | None:
    order = [int(state) for state in (getattr(basin, "exploration_order", []) or [])]
    queue = [int(state) for state in (getattr(basin, "states_to_explore", []) or [])]
    guidance = {
        int(state): float(score)
        for state, score in (
            getattr(basin, "last_exploration_guidance", {}) or {}
        ).items()
    }
    if not order and not queue and not guidance:
        return None

    if queue:
        guidance_states = queue
    else:
        guidance_states = sorted(int(state) for state in guidance)
    guidance_items = [
        f"{int(state)}:{float(guidance[int(state)]):.6e}"
        for state in guidance_states
        if int(state) in guidance
    ]
    decision_items = []
    decision_process_items = []
    decision_guidance_items = []
    for decision in getattr(basin, "exploration_decisions", []) or []:
        state = int(decision["state"])
        event_family = decision.get("event_family")
        event_text = "NA" if event_family is None else str(int(event_family))
        process_signature = decision.get("process_signature")
        process_text = "NA" if process_signature is None else str(process_signature)
        decision_items.append(f"{state}:{event_text}")
        decision_process_items.append(f"{state}:{process_text}")
        decision_guidance_items.append(
            f"{state}:{float(decision.get('guidance') or 0.0):.6e}"
        )
    return (
        "\t :=> Basin exploration trace "
        f"order={','.join(str(state) for state in order)}; "
        f"queue={','.join(str(state) for state in queue)}; "
        f"guidance={','.join(guidance_items)}; "
        f"closed_events={','.join(decision_items)}; "
        f"closed_processes={','.join(decision_process_items)}; "
        f"closed_guidance={','.join(decision_guidance_items)}"
    )


def environments_with_cataloged_searches(
    atomic_environment_list,
    event_outputs,
    valid_event_results,
) -> set[str | bytes]:
    """Return environment IDs supported by catalog insertion or duplicate evidence."""
    searched_environments: set[str | bytes] = set()
    for event_output, valid_result in zip(event_outputs, valid_event_results):
        if valid_result.is_ok():
            searched_environments.add(
                atomic_environment_list[int(event_output.central_atom_index)]
            )
            continue

        error = valid_result.err_value()
        error_type = getattr(error, "type", error)
        if error_type == ErrorType.EVENT_NOT_NEW:
            searched_environments.add(
                atomic_environment_list[int(event_output.central_atom_index)]
            )
    return searched_environments


def event_search_process_evidence(
    atomic_environment_list,
    event_outputs,
    valid_event_results,
) -> dict[str | bytes, EnvironmentSearchEvidence]:
    """Return per-environment process evidence from reference-event validation."""
    evidence: dict[str | bytes, EnvironmentSearchEvidence] = {}
    for event_output, valid_result in zip(event_outputs, valid_event_results):
        environment = atomic_environment_list[int(event_output.central_atom_index)]
        if environment not in evidence:
            evidence[environment] = EnvironmentSearchEvidence()
        environment_evidence = evidence[environment]
        environment_evidence.attempts += 1

        for process_key, rate in _process_keys_from_valid_result(valid_result):
            environment_evidence.process_counts[process_key] += 1
            if rate is not None:
                environment_evidence.process_rates[process_key] = float(rate)
    return evidence


def event_search_attempt_evidence(
    atomic_environment_list,
    central_atom_research_list,
    event_search_results,
    valid_event_results,
) -> dict[str | bytes, EnvironmentSearchEvidence]:
    """Return process evidence while counting failed event-search attempts."""
    evidence: dict[str | bytes, EnvironmentSearchEvidence] = {}
    valid_result_iter = iter(valid_event_results)
    for central_atom_index, search_result in zip(
        central_atom_research_list,
        event_search_results,
    ):
        if search_result.is_ok():
            event_output = search_result.ok_value()
            environment = atomic_environment_list[int(event_output.central_atom_index)]
            process_results = _process_keys_from_valid_result(next(valid_result_iter))
        else:
            environment = atomic_environment_list[int(central_atom_index)]
            process_results = []
        if environment not in evidence:
            evidence[environment] = EnvironmentSearchEvidence()
        environment_evidence = evidence[environment]
        environment_evidence.attempts += 1
        for process_key, rate in process_results:
            environment_evidence.process_counts[process_key] += 1
            if rate is not None:
                environment_evidence.process_rates[process_key] = float(rate)
    return evidence


def undercovered_environments_for_search(
    *,
    current_environments,
    new_environments,
    visited_environments,
    environment_search_evidence,
    disable_coverage_resampling: bool = False,
    zero_observation_attempt_limit: int | None = None,
) -> list[str | bytes]:
    """Return current environment IDs that should receive event-search work.

    When ``disable_coverage_resampling`` is True, only the ``new_environments``
    intersection survives -- the AMSEL coverage-driven resampling pass over
    visited environments is skipped. Used by benchmark sweeps that need
    a deterministic per-step search budget.
    """
    current_environment_set = set(current_environments)
    searchable = []
    seen = set()
    for environment in list(new_environments):
        if environment not in current_environment_set or environment in seen:
            continue
        searchable.append(environment)
        seen.add(environment)

    if disable_coverage_resampling:
        return searchable

    known_rate_scale = current_known_process_rate_mass(
        current_environments,
        environment_search_evidence,
    )
    attempted_or_visited_environments = set(visited_environments).union(
        environment_search_evidence
    )
    productive_searchable = []
    zero_yield_searchable = []
    for environment in sorted(
        current_environment_set.intersection(attempted_or_visited_environments),
        key=str,
    ):
        if environment == "crystal" or environment in seen:
            continue
        evidence = environment_search_evidence.get(environment)
        if (
            evidence is not None
            and zero_observation_attempt_limit is not None
            and not evidence.process_counts
            and int(evidence.attempts) >= int(zero_observation_attempt_limit)
        ):
            continue
        if evidence is None or not _needs_more_process_search(
            evidence,
            known_rate_scale=known_rate_scale,
        ):
            continue
        if evidence.process_counts:
            productive_searchable.append(environment)
        else:
            zero_yield_searchable.append(environment)
        seen.add(environment)
    if productive_searchable:
        searchable.extend(productive_searchable)
    else:
        searchable.extend(zero_yield_searchable)
    return searchable


def merge_environment_search_evidence(
    target: dict[str | bytes, EnvironmentSearchEvidence],
    update: dict[str | bytes, EnvironmentSearchEvidence],
) -> None:
    """Accumulate event-search process evidence by atomic environment."""
    for environment, evidence in update.items():
        if environment not in target:
            target[environment] = EnvironmentSearchEvidence()
        target_evidence = target[environment]
        target_evidence.attempts += int(evidence.attempts)
        target_evidence.process_counts.update(evidence.process_counts)
        target_evidence.process_rates.update(evidence.process_rates)


def environment_search_evidence_trace_lines(
    evidence_by_environment: dict[str | bytes, EnvironmentSearchEvidence],
    *,
    known_rate_scale: float | None = None,
) -> list[str]:
    """Return log lines summarizing AMSEL process-coverage evidence."""
    lines = []
    for environment, evidence in sorted(
        evidence_by_environment.items(),
        key=lambda item: _environment_label(item[0]),
    ):
        if not evidence.process_counts:
            continue
        certificate = _process_search_certificate_with_rate_scale(
            evidence,
            known_rate_scale=known_rate_scale,
        )
        lines.append(
            (
                "\t :=> AMSEL process coverage env={}; "
                "attempts={}; observations={}; unique_processes={}; "
                "singleton_processes={}; missing_process_mass={:.6e}; "
                "missing_rate_mass={:.6e}; needs_more_search={}"
            ).format(
                _environment_label(environment),
                int(certificate["attempts"]),
                int(certificate["observations"]),
                int(certificate["unique_processes"]),
                int(certificate["singleton_processes"]),
                float(certificate["missing_process_mass"]),
                float(certificate["missing_rate_mass"]),
                bool(certificate["needs_more_search"]),
            )
        )
    return lines


def current_known_process_rate_mass(
    current_environments,
    evidence_by_environment: dict[str | bytes, EnvironmentSearchEvidence],
) -> float:
    """Return the current active-state rate scale from observed processes."""
    environment_counts = Counter(current_environments)
    total_rate_mass = 0.0
    for environment, multiplicity in environment_counts.items():
        if environment == "crystal":
            continue
        evidence = evidence_by_environment.get(environment)
        if evidence is None:
            continue
        total_rate_mass += float(multiplicity) * sum(
            float(rate) for rate in evidence.process_rates.values()
        )
    return float(total_rate_mass)


def _needs_more_process_search(
    evidence: EnvironmentSearchEvidence,
    *,
    known_rate_scale: float | None = None,
) -> bool:
    return bool(
        _process_search_certificate_with_rate_scale(
            evidence,
            known_rate_scale=known_rate_scale,
        )["needs_more_search"]
    )


def _process_search_certificate_with_rate_scale(
    evidence: EnvironmentSearchEvidence,
    *,
    known_rate_scale: float | None = None,
) -> dict[str, object]:
    _ = known_rate_scale
    # Process completeness is the correctness gate for additional searches.
    # Rate-scale information belongs to scheduling priority, not coverage truth.
    return dict(_process_search_certificate(evidence))


def _evidence_signature(
    evidence: EnvironmentSearchEvidence,
) -> tuple:
    """Hashable snapshot of an EnvironmentSearchEvidence.

    The certificate ``_process_search_certificate`` only depends on the
    counts, rates, and attempts on the evidence. Snapshot those into a
    hashable tuple so the certificate computation can be memoized while
    the underlying evidence object stays mutable. Returning a sorted
    item view keeps the signature stable under dict ordering.
    """
    counts = tuple(sorted(evidence.process_counts.items())) if evidence.process_counts else ()
    rates = (
        tuple(sorted(evidence.process_rates.items())) if evidence.process_rates else ()
    )
    return (counts, rates, int(evidence.attempts))


_PROCESS_SEARCH_CERTIFICATE_CACHE: dict[tuple, dict[str, object]] = {}
_PROCESS_SEARCH_CERTIFICATE_CACHE_MAX = 4096
PROCESS_SEARCH_MISSING_RATE_FLOOR = 1.0e-12


def _process_search_certificate_cache_clear() -> None:
    """Drop the per-evidence certificate cache.

    Called by the harness at trial boundaries; not used in steady state
    because evidence mutations bump the signature and re-key the cache
    naturally. Exposed primarily so tests can guarantee a clean state.
    """
    _PROCESS_SEARCH_CERTIFICATE_CACHE.clear()


def _process_search_certificate(
    evidence: EnvironmentSearchEvidence,
) -> dict[str, object]:
    signature = _evidence_signature(evidence)
    cached = _PROCESS_SEARCH_CERTIFICATE_CACHE.get(signature)
    if cached is not None:
        return cached
    certificate = _compute_process_search_certificate(evidence)
    if len(_PROCESS_SEARCH_CERTIFICATE_CACHE) >= _PROCESS_SEARCH_CERTIFICATE_CACHE_MAX:
        # Bound memory: drop oldest half. Order is insertion order.
        oldest = list(_PROCESS_SEARCH_CERTIFICATE_CACHE.keys())[
            : _PROCESS_SEARCH_CERTIFICATE_CACHE_MAX // 2
        ]
        for key in oldest:
            _PROCESS_SEARCH_CERTIFICATE_CACHE.pop(key, None)
    _PROCESS_SEARCH_CERTIFICATE_CACHE[signature] = certificate
    return certificate


def _compute_process_search_certificate(
    evidence: EnvironmentSearchEvidence,
) -> dict[str, object]:
    if not evidence.process_counts:
        return {
            "attempts": int(evidence.attempts),
            "observations": 0,
            "unique_processes": 0,
            "singleton_processes": 0,
            "missing_process_mass": 1.0,
            "missing_rate_mass": math.inf,
            "needs_more_search": True,
        }
    if _amsel is not None and hasattr(_amsel, "event_completeness"):
        # A single search can register several process observations, so the
        # observation total may exceed the attempt count; attempts is a lower
        # bound on the search budget and must cover every observation.
        observation_total = int(sum(evidence.process_counts.values()))
        attempts = max(int(evidence.attempts), observation_total)
        certificate = _amsel.event_completeness(
            process_counts=dict(evidence.process_counts),
            process_rates=evidence.process_rates,
            attempts=attempts,
            use_py_heavy_tail=False,
        )
        missing_rate_mass = float(certificate.missing_rate_mass_estimate)
        known_rate_mass = float(sum(evidence.process_rates.values()))
        return {
            "attempts": int(certificate.attempts),
            "observations": int(certificate.observations),
            "unique_processes": int(certificate.unique_processes),
            "singleton_processes": int(certificate.singleton_processes),
            "missing_process_mass": float(certificate.unseen_process_probability),
            "missing_rate_mass": missing_rate_mass,
            "needs_more_search": _needs_rate_material_process_search(
                process_needs_more=bool(certificate.needs_more_search),
                missing_rate_mass=missing_rate_mass,
                known_rate_mass=known_rate_mass,
            ),
        }
    observations = sum(evidence.process_counts.values())
    singleton_count = sum(1 for count in evidence.process_counts.values() if count == 1)
    missing_process_mass = (
        float(singleton_count) / float(observations) if observations > 0 else 1.0
    )
    known_rate_mass = sum(evidence.process_rates.values())
    if known_rate_mass <= 0.0:
        missing_rate_mass = 0.0
    elif missing_process_mass >= 1.0:
        missing_rate_mass = math.inf
    else:
        missing_rate_mass = (
            known_rate_mass * missing_process_mass / (1.0 - missing_process_mass)
        )
    return {
        "attempts": int(evidence.attempts),
        "observations": int(observations),
        "unique_processes": len(evidence.process_counts),
        "singleton_processes": int(singleton_count),
        "missing_process_mass": float(missing_process_mass),
        "missing_rate_mass": float(missing_rate_mass),
        "needs_more_search": _needs_rate_material_process_search(
            process_needs_more=missing_process_mass > 0.05,
            missing_rate_mass=missing_rate_mass,
            known_rate_mass=known_rate_mass,
        ),
    }


def _needs_rate_material_process_search(
    *,
    process_needs_more: bool,
    missing_rate_mass: float,
    known_rate_mass: float,
) -> bool:
    if not process_needs_more:
        return False
    if known_rate_mass <= 0.0:
        return True
    if math.isinf(missing_rate_mass):
        return True
    return float(missing_rate_mass) > PROCESS_SEARCH_MISSING_RATE_FLOOR


def _environment_label(environment) -> str:
    if isinstance(environment, bytes):
        return environment.hex()[:16]
    return str(environment)


def _process_keys_from_valid_result(valid_result):
    if valid_result.is_ok():
        event_rows = valid_result.ok_value()
        return [
            _process_key_and_rate(row)
            for _, row in event_rows.iterrows()
        ]

    error = valid_result.err_value()
    error_type = getattr(error, "type", error)
    if error_type != ErrorType.EVENT_NOT_NEW:
        return []
    variables = getattr(error, "variables", None) or {}
    process_key = (
        variables.get("matched_idx_ref"),
        variables.get("event_id"),
        variables.get("id_final"),
    )
    if process_key == (None, None, None):
        return []
    return [(process_key, variables.get("k"))]


def _process_key_and_rate(row) -> tuple[tuple[object, object, object], float | None]:
    return (
        (
            int(row["idx_ref"]) if row.get("idx_ref") is not None else None,
            row.get("event_id"),
            row.get("id_final"),
        ),
        row.get("k"),
    )


# NOTE can maybe reimplment tries if empty catalog
#TODO: Add reconstruction info

class KMC:
    """Manage and execute the Kinetic Monte Carlo (KMC) simulation.

    This class acts as the central controller coordinating all phases of the simulation:
    initialization, event search, event refinement, event selection, system updates,
    minimization, logging, and termination.

    Attributes
    ----------
    config : Config
        The parameters of the simulation.
    loggers : LogKMC
        Handle logging of simulation progress.
    system : System
        The atomic system.
    engine : Engine
        The E/F engine used.
    neighbors_list : NeighborsList
        Store neighbors of atoms in the system.
    atomic_environment : AtomicEnvironment
        Store atomic environment of atoms in the system
    reference_table : ReferenceEventTable
        Store generic events that can be apply to the system.
    visited_environment : set[str|bytes]
        Track atomic environments already explored. Those for which event searches as been previously done.
    total_energy : float
        The total energy of the system.

    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.loggers = None
        self.system = None
        self.manager = None
        self.engine = None
        self.neighbors_list = None
        self.atomic_environment = None
        self.reference_table = None
        self.visited_environments = None
        self.environment_search_evidence: dict[str | bytes, EnvironmentSearchEvidence] = {}
        self.total_energy = None
        self.potential_energy = None

    def run(self) -> None:
        """Run the simulation."""
        self._seed_rngs()
        # Initialize the simulation, KMC attributes and minimize the system
        #self._initialize()
        self.manager.initialize_sessions(self.config, self.system)
        self.minimize_system()
        self.neighbors_list = NeighborsList(
                self.system,
                self.config.atomicenvironment.rnei,
                self.config.atomicenvironment.rcut,
            )
        self.atomic_environment = AtomicEnvironment(
                self.config.atomicenvironment.style,
                self.neighbors_list.neighbors_list["rnei"],
                self.neighbors_list.neighbors_list["rcut"],
                self.config.atomicenvironment.neighbors_add,
            )
        #Set new positions to all sessions/engine : 
        self.manager.use_local()
        self.manager.set_all_positions(self.system.positions)

        if self.config.control.restart_file is None: 
        # Write initial step to file
            self._append_snapshot_to_trajectory()
            last_step = 0 
            total_time = 0.0

        else : #read restart file
            self.loggers.info("log", ":=> Reading restart file")
            restart_info = np.load(self.config.control.restart_file) 
            last_step = restart_info["last_step"]
            total_time = restart_info["last_time"]
            self.loggers.info("log", ":=> last step = {}, last_end_time = {}ps".format(last_step, total_time))

        # LOOP KMC PARAMETERS
        nkmc_steps = self.config.control.n_steps
        last_step +=1 
        nsearch = self.config.eventsearch.nsearch

        

        # KMC LOOP
        for step in range(last_step, nkmc_steps+last_step):
            start_real = time.time()
            start_cpu = time.process_time()

            self.loggers.info(
                "log",
                "{}{}Step : {}{}".format(
                    Colors.BOLD.value, Colors.YELLOW.value, step, Colors.RESET.value
                ),
            )

            # == amsel barrierless capture (recombination sink) ==
            # The final V/SIA capture is downhill with no saddle, so neither
            # pARTn nor any min-mode search ever proposes it. When the defect
            # pair is within the SPONTANEOUS capture radius (the amsel
            # recomb product minimizes to a defect-free lower-energy state),
            # apply it directly as one downhill kMC step instead of searching.
            if getattr(self.config.control, "amsel_recomb_inject", False):
                dt_cap = self._try_amsel_capture()
                if dt_cap is not None:
                    total_time += dt_cap
                    self._save()
                    self._append_snapshot_to_trajectory()
                    self.neighbors_list = NeighborsList(
                        self.system,
                        self.config.atomicenvironment.rnei,
                        self.config.atomicenvironment.rcut,
                    )
                    self.atomic_environment = AtomicEnvironment(
                        self.config.atomicenvironment.style,
                        self.neighbors_list.neighbors_list["rnei"],
                        self.neighbors_list.neighbors_list["rcut"],
                        self.config.atomicenvironment.neighbors_add,
                    )
                    if set(self.atomic_environment.atomic_environment_list) == {"crystal"}:
                        self.loggers.info("log", ":=> Only atoms with cristalline environment")
                        self._close()
                        break
                    continue

            # == Find Current atomic environments that has not been visited ==
            new_environments = self.get_new_environments()
            # amsel KDB reuse: inject cached events for environments already
            # in the persistent catalogue and drop them from the search list
            # (skip pARTn). This is what stops the kMC starving when fresh
            # pARTn searches return "no event found".
            new_environments = self._reuse_events_from_kdb(new_environments)
            event_search_results, results_is_valid_events = (
                self.search_reference_events_until_covered(
                    new_environments,
                    nsearch,
                )
            )
            # == Refinement ==
            ##=>Subset of reference_event_table with generic event that can be apply to the current step (ie event_id in atomic environment)
            subset_reference_event_table = self.reference_table.has_id_subset_table(
                self.atomic_environment.atomic_environment_list
            )
            ##=>Refines all event in subset
            refinement = self.execute_refinements(subset_reference_event_table)

            # == ADD ACTIVE EVENT TO ACTIVE EVENT TABLE ==
            active_table = self.add_active_events(refinement.get_successes_results())
            active_table.remove_duplicates(self.system.cell, self.neighbors_list)  #To be sure
            self.loggers.info("log", "\t :=> {} active events after removing duplicates.".format(len(active_table.table)))


            # == Update System ==
            self.manager.use_global()
            result_reconstruction, delta_t, ktot, idx_selected_event, err_reference, err_ae = self.reconstruction(active_table)
            events_info = info_active_events(self.system.types, self.reference_table, active_table)
            if len(err_reference) != 0 :
                self.loggers.info("log", "\t :=> Removing reference event from which reconstruction failed.")
                self.reference_table.remove(list(set(err_reference)))
                self.loggers.info("log", "\t :=> Removing topology from known environments from which reconstruction failed.")
                self.visited_environments = self.visited_environments.difference(set(err_ae))
            events_info = events_info.output_msg()




            #INFO :
            self.loggers.events_file_step_first_line("events", step)
            self.loggers.events_applicable_info_line("events", idx_selected_event)
            self.loggers.info("events", events_info)

                #TODO: Temporary, need to unified kmc main loop and basin operations + ugly
            detector = DetectorThreshold()
                #IF selected event shows we are in a basin
            if self.config.control.basin and detector.detect(active_table.table.iloc[idx_selected_event], self.reference_table.table, self.config.basin.energy_thr, True) :
                self.loggers.info("log","\t :=> System is in a Basin." )
                self.loggers.info("log","\t :=> Exploring the Basin." )
                #get basin info/explore
                basin = BasinsGenericEvents(self.config, self.reference_table, self.visited_environments, self.manager)
                self.system.update_positions(result_reconstruction.ok_value().min1_positions)
                result_basin = basin.execute(self.system)
                trace_line = basin_exploration_trace_line(basin)
                if trace_line is not None:
                    self.loggers.info("log", trace_line)
                frontier = getattr(basin, "unresolved_frontier_diagnostics", {})
                if int(frontier.get("total", 0)) > 0:
                    boundary = getattr(basin, "frontier_boundary_diagnostics", {})
                    if int(boundary.get("total", 0)) > 0:
                        self.loggers.info(
                            "log",
                            (
                                "\t :=> Basin frontier boundary absorbed {} states; "
                                "boundary_committor={:.6e}; boundary_rate={:.6e}"
                            ).format(
                                int(boundary["total"]),
                                float(boundary["boundary_committor"]),
                                float(boundary["boundary_rate"]),
                            ),
                        )
                    else:
                        self.loggers.info(
                            "log",
                            (
                                "\t :=> Basin exploration budget left {} frontier states; "
                                "unresolved_committor={:.6e}; unresolved_rate={:.6e}"
                            ).format(
                                int(frontier["total"]),
                                float(frontier["unresolved_committor"]),
                                float(frontier["unresolved_rate"]),
                            ),
                        )
                basin_refinement = getattr(basin, "absorbing_refinement_diagnostics", {})
                if int(basin_refinement.get("skipped", 0)) > 0:
                    self.loggers.info(
                        "log",
                        (
                            "\t :=> Basin absorbing refinement skipped {} exits; "
                            "unresolved_committor={:.6e}; unresolved_rate={:.6e}"
                        ).format(
                            int(basin_refinement["skipped"]),
                            float(basin_refinement["unresolved_committor"]),
                            float(basin_refinement["unresolved_rate"]),
                        ),
                    )
                if result_basin.is_ok() : #Basin did no fail
                #move system to a state connected to the exit_state
                    self.system.update_positions(result_basin.ok_value().initial_system_positions)
                    self.neighbors_list = basin.states[result_basin.ok_value().from_state].neighbors_list
                #construct new active table with only event : new_actual_state - > exit_state
                    tmp_active_table = ActiveEventTable(self.config)
                    tmp_event = EventRefinementOutput(central_atom_index=result_basin.ok_value().central_atom,
                                                      saddle_positions=result_basin.ok_value().saddle_positions,
                                                      E_saddle=-1,
                                                      min2_positions=result_basin.ok_value().final_positions,
                                                      dE_forward=result_basin.ok_value().energy_barrier,
                                                      num_reference_event=result_basin.ok_value().num_reference_event)
                    neighbors = result_basin.ok_value().neighbors
                    tmp_active_table.add_events(tmp_event)
                #reconstruct event
                    self.manager.use_global()
                    result_basin_reconstruction = self._reconstruction_active_event(0, tmp_active_table)
                    if result_basin_reconstruction.is_ok() :
                        self.system.update_positions(result_basin_reconstruction.ok_value().min2_positions)
                        self.total_energy = result_basin_reconstruction.ok_value().min2_etot
                        delta_t = result_basin.ok_value().t_exit
                        ktot = result_basin.ok_value().k_tot
                        idx_selected_event = 0
                        active_table.table = tmp_active_table.table

                        #INFO
                        idx_exit_event, basin_info = info_basin_events(self.system.types, self.reference_table, basin.connectivity_table, result_basin.ok_value().exit_state)
                        basin_info = basin_info.output_msg()
                        self.loggers.events_basin_info_line("events",idx_exit_event )
                        self.loggers.info("events", basin_info)


                    else :
                       self.loggers.info("log", "\t :=> Reconstruction Exit State Basin fails with error {}, back to original event".format(result_basin_reconstruction.err_value()))
                       self.system.update_positions(basin.states[0].system.positions)
                       self.system.update_positions(result_reconstruction.ok_value().min2_positions)
                else :
                    self.loggers.info("log", "\t :=> Basin fails with error : {}, back to original event".format(result_basin.err_value()))
                    self.system.update_positions(result_reconstruction.ok_value().min2_positions)
                if basin.connectivity_table is not None :
                    basin.connectivity_table.save('basin_connectivity_'+str(step)+'.pickle')
                #update delta_t, ktot (use basin infos)
            else :
                self.system.update_positions(result_reconstruction.ok_value().min2_positions)
                self.total_energy = result_reconstruction.ok_value().min2_etot
            total_time += delta_t * 10**-12  # time is in seconds

            ###=> Synchronise all lammps instances with new positions 
            self.manager.use_local()
            self.manager.set_all_positions(positions=self.system.positions)
            ##=>Minimize

            # == Log informations ==
            atomic_environment_info = self.get_info_atomic_environments(
                new_environments
            )
            reference_event_searches_info = self.get_info_reference_event_searches(
                event_search_results
            )
            is_valid_events_info = self.get_info_is_valid_reference_events(
                results_is_valid_events
            )
            refinements_info = self.get_info_refinements(refinement.results)
            kmc_loop_info = KMCLoopInfo(
                step=step,
                atomic_environment_info=atomic_environment_info,
                reference_event_searches_info=reference_event_searches_info,
                valid_event_info=is_valid_events_info,
                refinements_info=refinements_info,
            )
            self.loggers.info("info", kmc_loop_info.output_msg())


            elapsed_real = time.time() - start_real
            elapsed_cpu = time.process_time() - start_cpu
            
            self.loggers.table_line_info_kmc(
                "output",
                step,
                delta_t * 10**-12,
                total_time,
                active_table.table.loc[idx_selected_event].at["num_reference_event"],
                active_table.table.loc[idx_selected_event].at["energy_barrier"],
                active_table.table.loc[idx_selected_event].at["k"],
                ktot,
                self.total_energy,
                elapsed_cpu, 
                elapsed_real
            )

            # == Update variables ==
            self.neighbors_list = NeighborsList(
                self.system,
                self.config.atomicenvironment.rnei,
                self.config.atomicenvironment.rcut,
            )
            self.atomic_environment = AtomicEnvironment(
                self.config.atomicenvironment.style,
                self.neighbors_list.neighbors_list["rnei"],
                self.neighbors_list.neighbors_list["rcut"],
                self.config.atomicenvironment.neighbors_add,
            )

            # == Save Reference Table and List visited environment :
            self._save()
            self._append_snapshot_to_trajectory()
            del active_table
            # == Check if only cristalline environments ==
            if set(list(self.atomic_environment.atomic_environment_list)) == {
                "crystal"
            }:
                self.loggers.info("log", ":=> Only atoms with cristalline environment")
                self._close()
        self._save_restart_file(step, total_time)
        self._close()

    def _seed_rngs(self) -> None:
        seed = getattr(self.config.control, "random_seed", None)
        if seed is None:
            return
        random.seed(int(seed))
        np.random.seed(int(seed))

    def _reuse_events_from_kdb(self, new_environments: list) -> list:
        """For each new environment, query the amsel KDB; on a HIT inject
        the cached reference events and mark the environment visited so
        pARTn does not re-search it. Returns the environments still needing
        a search."""
        kdb = getattr(self.reference_table, "kdb", None)
        if kdb is None:
            return new_environments
        remaining = []
        reused_envs = 0
        reused_events = 0
        for env in new_environments:
            rows = kdb.lookup_rows(env)
            if rows:
                self.reference_table.ingest_rows(rows)
                try:
                    self.visited_environments.add(env)
                except AttributeError:
                    self.visited_environments = set(self.visited_environments) | {env}
                reused_envs += 1
                reused_events += len(rows)
            else:
                remaining.append(env)
        if reused_events:
            self.loggers.info(
                "log",
                "\t :=> Reused {} events from amsel KDB for {} environments "
                "(skipped pARTn search)".format(reused_events, reused_envs),
            )
        return remaining

    def _try_amsel_capture(self):
        """Apply the barrierless V/SIA recombination directly when the pair
        is within the spontaneous capture radius. Returns dt (seconds) on
        capture, else None.

        Validated by minimizing the amsel recomb product: fires ONLY when
        the product is defect-free AND lower in energy than the current
        state (a true downhill sink), so metastable separations fall
        through to normal saddle-based migration. This supplies the one
        transition no saddle search can find (the sink has no saddle)."""
        try:
            from .basins.amsel_recomb import (
                build_product, detect_recomb, n_defects,
            )
        except Exception:
            return None
        pos = self.system.positions
        cell = self.system.cell
        cap = float(getattr(self.config.partn, "amsel_recomb_capture_mult", 1.6))
        det = detect_recomb(pos, cell, capture_mult=cap)
        if det is None:
            return None
        source, v_centroid = det
        try:
            product = build_product(pos, cell, source, v_centroid)
        except Exception:
            return None
        nd_cur = n_defects(pos, cell)
        self.manager.use_global()
        try:
            res = self.manager.minimize_with_results(
                self.config, positions=np.asarray(product, dtype=float)
            ).result()
        except Exception:
            return None
        if res is None:
            return None
        min_pos, e_prod = res
        nd_prod = n_defects(min_pos, cell)
        absorb = max(2, int(0.4 * nd_cur))
        e_cur = self.total_energy
        if nd_prod > absorb:
            return None  # product still defected -> not a capture, migrate
        if e_cur is not None and e_prod is not None and e_prod >= e_cur:
            return None  # not downhill -> metastable, migrate
        # Apply the downhill capture as one kMC step.
        self.system.update_positions(min_pos)
        self.total_energy = e_prod
        self.manager.use_local()
        self.manager.set_all_positions(positions=self.system.positions)
        pref = float(getattr(self.config.rateconstant, "prefactor", 1.0e13))
        dt = (1.0 / pref) if pref > 0 else 1.0e-13
        self.loggers.info(
            "log",
            "\t :=> amsel barrierless capture applied (downhill sink, "
            "n_defects {}->{}); dt={:.3e}s".format(nd_cur, nd_prod, dt),
        )
        return dt

    def get_new_environments(self) -> list[str | bytes]:
        """Get atomic environments of the current system that has not been already explored.

        Returns
        -------
        list[str|bytes]
            The atomic environments of the current system that are encounter for the first time.

        """
        new_environments = self.atomic_environment.get_new_environments(
            self.visited_environments
        )
        self.loggers.info(
            "log",
            "\t :=> {} new atomic environments found".format(len(new_environments)),
        )
        return new_environments

    def search_reference_events_until_covered(
        self,
        new_environments: list[str | bytes],
        nsearch: int,
    ) -> tuple[
        list[Result[EventSearchOutput, ErrorInfo]],
        list[Result[pd.DataFrame, ErrorInfo]],
    ]:
        """Search reference events until AMSEL process evidence is covered."""
        disable_coverage_resampling = bool(
            getattr(self.config.control, "disable_coverage_resampling", False)
        )
        searches_per_environment = max(1, int(nsearch))
        search_environments = undercovered_environments_for_search(
            current_environments=self.atomic_environment.atomic_environment_list,
            new_environments=new_environments,
            visited_environments=self.visited_environments,
            environment_search_evidence=self.environment_search_evidence,
            disable_coverage_resampling=disable_coverage_resampling,
            zero_observation_attempt_limit=searches_per_environment,
        )
        all_event_search_results: list[Result[EventSearchOutput, ErrorInfo]] = []
        all_valid_event_results: list[Result[pd.DataFrame, ErrorInfo]] = []
        first_round = True

        while search_environments:
            if disable_coverage_resampling:
                round_count = 1
                round_nsearch = searches_per_environment
            else:
                round_count = searches_per_environment
                round_nsearch = 1
            for _search_round in range(round_count):
                if not search_environments:
                    break
                if first_round:
                    repeated_environments = set(search_environments).difference(
                        set(new_environments)
                    )
                else:
                    repeated_environments = set(search_environments)
                if repeated_environments:
                    self.loggers.info(
                        "log",
                        "\t :=> Resampling {} undercovered atomic environments".format(
                            len(repeated_environments)
                        ),
                    )

                central_atom_research_list = self.central_atoms_research(
                    search_environments, round_nsearch
                )
                event_search = self.execute_event_searches(central_atom_research_list)
                all_event_search_results.extend(event_search.results)

                event_search_outputs = event_search.get_successes_results()
                results_is_valid_events = self.add_reference_events(
                    event_search_outputs
                )
                all_valid_event_results.extend(results_is_valid_events)

                searched_environments = environments_with_cataloged_searches(
                    self.atomic_environment.atomic_environment_list,
                    event_search_outputs,
                    results_is_valid_events,
                )
                merge_environment_search_evidence(
                    self.environment_search_evidence,
                    event_search_evidence_update := event_search_attempt_evidence(
                        self.atomic_environment.atomic_environment_list,
                        central_atom_research_list,
                        event_search.results,
                        results_is_valid_events,
                    ),
                )
                cumulative_event_search_evidence = {
                    environment: self.environment_search_evidence[environment]
                    for environment in event_search_evidence_update
                }
                for line in environment_search_evidence_trace_lines(
                    cumulative_event_search_evidence,
                    known_rate_scale=current_known_process_rate_mass(
                        self.atomic_environment.atomic_environment_list,
                        self.environment_search_evidence,
                    ),
                ):
                    self.loggers.info("log", line)
                self.loggers.info(
                    "log",
                    "\t :=> Marking {} atomic environments as searched".format(
                        len(searched_environments.difference(self.visited_environments))
                    ),
                )
                self.visited_environments.update(searched_environments)
                search_environments = undercovered_environments_for_search(
                    current_environments=self.atomic_environment.atomic_environment_list,
                    new_environments=[],
                    visited_environments=self.visited_environments,
                    environment_search_evidence=self.environment_search_evidence,
                    disable_coverage_resampling=disable_coverage_resampling,
                    zero_observation_attempt_limit=searches_per_environment,
                )
                if len(self.reference_table.table) == 0 and not search_environments:
                    self.loggers.error(
                        "log",
                        "No events have been found, empty reference events table. \n \tTry to increase nsearch or saddle point search algorithm's parameters. \n \tClosing the simulation.",
                    )
                    self._close()
                first_round = False
        return all_event_search_results, all_valid_event_results

    def central_atoms_research(
        self, new_environments: list[str | bytes], nsearch: int
    ) -> list[int]:
        """Generate list of central atoms on which we gonna perform generic event searches for the reference table.

        For each new environment it adds nseach atoms having that environment to the list.

        Parameters
        ----------
        new_environments : list[str|bytes]
            List of atomic environment ID.
        nsearch : int
            Number of searches per atomic environment.

        Returns
        -------
        list[int]
            List of central atoms

        Raises
        ------
        IndexError
            If no atoms are found for a given environment, random.choice will raise an IndexError.

        """
        central_atom_research_list = []
        # for each atomic environment hash in new_environment
        for env in new_environments:
            # find all index having that hash
            tmp1 = [
                i
                for i, e in enumerate(self.atomic_environment.atomic_environment_list)
                if e == env
            ]
            n_unique = min(int(nsearch), len(tmp1))
            tmp2 = random.sample(tmp1, n_unique)
            if int(nsearch) > n_unique:
                tmp2 += [random.choice(tmp1) for _i in range(int(nsearch) - n_unique)]
            central_atom_research_list += tmp2
        return central_atom_research_list

    def execute_event_searches(
        self, central_atom_research_list: list[int]
    ) -> EventSearch:
        """Execute an event search for each atom index in central_atom_research_list.

        Parameters
        ----------
        central_atom_research_list : list[int]
            The list of atom index on which we want to perform and event search.

        Returns
        -------
        EventSearch
            The EventSearch class containing results of the event searches.

        """
        event_search = EventSearch(self.config, self.system, self.manager, self.loggers)
        event_search.execute(central_atom_research_list)
        return event_search

    def add_reference_events(
        self, events: list[EventSearchOutput]
    ) -> list[pd.DataFrame]:
        """Add events to the reference table.

        Parameters
        ----------
        events : list[EventSearchOutput]
            List containing EventSearchOutput dataclass of successful events.

        Returns
        -------
        list[pd.DataFrame]
            List of event dataframe that has been added to the reference event table.

        """
        self._attach_vineyard_prefactors(events)
        results_is_valid_events = self.reference_table.add_events(events)
        self.loggers.info(
            "log",
            "\t :=> Adding {} events to the reference table".format(
                len([e for e in results_is_valid_events if e.is_ok()])
            ),
        )
        return results_is_valid_events

    def _attach_vineyard_prefactors(self, events: list[EventSearchOutput]) -> None:
        rate_cfg = getattr(self.config, "rateconstant", None)
        if getattr(rate_cfg, "style", "constant") != "amsel-vtst":
            return
        if not bool(getattr(rate_cfg, "compute_vineyard_prefactor", False)):
            return
        for event in events:
            try:
                active_indices = self._vineyard_active_indices(event)
                masses_amu = self._vineyard_masses_amu(active_indices)
                force_getter = getattr(
                    self.manager, "global_get_forces", self.manager.get_forces
                )

                def force_fn(positions):
                    positions = np.asarray(positions, dtype=float)
                    forces = force_getter(positions=positions)
                    if hasattr(forces, "result"):
                        forces = forces.result()
                    forces = np.asarray(forces, dtype=float)
                    if forces.ndim == 1 and forces.size == positions.size:
                        forces = forces.reshape(positions.shape)
                    return forces

                prefactors = vineyard_event_prefactors_from_forces(
                    force_fn,
                    event.min1_positions,
                    event.saddle_positions,
                    event.min2_positions,
                    active_indices=active_indices,
                    masses_amu=masses_amu,
                    step_A=float(getattr(rate_cfg, "vineyard_fd_step_A", 1.0e-3)),
                )
            except Exception as exc:
                if getattr(self, "loggers", None) is not None:
                    self.loggers.info(
                        "log",
                        "\t :=> Vineyard prefactor failed for event at atom {}: {}".format(
                            event.central_atom_index, exc
                        ),
                    )
                continue
            event.prefactor_inv_s = prefactors.forward_prefactor_inv_s
            event.product_prefactor_inv_s = prefactors.backward_prefactor_inv_s
            event.prefactor_source = "vineyard-finite-difference"
            event.saddle_freq_invcm = prefactors.saddle_freq_invcm
            event.barrier_omega_rad_per_s = prefactors.barrier_omega_rad_per_s

    def _vineyard_active_indices(self, event: EventSearchOutput) -> list[int]:
        positions = np.asarray(event.min1_positions, dtype=float)
        saddle = np.asarray(event.saddle_positions, dtype=float)
        product = np.asarray(event.min2_positions, dtype=float)
        center = int(event.move_atom_index)
        if center < 0 or center >= len(positions):
            center = int(event.central_atom_index)
        if center < 0 or center >= len(positions):
            raise ValueError("Vineyard prefactor event center is out of range")
        cell = getattr(event, "cell", None)
        if cell is None:
            cell = getattr(self.system, "cell", None)
        if saddle.shape != positions.shape or product.shape != positions.shape:
            raise ValueError(
                "Vineyard prefactor event positions must have matching shapes"
            )

        def minimum_image_delta(target, reference):
            delta = np.asarray(target, dtype=float) - np.asarray(reference, dtype=float)
            active_cell = cell
            if active_cell is not None:
                active_cell = np.asarray(active_cell, dtype=float)
                if active_cell.shape == (3, 3):
                    lengths = np.diag(active_cell)
                    for axis, length in enumerate(lengths):
                        if length > 0.0:
                            delta[:, axis] -= length * np.round(
                                delta[:, axis] / length
                            )
            return delta

        displacement = np.maximum.reduce(
            [
                np.linalg.norm(minimum_image_delta(saddle, positions), axis=1),
                np.linalg.norm(minimum_image_delta(product, positions), axis=1),
                np.linalg.norm(minimum_image_delta(product, saddle), axis=1),
            ]
        )
        fd_step = float(
            getattr(
                getattr(self.config, "rateconstant", None),
                "vineyard_fd_step_A",
                1.0e-3,
            )
        )
        displacement_tol = max(10.0 * fd_step, 1.0e-3)
        active = {
            int(index)
            for index in np.flatnonzero(displacement > displacement_tol)
        }
        active.add(center)
        central = int(event.central_atom_index)
        if 0 <= central < len(positions):
            active.add(central)
        return sorted(active)

    def _vineyard_masses_amu(self, active_indices: list[int]) -> list[float]:
        types = getattr(self.system, "types", None)
        if types is None:
            raise ValueError("Vineyard prefactor requires system atom types")
        return [
            float(atomic_masses[atomic_numbers[str(types[int(index)])]])
            for index in active_indices
        ]

    def execute_refinements(self, df_reference_events: pd.DataFrame) -> Refinement:
        """Refine all events in df_reference_events for all atoms on which they can be apply.

        Parameters
        ----------
        df_reference_events : pd.DataFrame
            Subset of the reference table with events that can be apply to the current system.

        Returns
        -------
        Refinement
            The refinement class with results.

        """
        refinement = Refinement(
            self.config,
            self.loggers,
            self.system,
            self.neighbors_list,
            self.atomic_environment,
            self.manager,
        )
        #refinement.execute(df_reference_events, self.potential_energy)
        refinement.execute(df_reference_events, self.total_energy)
        return refinement

    def add_active_events(
        self, events: list[EventRefinementOutput]
    ) -> ActiveEventTable:
        """Create a new ActiveEventTable, add active events and return it.

        Parameters
        ----------
        events : list[RefinementsInfo]
            List of events to be added.

        Returns
        -------
        ActiveEventTable
            The active event table object.

        """
        active_table = ActiveEventTable(self.config)
        active_table.add_events(events)
        return active_table

    def _select_event(self, active_table: ActiveEventTable) -> tuple[int, float, float]:
        """Select an event in the active table based on the refection free algorithm.

        Parameters
        ----------
        active_table : ActiveEventTable
            The ActiveEventTable object with active events.

        Returns
        -------
        tuple[int, float, float]
            A typle containing :
            - int: Index of the selected event in the ActiveEventTable table.
            - float: time increment associated with the event.
            - float: total rate constant of the active events.

        """
        # list of rate constant
        l_k = np.array(
            [active_table.table.loc[i].at["k"] for i in range(len(active_table.table))]
        )
        idx_selected_event, delta_t, ktot = rejection_free(l_k)
        return idx_selected_event, delta_t, ktot


    def reconstruction(self, active_table) : 
            #TODO make a Result

            err_reference = []
            err_ae = []
            while len(active_table.table) > 0 : 
                ##=>Select event
                idx_selected_event, delta_t, ktot = self._select_event(active_table)
                ##=>Reconstruct event 
                self.loggers.info("log", "\t :=> Event Reconstruction")
                result_reconstruction = self._reconstruction_active_event(idx_selected_event, active_table)
                if result_reconstruction.is_ok() : 
                    break 
                else : 
                    num_ref_event = active_table.table.loc[idx_selected_event].at['num_reference_event']
                    self.loggers.info("log", "\t :=> Reconstruction fails (reference event {}) :  {}".format(num_ref_event, result_reconstruction.err_value().message))
                    ae_topo = self.reference_table.table[self.reference_table.table['idx_ref'] == num_ref_event]['event_id'].values[0]
                    err_reference.append(num_ref_event)
                    err_ae.append(ae_topo)

                    self.loggers.info("log", "\t :=> Removing active event.")
                    active_table.remove(idx_selected_event)
            else : 
                self.loggers.error("log", "All event reconstuctions failed.")
                self._close()
            return result_reconstruction, delta_t, ktot, idx_selected_event, err_reference, err_ae

    def _reconstruction_active_event(self, idx_selected_event: int, active_table: AtomicEnvironment) :
        central_atom = active_table.table.loc[idx_selected_event].at["atom_index"]
        neighbors = self.neighbors_list.get_neighbors("rcut", central_atom)
        saddle_positions = copy.deepcopy(active_table.table.loc[idx_selected_event].at["saddle_positions"])
        supposed_final_positions = copy.deepcopy(active_table.table.loc[idx_selected_event].at["final_positions"])
        supposed_initial_positions = copy.deepcopy(self.system.positions[neighbors])




        #Move the system to the saddle point
        self.system.update_positions(new_positions= saddle_positions, atom_idx = neighbors)

        #try to reconstruct
        result = Reconstruction(self.config, self.manager).reconstruct(supposed_initial_positions, supposed_final_positions, self.system.positions, self.system.cell, self.config.psr.matching_score_thr, neighbors)
        #result with min1, saddle, min2 pos

        #Back to original positions, in case reconstruction fails
        self.system.update_positions(new_positions = supposed_initial_positions, atom_idx = neighbors)
        return result

    def _apply_event(
        self, idx_selected_event: int, active_table: ActiveEventTable
    ) -> None:
        """Apply an active event to the system.

        Parameters
        ----------
        idx_selected_event : int
            index of the selected event in the active_table's table
        active_table : ActiveEventTable
            The ActiveEventTable okbject with active events.

        """
        new_positions = active_table.table.loc[idx_selected_event].at["final_positions"]
        self.system.update_positions(new_positions)

    def minimize_system(self, positions = None) -> None:
        """Minimize the system and update its positions."""
        if self.config.control.restart_file is None: 
            self.loggers.info("log", ":=> Minimizing the system")
        else : 
            self.loggers.info("log", ":=> Computing energies")
        new_positions, total_energy = self.manager.global_minimize_with_results(self.config, positions=positions)
        #TEST
        #future = self.manager.minimize_with_results(self.config, positions=positions)
        #new_positions, total_energy = future.result()
        #np.savetxt('before_min.dat', self.system.positions)
        #np.savetxt('after_min.dat', new_positions)
        if self.config.control.restart_file is None : 
            self.system.update_positions(new_positions)
        self.total_energy = total_energy
        self.potential_energy = self.manager.global_get_potential_energy()

    def get_info_atomic_environments(
        self, new_environments: list[str | bytes]
    ) -> AtomicEnvironmentInfo:
        """Get atomic environments informations for outputs.

        See :func:`pykmc.info_simulation.info_atomic_environments`.

        Parameters
        ----------
        new_environments : list[str | bytes]
            List of new environments detected.

        Returns
        -------
        AtomicEnvironmentInfo
            The Dataclass with atomic environments informations.

        """
        return info_atomic_environments(self, new_environments)

    def get_info_reference_event_searches(
        self,
        results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]],
    ) -> ReferenceEventSearchInfo:
        """Get reference event searches informations for outputs.

        See :func:`pykmc.info_simulation.info_reference_event_searches`.

        Parameters
        ----------
        results_reference_event_searches : list[Result[EventSearchOutput, ErrorInfo]]
            The list of Result from event searches.

        Returns
        -------
        ReferenceEventSearchInfo
            The Dataclass with reference event searches informations.

        """
        return info_reference_event_searches(results_reference_event_searches)

    def get_info_is_valid_reference_events(
        self, results_is_valid_events: list[Result[pd.DataFrame, ErrorInfo]]
    ) -> ReferenceValidEventsInfo:
        """Get informations on whether or not an event is valid.

        See :func:`pykmc.info_simulation.info_is_valid_reference_events`.

        Parameters
        ----------
        results_is_valid_events : list[Result[pd.DataFrame, ErrorInfo]]
            List of Results from ReferenceEventTable.is_valid_event().

        Returns
        -------
        ReferenceValidEventsInfo
            The Dataclass with information on whether an event is valid or not.

        """
        return info_is_valid_reference_events(results_is_valid_events)

    def get_info_refinements(
        self, results_refinements: list[Result[EventSearchOutput, ErrorType]]
    ) -> RefinementsInfo:
        """Get informations on refined events.

        See :func:`pykmc.info_simulation.info_refinements`.

        Parameters
        ----------
        results_refinements : list[Result[EventSearchOutput, ErrorType]]
           List of Results from the refinements.

        Returns
        -------
        RefinementsInfo
           The dataclass with refinements informations.

        """
        return info_refinements(results_refinements)

    def _initialize(self) -> None:
        """Initialize the KMC attributes.

        See :func:`pykmc.Initializer.initialize()`.

        """
        Initializer(self).initialize()

    def _append_snapshot_to_trajectory(self) -> None:
        """Append the configurations positions to the trajectory file."""
        atoms = Atoms(
            self.system.types,
            positions=self.system.positions,
            cell=self.system.cell,
            pbc=self.system.pbc,
        )
        write(self.config.control.trajectory_output, atoms, append=True)

    def _save(self) -> None:
        """Save the reference event table and the list of visited environments."""
        self.reference_table.save("reference_table.pickle")
        with open(self.config.control.visited_environments_output, "wb") as file:
            pickle.dump(self.visited_environments, file)

    def _save_restart_file(self, last_step, last_time) : 
        """ 
        Save end simulation informations
        """
        np.savez("restart_"+str(last_step)+".npz", 
                 last_step = last_step, 
                 last_time = last_time)


    def _close(self) -> None:
        """Close the simulation."""
        self.loggers.info("log", ":=> End of simulation")
        self.manager.close_all()
        sys.exit()
