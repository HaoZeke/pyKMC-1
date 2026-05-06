# Basins

Basin settings are defined in the `[Basin]` section of the input file.
The only required parameter is the energy threshold below which a state is considered part of the basin.
You must also enable basin mode in the `[Control]` section.

Example:

```INI
[Control]
...
basin = True
...

[Basin]
energy_thr = 0.1
selector = auto
```

`selector` controls the basin exit-time implementation:

* `auto`: use AMSEL sampled FPTA when AMSEL is available, otherwise use legacy FPTA
* `legacy-fpta`: use the original NumPy/SciPy FPTA selector
* `amsel-sampled`: use AMSEL's sampled FPTA clock
* `amsel-mean`: use AMSEL's deterministic mean first-passage clock
* `amsel-adaptive`: use a sampled AMSEL rank-1 reduced clock when diagnostics support it, otherwise use sampled FPTA

AMSEL also provides catalog-level diagnostics through
`scripts/guide_amsel_catalog.py`. This script leaves the KMC loop
unchanged: it reads a saved basin connectivity table, splits each
absorbing catalog row into its own absorbing channel, and runs AMSEL's
NGT graph-transformation solver for each channel. The output ranks
catalog exits by exact committor probability from the basin entry and
reports the corresponding NGT effective rate and MFPT:

```bash
python scripts/guide_amsel_catalog.py \
  --connectivity tests/data/basin_connectivity_Cu_fake.pickle \
  --case-name cu-fixture \
  --format csv
```

The CSV/JSON rows retain `event_connexion`, `central_atom`, `sym`,
`state`, and `state_connexion`, so the ranking can be joined back to
the catalog event that produced each absorbing exit. Equal scores mean
that the current connectivity graph does not distinguish those catalog
branches kinetically.

---

## General Idea

During a KMC step, if the selected event has both forward and backward barriers lower than `energy_thr`, a `Basin` object is created.
It explores the basin, computes the exit time, and determines the exit state.
Once finished, the selected event in the KMC loop is replaced with the basin event.

FIGURE

While exploring, the main objective is to build a connectivity table containing all information required to apply the exit algorithm.
This table is stored as a `pandas.DataFrame`, managed by a `Connectivity` object (merge, remove, search for connections, ...).

FIGURE

Column meaning:

* **state**: a transient state
* **state_connexion**: a state reachable from `state`
* **event_connexion**: ID of the event that takes `state → state_connexion`
* **central_atom**: atom index on which the event must be applied
* **sym**: symmetry index of the event
* **transient**: whether `state_connexion` is transient or not
* **dE_forward / k_forward**: barrier and rate from `state` to `state_connexion`
* **dE_backward / k_backward**: barrier and rate from `state_connexion` back to `state`

_Note: Pandas Dataframe were choosen for fast querying and sorting, and it can easily be converted to a graph for analysis and characterization._

The `Basin` object uses two additional components:

* **Explorer**: explores a given state by creating its connectivity table
* **Selector**: use the connectivity table to compute the exit state and exit time

*This structure is designed to support multiple future exploration/selection algorithms.*


## Algorithm

1. **Initialization**: state to explore = current state
2. **While states remain to explore:**

   1. If the state is already known (distance-based check), stop
   2. If the state has an unknown atomic environment → mark as absorbing → stop
   3. If the state is absorbing, stop
   4. Find the connection (event) between a known state and this state
   5. Apply the event (reconstruction using a generic event)
   6. Send the state to the Explorer:

      * detect applicable events
      * build its connectivity table
   7. Merge with the global connectivity table
3. **Refine** all transient → absorbing transitions (update dE and k)
4. **Selector step**:

   * build matrix ( M ), solve ( P = e^{-Mt} P_0 )
   * use bisection to find ( t_{\text{exit}} ) and the exit state
5. Build the result and return it to the KMC loop
6. Replace the initially chosen KMC event with the basin event


The basin process may fail during PSR, refinement, reconstruction, or exit-time calculation.
If a failure occurs, the basin returns an `Err`, and the originally selected KMC event is applied instead.
