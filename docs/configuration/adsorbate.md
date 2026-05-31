# Adsorbate Workflow Configuration

This page documents the YAML keys used by:

- `AdsorbateCMCWorkflow`
- `AdsorbateReplicaExchangeWorkflow`
- `AdsorbateGCMCWorkflow`
- `AdsorbateGCMCScanWorkflow`

## 1. Adsorbate identity and site construction

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `adsorbate` | all adsorbate workflows | `H` for CMC, `None` for GCMC | Adsorbate identity. Supported built-ins are `H`, `OH`, `OOH`, or a path to an ASE-readable structure such as `H2O.xyz`. |
| `adsorbate_anchor_index` | all | `0` | Anchor atom inside a molecular adsorbate. Site placement and reorientation keep this atom fixed. |
| `site_elements` | all | `[]` | Elements used to generate adsorption sites. |
| `site_type` | all | `['atop']` for CMC, `['atop', 'bridge']` for GCMC | Site families to build. Supported by the site registry: `atop`, `bridge`, `fcc`, `hcp`. |
| `substrate_elements` | all | `[]` | Elements treated as the slab framework. Also used when inferring `functional_elements`. |
| `functional_elements` | all | `None` | Explicit termination elements. If omitted, the workflow infers them as `all_symbols - substrate_elements`. |
| `surface_side` | all | `top` | Surface side used for site generation: `top` or `bottom`. |
| `surface_layer_tol` | all | `0.5` | Surface-layer clustering tolerance used for MXene site generation and `fix_z_layers`. |
| `site_match_tol` | all | `0.6` | Tolerance for site matching and hollow classification. |
| `support_xy_tol` | all | derived | Lateral support radius used for placement-height estimation. Defaults to `max(1.2, 2.5 * site_match_tol)`. |
| `termination_site_xy_tol` | all | derived | Lateral tolerance used when mapping terminations to metal reference sites. Defaults to `support_xy_tol`. |
| `termination_clearance` | all | `0.8` | Minimum clearance from `functional_elements`. |
| `min_clearance` | all | `0.9` | Generic minimum distance from the adsorbate to the slab. |
| `vertical_offset` | all | `1.8` | Initial anchor height above the site support plane. |
| `vertical_adjust_step` | all | `0.25` | Increment used by bounded vertical retry when a trial starts too close to the surface. |
| `max_vertical_adjust` | all | `1.5` | Maximum total vertical lifting applied during bounded retry. |
| `write_site_overlay` | all adsorbate workflows | `False` | Write a slab-plus-marker trajectory showing the generated site registry before sampling starts. |
| `site_overlay_include_blocked` | all adsorbate workflows | `False` | Include sites blocked by nearby terminations in the overlay output. |
| `site_overlay_z_field` | all adsorbate workflows | `suggested_z_A` | Registry z-coordinate used for marker atoms. Allowed values: `suggested_z_A`, `anchor_z_A`. |
| `bridge_cutoff` | `Ad-CMC` | `None` | Optional bridge-site cutoff used during fixed-loading initialization and registry building. |
| `top_layer_element` | `Ad-CMC` | `None` | Optional explicit top-layer element hint for legacy initialization paths. |
| `z_max_support` | `Ad-CMC` | `3.5` | Maximum support search height for local displacement proposals. |
| `detach_tol` | `Ad-CMC` | `3.0` | Distance used to detect detached adsorbates in canonical adsorbate MC. |

## 2. Canonical move controls

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `move_mode` | all | `displacement` for CMC, `hybrid` for GCMC | Canonical move family. Supported values: `displacement`, `site_hop`, `reorientation`, `puckering`, `puckering_hop`, `hybrid`. |
| `displacement_sigma` | all | `0.6` for CMC, `0.25` for GCMC | Step size for local displacement moves. |
| `max_displacement_trials` | all | `20` | Retry budget for generating a valid displacement proposal. |
| `site_hop_prob` | all | `0.5` for CMC, `0.25` for GCMC | In `move_mode: hybrid`, probability of selecting a site hop. |
| `reorientation_prob` | all | `0.2` for CMC, `0.0` for GCMC | In `move_mode: hybrid`, probability of selecting a rigid-body reorientation. |
| `puckering_prob` | all | `0.0` | In `move_mode: hybrid`, probability of selecting a coupled local surface-atom puckering move. |
| `puckering_hop_prob` | all | `0.0` | In `move_mode: hybrid`, probability of resetting the current puckered support atom to its initial local-CMC height, hopping to another site, and puckering a support atom at the target site. |
| `puckering_elements` | all | `None` | Elements eligible for puckering. `None` uses `site_elements`; for MXenes set this to the metal elements, e.g. `[Ti]` or `[Ti, Zr, Mo]`. |
| `puckering_height_A` | all | `0.15` | Maximum outward displacement relative to the initial local-CMC support-atom height for puckering proposals. |
| `max_puckering_trials` | all | `None` | Retry budget for puckering proposals. `None` uses the displacement retry budget. |
| `rotation_max_angle_deg` | all | `25.0` | Maximum absolute rotation angle used for reorientation. |
| `max_reorientation_trials` | all | `None` | Retry budget for rigid-body reorientation proposals. `None` lets the engine use its internal default. |

### Reorientation semantics

- Molecular reorientation is a rigid rotation about the anchor atom.
- The anchor atom stays fixed.
- Internal bond lengths and angles are preserved.
- For `OH`, the H rotates around the anchored O.
- For `H2O`, both H atoms rotate as a rigid body around the anchored O.

## 3. Adsorbate CMC keys

These keys live under the `cmc:` section in nested YAML.

| Key | Default | Meaning |
| --- | --- | --- |
| `initialization_mode` | `clean_surface` | Initial loading strategy. Supported values: `clean_surface`, `fixed_count`, `preloaded`. |
| `coverage` | `1.0` | Coverage used by `clean_surface` initialization in `AdsorbateCMC.from_clean_surface(...)`. |
| `n_adsorbates` | `None` | Required when `initialization_mode: fixed_count`. |
| `temperature` | `300.0` | Canonical sampling temperature in kelvin. |
| `nsweeps` | `200` | Total canonical MC sweeps. |
| `sample_interval` | `2` | Sampling interval passed into `AdsorbateCMC.run(...)`. |
| `equilibration` | `40` | Number of initial sweeps discarded for accumulated averages. |
| `write_interval` | `10` | Trajectory/thermo write interval. The older alias `interval` is also accepted. |
| `seeds` | `None` | Optional list of CMC seeds. When provided, the workflow runs one fixed-loading CMC simulation per seed and writes a `summary.csv`. |
| `backend` | `multiprocessing` | Backend used when `seeds` contains multiple entries: `multiprocessing` or `ray`. |
| `n_workers` | `1` | Number of local worker processes for multi-seed CMC when `backend: multiprocessing`. |
| `output_dir` | `None` | Optional directory prepended to `output_prefix` when `output_prefix` is relative. Useful for keeping all CMC outputs in a dedicated folder. |
| `output_prefix` | `adsorbate_cmc` | Prefix used for `.traj`, `.dat`, `.pkl`, and `_initial.traj`. |
| `seed` | `81` | Random seed for the canonical MC engine. |
| `checkpoint_interval` | `100` | Checkpoint write interval in sweeps. |
| `resume` | `False` | Resume from the checkpoint file instead of starting a new run. On resume, `nsweeps` remains the total target sweep count, not an additional sweep increment. |
| `ray_num_cpus_per_task` | `1` | CPU reservation per Ray task when multi-seed CMC uses `backend: ray`. |
| `ray_num_gpus_per_task` | `None` | GPU reservation per Ray task when multi-seed CMC uses `backend: ray`. |
| `ray_task_max_retries` | `0` | Ray task retry count for multi-seed CMC. `0` fails fast on worker crashes. |
| `ray_retry_exceptions` | `False` | Whether Ray retries tasks that raise Python exceptions. |
| `ray_get_timeout_s` | `None` | Optional timeout for waiting on Ray task results. |
| `relax` | `False` | Enable local geometry relaxation after accepted moves. |
| `relax_steps` | `20` | Maximum local relax steps. |
| `relax_z_only` | `False` | Restrict relaxation to `z` when supported by the local relaxer. |
| `verbose_relax` | `False` | Print relaxer output. |
| `fmax` | `0.05` | Force convergence threshold for local relaxation. |

### Multi-seed CMC output layout

If `seeds` is provided with more than one entry, `AdsorbateCMCWorkflow` writes one subdirectory per seed:

- `seed_067/<output_prefix>.traj`
- `seed_067/<output_prefix>.dat`
- `seed_067/<output_prefix>.pkl`
- `seed_067/<output_prefix>_initial.traj`

and a top-level `summary.csv` under `output_dir` (or under the parent directory of `output_prefix` if `output_dir` is omitted).

If `write_site_overlay: true`, `AdsorbateCMCWorkflow` also writes `<output_prefix>_sites.traj`.

## 4. Adsorbate temperature replica exchange keys

These keys live under the `pt:` section in nested YAML for `AdsorbateReplicaExchangeWorkflow`.

| Key | Default | Meaning |
| --- | --- | --- |
| `initialization_seed` | `81` | Seed used only to build the shared starting adsorbate configuration before PT begins. |
| `T_start` | `800.0` | High end of the temperature ladder in kelvin. |
| `T_end` | `50.0` | Low end of the temperature ladder in kelvin. |
| `T_step` | `50.0` | Temperature spacing when `n_replicas` is not set. |
| `n_replicas` | `None` | Optional number of replicas for a nonuniform temperature grid. |
| `fine_grid_temps` | `[]` | Optional temperatures to densify when `n_replicas` is used. |
| `fine_grid_weights` | `[]` | Relative weights for `fine_grid_temps`. |
| `fine_grid_strength` | `4.0` | Strength of the nonuniform temperature focusing. |
| `fine_grid_width` | `None` | Optional width parameter for the focused temperature grid. |
| `grid_space` | `temperature` | Grid space passed into the nonuniform temperature-grid generator. |
| `swap_interval` | `20` | Number of local CMC sweeps between exchange attempts. |
| `swap_stride` | `1` | Neighbor stride for exchange attempts. `1` means nearest-neighbor swaps. |
| `report_interval` | `5` | Per-replica trajectory/thermo write interval inside each PT chunk. |
| `sampling_interval` | `1` | Sampling interval inside each PT chunk. |
| `local_eq_fraction` | `0.2` | Fraction of each chunk used as short local re-equilibration after swaps. |
| `n_cycles` | `2` | Total number of PT cycles. Total local sweeps per replica are `n_cycles * swap_interval`. |
| `equilibration_cycles` | `0` | Number of initial PT cycles treated as equilibration. |
| `seed_nonce` | `0` | Seed offset used to derive deterministic per-temperature RNG streams. |
| `checkpoint_interval` | `10` | PT master-checkpoint interval in cycles. |
| `worker_checkpoint_interval` | `0` | Optional per-replica worker checkpoint interval in sweeps. Disabled by default because the PT master checkpoint is the authoritative restart state. In nested YAML, `cmc.checkpoint_interval` is accepted as an alias for this key. |
| `resume` | `False` | Resume PT from the master checkpoint. |
| `ray_num_cpus_per_task` | `1` | CPU reservation per Ray actor when `backend: ray`. |
| `ray_num_gpus_per_task` | `None` | GPU reservation per Ray actor when `backend: ray`. |
| `ray_actor_max_restarts` | `0` | Ray actor restart budget. `0` fails fast on actor crashes. |
| `ray_actor_max_task_retries` | `0` | Ray actor-method retry budget. |
| `ray_get_timeout_s` | `None` | Optional timeout for waiting on Ray actor results. |
| `output_dir` | `adsorbate_pt` | Output directory for PT trajectories, thermo files, checkpoints, and `results.csv`. |
| `stats_file` | `replica_stats.csv` | Swap statistics written under `output_dir`. |
| `results_file` | `results.csv` | Per-cycle replica summary written under `output_dir`. |
| `checkpoint_file` | `pt_state.pkl` | PT master checkpoint written under `output_dir`. |
| `initial_traj_file` | `adsorbate_pt_initial.traj` | Shared initialized adsorbate structure written once before PT starts. |

On resume, PT output files are truncated back to the master-checkpoint boundary
before new rows are appended.  This avoids duplicate sweep/cycle segments in
`replica_*K.dat`, `replica_*K.traj`, `results.csv`, and `replica_stats.csv`.

### Adsorbate PT output layout

`AdsorbateReplicaExchangeWorkflow` writes one set of files per temperature slot under `output_dir`, for example:

- `replica_300K.traj`
- `replica_300K.dat`
- `checkpoint_300K.pkl`
- `results.csv`
- `replica_stats.csv`
- `adsorbate_pt_initial.traj`

If `write_site_overlay: true`, the workflow also writes `output_dir/site_overlay.traj`.

The initialized structure is shared by all temperatures; each replica then evolves independently and exchanges configurations through the PT driver.

## 5. Adsorbate GCMC keys

These keys live under the `gcmc:` section in nested YAML for single-run and scan workflows.

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `temperature` | `Ad-GCMC`, `Ad-SCAN` | `300.0` | Grand-canonical sampling temperature in kelvin. |
| `chemical_potential` | `Ad-GCMC` | `0.0` | Chemical potential used by the single-run GCMC workflow. |
| `mu_values` | `Ad-SCAN` | `[-3, -2, -1, 0, 1]` | Explicit list of chemical potentials for scan or `mu`-exchange runs. |
| `mu_range` | `Ad-SCAN` | `None` | Two-element range `[mu_min, mu_max]` used to generate `mu_values`. |
| `n_mu_points` | `Ad-SCAN` | `None` | Number of points used with `mu_range`. |
| `max_n_adsorbates` | `Ad-GCMC`, `Ad-SCAN` | `None` | Maximum number of adsorbates allowed in the grand-canonical state. |
| `w_insert` | `Ad-GCMC`, `Ad-SCAN` | `1.0` | Relative weight of insertion proposals. |
| `w_delete` | `Ad-GCMC`, `Ad-SCAN` | `1.0` | Relative weight of deletion proposals. |
| `w_canonical` | `Ad-GCMC`, `Ad-SCAN` | `1.0` | Relative weight of canonical within-occupancy moves. |
| `seed` | `Ad-GCMC` | `81` | Random seed for a single GCMC run. |
| `seeds` | `Ad-SCAN` | `[81, 82]` | Independent seeds for a scan or `mu`-exchange ladder. |
| `nsweeps` | `Ad-GCMC`, `Ad-SCAN` | `200` | Total sweeps per run or per `mu` slot. |
| `sample_interval` | `Ad-GCMC`, `Ad-SCAN` | `2` | Sampling interval used by `AdsorbateGCMC.run(...)`. |
| `equilibration` | `Ad-GCMC`, `Ad-SCAN` | `40` | Number of initial sweeps discarded for accumulated averages. |
| `write_interval` | `Ad-GCMC`, `Ad-SCAN` | `10` | Write interval for trajectory and thermo output. |
| `write_attempted_traj` | `Ad-GCMC`, `Ad-SCAN` | `False` | Write a trajectory containing attempted GCMC proposals. |
| `progress_interval_moves` | `Ad-GCMC`, `Ad-SCAN` | `0` | Optional move-count progress logging interval. |
| `checkpoint_interval` | `Ad-GCMC`, `Ad-SCAN` | `100` | Checkpoint write interval in sweeps. |
| `resume` | `Ad-GCMC`, `Ad-SCAN` | `False` | Resume from `.pkl` checkpoints. On resume, `nsweeps` remains the total target sweep count for each run or `mu` slot. |
| `output_prefix` | `Ad-GCMC` | `adsorbate_gcmc` | Prefix used by the single-run workflow. |
| `output_dir` | `Ad-SCAN` | `adsorbate_gcmc_scan` | Output directory for multi-`mu` scans or `mu` exchange. |
| `ray_num_cpus_per_task` | `Ad-SCAN` | `1` | CPU reservation per Ray task or actor when `backend: ray`. |
| `ray_num_gpus_per_task` | `Ad-SCAN` | `None` | GPU reservation per Ray task or actor when `backend: ray`. |
| `ray_task_max_retries` | `Ad-SCAN` | `0` | Ray task retry count for non-exchange scan jobs. |
| `ray_retry_exceptions` | `Ad-SCAN` | `False` | Whether Ray retries scan tasks that raise Python exceptions. |
| `ray_actor_max_restarts` | `Ad-SCAN` | `0` | Ray actor restart budget for `mu`-exchange workers. |
| `ray_actor_max_task_retries` | `Ad-SCAN` | `0` | Ray actor-method retry budget for `mu`-exchange workers. |
| `ray_get_timeout_s` | `Ad-SCAN` | `None` | Optional timeout for waiting on Ray task or actor results. |

If `write_site_overlay: true`, `AdsorbateGCMCWorkflow` writes `<output_prefix>_sites.traj`, while `AdsorbateGCMCScanWorkflow` writes `output_dir/site_overlay.traj`.

## 6. `mu`-exchange keys

These keys are only used by `AdsorbateGCMCScanWorkflow` when `use_mu_exchange: true`.

| Key | Default | Meaning |
| --- | --- | --- |
| `use_mu_exchange` | `False` | Run one fixed-temperature `mu`-replica-exchange ladder per seed instead of independent `mu` windows. |
| `swap_interval` | `20` | Number of local GCMC sweeps between exchange attempts. |
| `swap_stride` | `1` | Neighbor stride for exchange attempts. `1` means nearest-neighbor `mu` swaps. |
| `exchange_report_interval` | `1` | How often the exchange driver writes swap summary rows. |
| `mu_exchange_cycles` | `None` | Optional cycle-count interface. If set, total sweeps become `mu_exchange_cycles * swap_interval` and override `nsweeps`. |
| `mu_exchange_equilibration_cycles` | `None` | Optional equilibration-cycle interface. If set, equilibration sweeps become `mu_exchange_equilibration_cycles * swap_interval` and override `equilibration`. |
| `mu_exchange_checkpoint_interval` | `10` | Exchange-driver checkpoint interval in cycles. |
| `mu_exchange_stats_file` | `mu_exchange_stats.csv` | Exchange-level statistics file written inside each seed directory. |
| `mu_exchange_results_file` | `mu_exchange_results.csv` | Per-slot summary file written inside each seed directory. |
| `mu_exchange_checkpoint_file` | `mu_exchange_state.pkl` | Driver checkpoint written inside each seed directory. |
| `mu_exchange_parallel_seeds` | `False` | Run multiple seed ladders concurrently. Currently requires `backend: ray`. |
| `mu_exchange_max_concurrent_seeds` | `None` | Optional cap on the number of concurrent seed ladders. |

### `mu`-exchange scheduling

If `use_mu_exchange: true`:

- `nsweeps` still means total local sweeps per `mu` slot unless `mu_exchange_cycles` is set.
- `swap_interval` controls the local sweeps between exchange attempts.
- `mu_exchange_cycles` is the cycle-count equivalent of `nsweeps`.

Example:

```yaml
gcmc:
  use_mu_exchange: true
  swap_interval: 20
  mu_exchange_cycles: 50
  mu_exchange_equilibration_cycles: 5
```

This gives:

- `50` exchange cycles
- `20` sweeps between swaps
- `1000` total sweeps per `mu` slot
- `100` equilibration sweeps

## 7. Hybrid MD and relaxation

These keys are available in all adsorbate workflows unless noted otherwise.

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `relax` | all | `False` | Perform local structural relaxation after accepted moves. |
| `relax_steps` | all | `10` in GCMC, `20` in CMC | Maximum local relaxation steps. |
| `fmax` | all | `0.05` | Force threshold for local relaxation. |
| `enable_hybrid_md` | all | `False` in GCMC, `False` in CMC defaults but examples often enable it | Enable short MD bursts as proposal moves. |
| `md_move_prob` | all | `0.0` in GCMC, `0.1` in CMC | Probability of attempting an MD proposal instead of an MC move. |
| `md_steps` | all | `0` in GCMC, `50` in CMC | Number of MD steps per burst. |
| `md_timestep_fs` | all | `1.0` | MD timestep in femtoseconds. |
| `md_ensemble` | all | `nve` | MD burst ensemble. `nve` is the principled choice with `md_accept_mode: hamiltonian`. |
| `md_accept_mode` | all | `potential` | Acceptance mode for MD proposals. Use `hamiltonian` with `nve` and fresh momenta for HMC-style proposals. |
| `md_friction` | all | `0.01` | Friction parameter for stochastic MD integrators. Ignored by pure `nve`. |
| `md_planar` | all | `False` | Constrain MD motion to a plane during bursts. This is global to the burst, not species-selective. |
| `md_planar_axis` | all | `2` | Axis removed when `md_planar: true` (`2` means suppress `z` momentum). |
| `md_init_momenta` | all | `True` | Draw fresh Maxwell-Boltzmann momenta for each MD burst. Required for Hamiltonian acceptance. |
| `md_remove_drift` | all | `True` | Remove center-of-mass drift before the burst. |
| `md_without_adsorbate` | `Ad-GCMC`, `Ad-SCAN` | `False` | Allow slab-only MD when the grand-canonical state is empty. |

## 8. Adsorbate-specific notes

- `AdsorbateGCMC` and `AdsorbateGCMCScanWorkflow` treat `OH`, `OOH`, `H2O`, etc. as rigid tagged molecular adsorbates for insertion/deletion and reorientation.
- Molecular adsorbates are validated against the template bond graph after local relaxation or hybrid MD. Trial states that dissociate are rejected rather than accepted into the MC trajectory.
- `support_xy_tol` and `termination_site_xy_tol` are intentionally separate:
  - `support_xy_tol` controls local height construction
  - `termination_site_xy_tol` controls termination-to-site matching on MXene surfaces
