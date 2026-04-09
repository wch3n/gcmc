# Alloy Workflow Configuration

This page documents the YAML keys used by:

- `AlloyCMCWorkflow`
- `AlloyReplicaExchangeWorkflow`

## 1. Alloy initialization

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `site_element` | both | `None` | Sublattice element to replace when initializing an alloy from a parent structure. |
| `composition` | both | `None` | Target composition mapping used with `site_element`, e.g. `{Ti: 0.5, Mo: 0.5}`. |
| `initialization_seed` | both | `67` | Seed used by `initialize_alloy_sublattice(...)`. |

If `site_element` and `composition` are both present, the workflow initializes the alloyed sublattice before MC/PT begins.

## 2. Alloy move controls (`mc`)

These keys live under `mc:` for nested YAML in both alloy workflows.

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `swap_elements` | both | `[]` | Elements allowed to exchange. If empty, the engine uses its internal defaults. |
| `swap_mode` | both | `hybrid` | Swap proposal mode. |
| `hybrid_neighbor_prob` | both | `0.5` | In `swap_mode: hybrid`, probability of choosing a neighbor-aware swap. |
| `neighbor_cutoff` | both | `3.5` | Neighbor cutoff used by neighbor-aware swap proposals. |
| `neighbor_backend` | both | `auto` | Neighbor-list backend. |
| `neighbor_cache` | both | `True` | Cache neighbor lists between moves when possible. |
| `relax` | both | `False` | Perform relaxation after accepted alloy moves. |
| `relax_steps` | both | `10` | Maximum number of local relaxation steps. |
| `local_relax` | both | `False` | Restrict relaxation to a local region around the swap. |
| `relax_radius` | both | `4.0` | Radius used when `local_relax: true`. |
| `fmax` | both | `0.05` | Force threshold for local relaxation. |
| `enable_hybrid_md` | both | `False` | Enable short MD proposals in the alloy MC kernel. |
| `md_move_prob` | both | `0.1` | Probability of attempting an MD proposal instead of a swap move. |
| `md_steps` | both | `50` | MD steps per burst. |
| `md_timestep_fs` | both | `1.0` | MD timestep in femtoseconds. |
| `md_ensemble` | both | `nve` | MD burst ensemble. |
| `md_accept_mode` | both | `potential` | Acceptance rule for MD bursts. |
| `md_friction` | both | `0.01` | Friction used by stochastic MD integrators. |
| `md_planar` | both | `False` | Global planar MD mode. |
| `md_planar_axis` | both | `2` | Axis suppressed by planar MD. |
| `md_init_momenta` | both | `True` | Draw fresh thermal momenta before each burst. |
| `md_remove_drift` | both | `True` | Remove center-of-mass drift before each burst. |
| `checkpoint_interval` | both | `100` in `A-CMC`, `10` in `A-PT` | MC-engine checkpoint interval. |

## 3. Single-temperature alloy CMC keys

These keys are used by `AlloyCMCWorkflow`.

| Key | Default | Meaning |
| --- | --- | --- |
| `temperature` | `300.0` | Canonical MC temperature in kelvin. |
| `nsweeps` | `200` | Total number of canonical alloy sweeps. |
| `sample_interval` | `1` | Sampling interval passed to `AlloyCMC.run(...)`. |
| `equilibration` | `0` | Number of initial sweeps excluded from accumulated averages. |
| `write_interval` | `10` | Trajectory/thermo write interval. `interval` is accepted as an alias. |
| `seed` | `67` | Random seed for the single-temperature alloy MC engine. |
| `resume` | `False` | Resume from the per-run checkpoint instead of starting fresh. |
| `output_prefix` | `alloy_cmc` | Prefix used for `.traj`, `_accepted.traj`, `.dat`, and `.pkl`. |

## 4. Replica-exchange keys (`pt`)

These keys are used by `AlloyReplicaExchangeWorkflow` and live under the `pt:` section in nested YAML.

| Key | Default | Meaning |
| --- | --- | --- |
| `T_start` | `800.0` | Highest temperature in kelvin. |
| `T_end` | `50.0` | Lowest temperature in kelvin. |
| `T_step` | `50.0` | Step used when generating a uniform temperature grid. |
| `n_replicas` | `None` | Explicit replica count. Required for some nonuniform grids. |
| `grid_space` | `temperature` | Replica grid type. |
| `fine_grid_temps` | `[]` | Optional temperatures for manual grid refinement. |
| `fine_grid_weights` | `[]` | Optional weights for manual grid refinement. |
| `fine_grid_strength` | `4.0` | Controls local densification around `fine_grid_temps`. |
| `fine_grid_width` | `None` | Optional explicit width for refined temperature bands. |
| `n_cycles` | `2` | Total number of replica-exchange cycles. |
| `equilibration_cycles` | `0` | Number of initial PT cycles excluded from averages. |
| `local_eq_fraction` | `0.2` | Fraction of each cycle spent on local equilibration before reporting. |
| `swap_interval` | `20` | Local MC sweeps between swap attempts. |
| `swap_stride` | `1` | Neighbor stride for swap attempts. |
| `report_interval` | `5` | Cycle interval for detailed PT reporting. |
| `sampling_interval` | `1` | Sampling interval within each local MC block. |
| `checkpoint_interval` | `10` | Driver checkpoint interval in PT cycles. |
| `checkpoint_file` | `pt_state.pkl` | Replica-exchange driver checkpoint. |
| `results_file` | `results.csv` | Per-cycle summary output. |
| `stats_file` | `replica_stats.csv` | Pair-swap statistics output. |
| `output_dir` | `alloy_pt` | Output directory for the PT workflow. |
| `resume` | `False` | Resume from `checkpoint_file`. |
| `seed_nonce` | `0` | Extra seed offset used to decorrelate otherwise identical PT runs. |
| `track_composition` | `[]` | Optional species list for composition tracking during PT. |

## 5. Replica-exchange backend keys

These keys work together with the shared backend options documented in `docs/configuration/shared.md`.

| Key | Default | Meaning |
| --- | --- | --- |
| `backend` | `multiprocessing` | Execution backend for replica exchange: `multiprocessing` or `ray`. |
| `n_gpus` | `None` | Total GPU count for allocating replica workers. |
| `workers_per_gpu` | `None` | Number of replica workers per GPU. |
| `ray_num_cpus_per_task` | `1` | CPU reservation per Ray actor. |
| `ray_num_gpus_per_task` | `None` | GPU reservation per Ray actor. |
| `use_placement_group` | `False` | Whether to group Ray actors into a placement group. |
| `placement_group_strategy` | `SPREAD` | Placement strategy for Ray placement groups. |
| `remove_placement_group_on_stop` | `True` | Remove placement group on workflow shutdown. |
| `shutdown_on_stop` | `False` | Shut down the Ray runtime when the workflow exits. |

## 6. Notes

- The alloy PT workflow consumes both `pt:` and `mc:` sections. `pt:` controls the exchange driver; `mc:` controls the local `AlloyCMC` kernel.
- `swap_interval` is defined in PT cycles, not wall-clock time.
- Resume safety depends on keeping the same replica grid and MC kernel settings between runs.
- For multi-node Ray jobs, keep the Slurm bootstrap in `scripts/slurm/ray_slurm_common.sh` and put job-specific resource settings in the local `run.slurm`.
