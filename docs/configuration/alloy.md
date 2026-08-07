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
| `checkpoint_interval` | `A-CMC` | `100` | Single-temperature MC-engine checkpoint interval in sweeps. In `A-PT`, use this only under `mc:` when explicit per-worker checkpoints are needed. |
| `worker_checkpoint_interval` | `A-PT` | `0` | Optional per-replica worker checkpoint interval in sweeps. Disabled by default because the PT master checkpoint is the authoritative restart state. In nested YAML, `mc.checkpoint_interval` is accepted as an alias for this key. |

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
| `resume` | `False` | Resume from the per-run checkpoint instead of starting fresh. On resume, `nsweeps` remains the total target sweep count. |
| `output_dir` | `None` | Optional directory prepended to `output_prefix` when `output_prefix` is relative. Useful for keeping alloy CMC outputs in a dedicated `results/` folder. |
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
| `checkpoint_interval` | `10` | Driver checkpoint interval in PT cycles. This is independent of `mc.worker_checkpoint_interval` / `mc.checkpoint_interval`. |
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
| `ray_actor_max_restarts` | `0` | Ray actor restart budget. `0` fails fast on actor crashes. |
| `ray_actor_max_task_retries` | `0` | Ray actor-method retry budget. |
| `ray_get_timeout_s` | `None` | Optional timeout for waiting on Ray actor results. |
| `use_placement_group` | `False` | Whether to group Ray actors into a placement group. |
| `placement_group_strategy` | `SPREAD` | Placement strategy for Ray placement groups. |
| `remove_placement_group_on_stop` | `True` | Remove placement group on workflow shutdown. |
| `shutdown_on_stop` | `False` | Shut down the Ray runtime when the workflow exits. |

## 6. Notes

- The alloy PT workflow consumes both `pt:` and `mc:` sections. `pt:` controls the exchange driver; `mc:` controls the local `AlloyCMC` kernel.
- `swap_interval` is defined in PT cycles, not wall-clock time.
- On resume, PT output files are truncated back to the master-checkpoint boundary before new rows are appended, preventing duplicate sweep/cycle segments in `replica_*K.dat`, `replica_*K.traj`, `results.csv`, and `replica_stats.csv`.
- Resume safety depends on keeping the same replica grid and MC kernel settings between runs.
- For multi-node Ray jobs, keep the Slurm bootstrap in `scripts/slurm/ray_slurm_common.sh` and put job-specific resource settings in the local `run.slurm`.

## 7. SRO and layer-LRO analysis

Use the packaged analysis command to summarize all fixed-temperature PT
trajectories below a run directory:

```bash
gcmc-analyze-alloy-ordering \
  --run-dir /path/to/alloy/run \
  --start-fraction 0.5 \
  --step 10 \
  --n-blocks 5
```

The command infers the alloy elements and output directory from `config.yaml`.
It builds fixed reference-lattice neighbor lists once, inferring the first
intralayer and nearest-interlayer coordinations from gaps in the ranked metal
distances. The two SRO curves are therefore not mixed by thermal distortions or
a shared distance cutoff. Use `--intralayer-coordination` and
`--interlayer-coordination` to override the inferred values.

The command writes a two-panel PDF/PNG, a temperature summary CSV, and the
underlying block means. The left panel reports separate intralayer-1NN and
nearest-interlayer unlike-pair Warren-Cowley parameters. Its dashed random
reference is determined automatically for the fixed-composition finite cell as
`-1 / (N_alloy - 1)` and is also recorded in the summary CSV. Values below that
line indicate enhanced unlike-neighbor preference; values above it indicate
enhanced like-neighbor preference. The right panel reports the two-metal-layer
polarization `|x_A(top) - x_A(bottom)|`, which is a long-range order parameter
for layer segregation. It shows zero for identical layer compositions and an
automatically determined finite-cell random expectation as a dashed line. The
latter is calculated exactly from the reference layer sizes, global target-
element count, and canonical combinatorial distribution; it is also included
in the summary CSV. A general in-plane LRO parameter requires an explicitly
chosen ordered phase or ordering wavevector.
