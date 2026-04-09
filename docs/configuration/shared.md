# Shared Configuration Keys

This page covers keys that are reused across multiple workflow types.

Workflow abbreviations:

- `A-CMC`: `AlloyCMCWorkflow`
- `A-PT`: `AlloyReplicaExchangeWorkflow`
- `Ad-CMC`: `AdsorbateCMCWorkflow`
- `Ad-GCMC`: `AdsorbateGCMCWorkflow`
- `Ad-SCAN`: `AdsorbateGCMCScanWorkflow`

## `system`

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `snapshot` | all | `None` | Input structure file. Required in practice. |
| `frame` | all | `0` | Frame index used for trajectory-like inputs. For plain structure files the loader uses frame `0`. |
| `repeat` | all | `[1, 1, 1]` | ASE repeat vector applied after loading the snapshot. |
| `supercell_matrix` | all | `None` | Integer supercell matrix applied before `repeat`. |
| `fix_below_z` | all | `None` | Applies `FixAtoms` to all atoms with `z < fix_below_z`. This freezes all Cartesian components. |
| `fix_z_elements` | all | `[]` | Element list for selective `z`-only constraints. Works together with `fix_z_layers`. |
| `fix_z_layers` | all | `None` | Layer-index selector for `z`-only constraints, e.g. `{top: [1], bottom: [1]}` or `{bottom: all}`. |

### `fix_z_layers` syntax

`fix_z_layers` is interpreted by physical layer number from each surface:

```yaml
system:
  fix_z_elements: [Ti, Zr, Mo]
  fix_z_layers:
    top: [1]
    bottom: [1, 2]
```

- Layer counting starts at `1`.
- `all` means every detected layer on that side.
- Layer detection uses the same surface-layer clustering logic as the workflow, with `surface_layer_tol`.

## `calculator`

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `calculator` | all | `lj` | Calculator backend. Supported in workflows: `lj`, `mace`, `symmetrix`. |
| `lj_cutoff` | all | `6.0` | Lennard-Jones cutoff used when `calculator: lj`. |
| `model` | all | `None` | Alternative model path. Used mainly by `mace`; also accepted as a fallback for `symmetrix`. |
| `model_file` | all | `None` | Primary model file for `mace` or `symmetrix`. |
| `device` | `A-CMC`, `A-PT`, `Ad-GCMC`, `Ad-CMC` | `cuda` | Device string passed to `mace` or used when constructing single-run calculators. |
| `default_dtype` | `A-PT` | `None` | Optional dtype override passed to `MACECalculator`. |
| `use_kokkos` | all | `True` | Passed to `symmetrix` calculators. Ignored by `lj` and `mace`. |

## `backend`

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `backend` | `A-PT`, `Ad-SCAN` | `multiprocessing` | Execution backend. Supported values depend on workflow: `multiprocessing` or `ray`. |
| `n_workers` | `Ad-SCAN` | `1` | Number of local workers for multiprocessing scans. |
| `devices` | `Ad-SCAN` | `['cuda:0']` | Device strings assigned round-robin in multiprocessing scans. |
| `gpu_ids` | `Ad-SCAN` | `[]` | CUDA-visible GPU ids used by multiprocessing scans when `devices` is not provided. |
| `n_gpus` | `A-PT` | `None` | Total GPU count for alloy replica exchange. Required for `ray` unless it can be inferred from the environment. |
| `workers_per_gpu` | `A-PT` | `None` | Replica workers per GPU for alloy replica exchange. Derived automatically for Ray if omitted. |
| `ray_address` | `A-PT`, `Ad-SCAN` | `None` | Ray cluster address. Use `auto` or rely on `RAY_ADDRESS` for Slurm-managed clusters. |
| `ray_log_to_driver` | `A-PT`, `Ad-SCAN` | `False` | Whether Ray worker logs are forwarded to the driver. |
| `ray_num_cpus_per_task` | `A-PT`, `Ad-SCAN` | `1` | CPU reservation per Ray task/actor. |
| `ray_num_gpus_per_task` | `A-PT`, `Ad-SCAN` | `None` | GPU reservation per Ray task/actor. If omitted in adsorbate scans, the workflow uses `1.0` for `mace` and `0.0` otherwise. |
| `use_placement_group` | `A-PT` | `False` | Enables Ray placement groups for alloy replica exchange. |
| `placement_group_strategy` | `A-PT` | `SPREAD` | Placement group strategy passed to Ray. |
| `remove_placement_group_on_stop` | `A-PT` | `True` | Remove placement group at workflow shutdown. |
| `shutdown_on_stop` | `A-PT` | `False` | Whether the workflow shuts down the Ray runtime when stopping. |

## `output`

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `output_prefix` | `A-CMC`, `Ad-CMC`, `Ad-GCMC` | workflow-specific | Prefix used to derive `.traj`, `.dat`, `.pkl`, and accepted/attempted trajectory paths. |
| `output_dir` | `A-PT`, `Ad-SCAN` | workflow-specific | Output directory used for replica exchange or scan outputs. |
| `checkpoint_file` | `A-PT` | `pt_state.pkl` | Top-level checkpoint file for alloy replica exchange. |
| `stats_file` | `A-PT` | `replica_stats.csv` | Swap-level replica exchange statistics file. |
| `results_file` | `A-PT` | `results.csv` | Per-cycle replica summary written by alloy replica exchange. |

## Notes

- Shared path fields are resolved relative to the YAML file:
  - `snapshot`
  - `model`
  - `model_file`
  - `output_dir`
- `output_prefix` is not rewritten; it is used exactly as provided.
