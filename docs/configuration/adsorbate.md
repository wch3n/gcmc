# Adsorbate Workflow Configuration

This page documents the YAML keys used by:

- `AdsorbateCMCWorkflow`
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
| `bridge_cutoff` | `Ad-CMC` | `None` | Optional bridge-site cutoff used during fixed-loading initialization and registry building. |
| `top_layer_element` | `Ad-CMC` | `None` | Optional explicit top-layer element hint for legacy initialization paths. |
| `z_max_support` | `Ad-CMC` | `3.5` | Maximum support search height for local displacement proposals. |
| `detach_tol` | `Ad-CMC` | `3.0` | Distance used to detect detached adsorbates in canonical adsorbate MC. |

## 2. Canonical move controls

| Key | Workflows | Default | Meaning |
| --- | --- | --- | --- |
| `move_mode` | all | `displacement` for CMC, `hybrid` for GCMC | Canonical move family. Supported values: `displacement`, `site_hop`, `reorientation`, `hybrid`. |
| `displacement_sigma` | all | `0.6` for CMC, `0.25` for GCMC | Step size for local displacement moves. |
| `max_displacement_trials` | all | `20` | Retry budget for generating a valid displacement proposal. |
| `site_hop_prob` | all | `0.5` for CMC, `0.25` for GCMC | In `move_mode: hybrid`, probability of selecting a site hop. |
| `reorientation_prob` | all | `0.2` for CMC, `0.0` for GCMC | In `move_mode: hybrid`, probability of selecting a rigid-body reorientation. |
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
| `output_prefix` | `adsorbate_cmc` | Prefix used for `.traj`, `.dat`, `.pkl`, and `_accepted.traj`. |
| `seed` | `81` | Random seed for the canonical MC engine. |
| `checkpoint_interval` | `100` | Checkpoint write interval in sweeps. |
| `resume` | `False` | Resume from the checkpoint file instead of starting a new run. |
| `relax` | `False` | Enable local geometry relaxation after accepted moves. |
| `relax_steps` | `20` | Maximum local relax steps. |
| `relax_z_only` | `False` | Restrict relaxation to `z` when supported by the local relaxer. |
| `verbose_relax` | `False` | Print relaxer output. |
| `fmax` | `0.05` | Force convergence threshold for local relaxation. |

## 4. Adsorbate GCMC keys

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
| `resume` | `Ad-GCMC`, `Ad-SCAN` | `False` | Resume from `.pkl` checkpoints. |
| `output_prefix` | `Ad-GCMC` | `adsorbate_gcmc` | Prefix used by the single-run workflow. |
| `output_dir` | `Ad-SCAN` | `adsorbate_gcmc_scan` | Output directory for multi-`mu` scans or `mu` exchange. |

## 5. `mu`-exchange keys

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

## 6. Hybrid MD and relaxation

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

## 7. Adsorbate-specific notes

- `AdsorbateGCMC` and `AdsorbateGCMCScanWorkflow` treat `OH`, `OOH`, `H2O`, etc. as rigid tagged molecular adsorbates for insertion/deletion and reorientation.
- If a molecular adsorbate dissociates during MD, that is useful as a stability signal but it is outside the intended molecular bookkeeping model.
- `support_xy_tol` and `termination_site_xy_tol` are intentionally separate:
  - `support_xy_tol` controls local height construction
  - `termination_site_xy_tol` controls termination-to-site matching on MXene surfaces
