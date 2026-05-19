# Reaction Post-Processing Workflow

The reaction post-processing workflow turns MC/PT adsorbate trajectories into
site-resolved parent adsorbate ensembles. It is intended as the reusable first
stage for parent-conditioned reaction analysis, including OER.

Current implemented stages:

- identify populated parent adsorbate sites from one or more trajectories;
- aggregate equivalent sites across temperatures or replicas using
  `site_type + sorted support_indices`;
- write representative structures for the top parent sites;
- generate parent-conditioned clean `*`, `OH*`, `O*`, and `OOH*` candidate structures;
- optionally relax the generated state candidates with an ASE calculator;
- optionally evaluate local harmonic adsorbate/slab corrections with ASE vibrations for every relaxed candidate;
- optionally relax gas-phase H2/H2O references and evaluate their IdealGasThermo corrections;
- optionally evaluate site-conditioned OER free-energy changes with CHE references.

## Runner

```bash
gcmc-reaction-postprocess --config reaction_postprocess.yaml
```

## Layout

```yaml
parent:
  traj: [replica_295K.traj, replica_303K.traj]
  start: 900
  stop: null
  step: 1

site_analysis:
  site_elements: [Ti, Zr, Mo]
  substrate_elements: [Ti, Zr, Mo, C]
  functional_elements: [O]
  site_types: [atop, fcc, hcp]
  surface_side: top
  anchor_element: O

selection:
  representative_top_k: 10
  representative_n_per_group: 1

calculator:
  calculator: symmetrix
  model_file: model.json

output:
  output_dir: reaction_postprocess
  out_prefix: parent_sites
  write_site_directories: true
  sites_dir: sites
  reactions_dir: reactions
  reaction_state_dirs: [00_clean, 01_OH, 02_O, 03_OOH]

candidate_generation:
  enabled: true
  include_original_o_site: true
  include_nearby_o_sites: true
  nearby_site_radius_A: 3.0
  max_nearby_sites: 8
  ooh_orientations: 8

state_relaxation:
  enabled: false
  states: [00_clean, 01_OH, 02_O, 03_OOH]
  fmax: 0.05
  steps: 300
  progress_log: state_relaxation.log
  progress_stdout: true
  max_candidates_per_state: null
  skip_existing: true
  enforce_adsorbate_integrity: true
  adsorbate_bond_detection_factor: 1.2
  adsorbate_bond_stretch_factor: 1.35
  adsorbate_bond_abs_tol_A: 0.35

vibrations:
  enabled: false
  temperature_K: 303.0
  slab_cutoff_A: 3.5
  delta_A: 0.01
  nfree: 2
  ignore_imag_modes: false
  states: [clean, oh, o, ooh]
  skip_existing: true
  progress_log: vibration.log
  progress_stdout: true

reference_thermo:
  enabled: false
  molecules: [H2, H2O, O2]
  output_dir: references
  skip_existing: true
  overwrite_che_references: false
  vacuum_A: 8.0
  temperature_K: 303.0
  pressure_Pa: 101325.0
  h2:
    energy_eV: null
    correction_eV: 0.0
    pressure_Pa: 101325.0
  h2o:
    energy_eV: null
    correction_eV: 0.0
    pressure_Pa: 101325.0
  o2:
    energy_eV: null
    correction_eV: 0.0
    pressure_Pa: 101325.0
    geometry: linear
    symmetrynumber: 2
    spin: 1.0
  relaxation:
    enabled: true
    fmax: 0.001
    steps: 1000
    log_file: relax.log
    progress_log: reference_thermo.log
    progress_stdout: true
  vibrations:
    delta_A: 0.01
    nfree: 2
    ignore_imag_modes: false

che:
  enabled: false
  clean_reference_mode: lowest_energy
  use_vibrational_free_energies: true
  oer_reference_mode: auto
  total_oer_free_energy_eV: 4.92
  equilibrium_potential_V: 1.23
  boltzmann_weight_states: true
  boltzmann_temperature_K: 303.0
  boltzmann_energy_cluster_tol_eV: 0.01
```

## Sections

### `parent`

| Key | Meaning |
| --- | --- |
| `traj` | One or more tagged adsorbate trajectories. |
| `reference` | Optional clean/reference slab. If omitted, tagged adsorbates are stripped from the first analyzed frame. |
| `start`, `stop`, `step` | Frame window used as the production sample. |

### `site_analysis`

These keys are passed to `LocalAdsorptionMotifAnalyzer`.

| Key | Meaning |
| --- | --- |
| `site_elements` | Surface metal elements used to build adsorption sites. |
| `substrate_elements` | Slab/substrate elements used when stripping adsorbates and terminations. |
| `functional_elements` | Termination/surface functional elements. |
| `site_types` | Site registry types, e.g. `atop`, `fcc`, `hcp`. |
| `surface_side` | `top` or `bottom`. |
| `anchor_element` | Adsorbate atom used as the parent-site anchor, e.g. `O` for `OH*`. |
| `shell1_size`, `shell2_size` | Number of local metal atoms used in shell fingerprints. |
| `functional_cutoff` | Cutoff for nearby termination counts. |
| `surface_layer_tol`, `site_match_tol`, `support_xy_tol`, `termination_site_xy_tol`, `vertical_offset`, `termination_clearance` | Site-registry geometry controls. |

### `selection`

| Key | Meaning |
| --- | --- |
| `aggregate_group_by` | Currently use `site`; equivalent sites are grouped by `site_type + sorted support_indices`. |
| `representative_top_k` | Number of aggregate parent sites to write representatives for. |
| `representative_n_per_group` | Number of representative frames per selected parent site. |

### `calculator`

The same calculator names as adsorbate MC workflows are accepted:

- `lj`
- `mace`
- `symmetrix`

For production OER screening, use the same MLIP model as the MC/PT runs.

### `output`

| Key | Meaning |
| --- | --- |
| `output_dir` | Directory for CSVs and representative trajectories. |
| `out_prefix` | Prefix used by per-trajectory analysis exports. Top-level workflow files use fixed short names such as `summary.csv` and `oer_routes.csv`. |
| `write_per_trajectory` | Also write per-trajectory motif CSVs. |
| `write_representatives` | Also write per-trajectory representative frames. |
| `write_site_directories` | Write stable per-parent-site directories for downstream reactions. |
| `sites_dir` | Directory under `output_dir` containing per-site folders. |
| `reactions_dir` | Subdirectory under each site reserved for child reaction stages. |
| `reaction_state_dirs` | State directories created under `reactions_dir`; use `[00_clean, 01_OH, 02_O, 03_OOH]` for OER CHE. |

### `candidate_generation`

This optional stage uses the parent representatives to generate structures that
can be relaxed in each OER state directory. `00_clean` is generated by removing
the tagged parent adsorbate from the representative OH* frame.

| Key | Meaning |
| --- | --- |
| `enabled` | If true, write `candidates.traj` and `candidates.csv` under each reaction state. |
| `include_original_o_site` | Generate direct O* by removing H from parent OH*. |
| `include_nearby_o_sites` | Generate O* candidates on nearby adsorption sites around the parent OH* anchor. |
| `nearby_site_radius_A` | Lateral radius used to collect nearby shifted O* basins. |
| `max_nearby_sites` | Maximum nearby O* sites per parent representative. |
| `ooh_orientations` | Number of OOH* orientations generated from each O* candidate. |
| `oo_bond_A`, `terminal_oh_bond_A` | Initial O-O and terminal O-H distances for OOH* candidates. |

### `state_relaxation`

This optional stage relaxes the generated `candidates.traj` files inside each
reaction state directory.

| Key | Meaning |
| --- | --- |
| `enabled` | If true, relax generated state candidates and write `relaxed.traj` plus `energies.csv`. |
| `states` | Optional state filter. Accepts state directory names such as `02_O` or species names such as `O`. Empty means all states. |
| `fmax` | Force threshold in eV/A. |
| `steps` | Maximum LBFGS optimizer steps per candidate. |
| `log_file` | Optional ASE optimizer log file name or path. A bare file name is written per candidate in the state directory. |
| `progress_log` | Progress log path. If omitted, writes `state_relaxation.log` under `output_dir`; set to `false` to disable the file. |
| `progress_stdout` | If true, also print progress messages to the terminal or Slurm output. |
| `max_candidates_per_state` | Optional cap for testing or staged screening before relaxing every generated candidate. |
| `skip_existing` | If true, keep existing `energies.csv`/`relaxed.traj` outputs. |
| `enforce_adsorbate_integrity` | If true, molecular adsorbates such as OH* and OOH* must retain their initial adsorbate bond graph after relaxation. Failed candidates are marked non-converged and are excluded from CHE/vibrations. |
| `adsorbate_bond_detection_factor` | Covalent-radius factor used to detect the initial molecular adsorbate bonds. |
| `adsorbate_bond_stretch_factor`, `adsorbate_bond_abs_tol_A` | Maximum allowed relaxed bond distance is `max(initial_distance * stretch_factor, initial_distance + abs_tol)`. |
| `calculator`, `model`, `model_file`, `device`, `use_kokkos` | Optional calculator overrides. If omitted, the top-level `calculator` section is used. |

### `vibrations`

This optional stage runs ASE finite-difference vibrations on every relaxed
candidate listed in `state_relaxation_manifest.csv` / `energies.csv`. For each
parent site and parent representative, the local slab mask is defined from the
corresponding relaxed clean candidate using the parent support atoms and a
distance cutoff, then the exact same slab atoms are reused for the adsorbate
candidates from that same parent representative. Candidate atom-origin metadata
is written during candidate generation so the selected slab atoms can be
matched exactly across `00_clean`, `01_OH`, `02_O`, and `03_OOH`.

| Key | Meaning |
| --- | --- |
| `enabled` | If true, write `vibration_summary.csv` with one row per relaxed candidate. |
| `temperature_K` | Temperature passed to `HarmonicThermo`. |
| `slab_cutoff_A` | Radius around the clean-state parent support site used to define the reusable local slab mask. |
| `delta_A` | Finite-difference displacement for ASE `Vibrations`. |
| `nfree` | Number of finite-difference points (`2` or `4`). |
| `ignore_imag_modes` | Passed to `HarmonicThermo`; keep `false` to fail on non-minima, or set `true` to ignore residual soft imaginary modes. |
| `states` | Optional label filter from `clean`, `oh`, `o`, `ooh`. |
| `skip_existing` | If true, reuse an existing `reactions/<state>/vibrations/<candidate_id>/result.csv`. |
| `progress_log` | Progress log path. If omitted, writes `vibration.log` under `output_dir`; set to `false` to disable the file. |
| `progress_stdout` | If true, also print vibration progress messages to the terminal or Slurm output. |
| `calculator`, `model`, `model_file`, `device`, `use_kokkos` | Optional calculator overrides. If omitted, the top-level `calculator` section is used. |

### `reference_thermo`

This optional stage runs before CHE. It relaxes isolated gas-phase references
and runs ASE `Vibrations` + `IdealGasThermo`, then injects the resulting
molecular reference free-energy inputs into the active CHE config. Put the DFT
electronic reference energies in the molecule-specific blocks as
`energy_eV`; the vibrational/ideal-gas corrections are still computed from the
relaxed molecule. If `energy_eV` is omitted, the relaxed molecule energy from
the configured calculator is used instead.

| Key | Meaning |
| --- | --- |
| `enabled` | If true, run molecular reference thermo before CHE. |
| `molecules` | Molecules to evaluate. Use `[H2, H2O]` for CHE closure mode, or `[H2, H2O, O2]` when `che.oer_reference_mode: explicit_o2` is desired. |
| `output_dir` | Directory under `output_dir` for molecular outputs, or an absolute path. |
| `skip_existing` | If true, reuse existing `summary.yaml` and `che_snippet.yaml` for a molecule. If a molecule-specific `energy_eV` no longer matches the cached summary, the molecule is recomputed. |
| `overwrite_che_references` | Legacy escape hatch. Direct CHE reference entries are normally omitted; molecule-specific `energy_eV` or `correction_eV` values in `reference_thermo` always take precedence. |
| `vacuum_A` | Vacuum added around built-in ASE molecule geometries. |
| `temperature_K`, `pressure_Pa` | Default gas-phase thermodynamic state. If `temperature_K` is omitted, the stage uses `che.boltzmann_temperature_K`, then `303.0` K. |
| `relaxation` | LBFGS settings for each isolated molecule. |
| `vibrations` | ASE `Vibrations` settings for each isolated molecule. |
| `calculator`, `h2`, `h2o`, `o2` | Optional overrides. Use `calculator` for a different model/device; use molecule-specific blocks for `energy_eV`, `correction_eV`, custom `atoms`, `geometry`, `symmetrynumber`, `spin`, or pressure. |

### `che`

This optional stage reads the relaxed `00_clean`, `01_OH`, `02_O`, and
`03_OOH` energies for each parent site and computes the standard four OER
CHE free-energy changes. Molecular electronic energies and gas-phase thermo
settings should be supplied under `reference_thermo`. `O2` is optional: by
default the final OER step is closed with the experimental total OER free
energy, but an explicit O2 reference can be supplied through
`reference_thermo.o2.energy_eV` when desired.

For a fixed alloy slab, keep the per-site `00_clean` directories as diagnostics,
but use `clean_reference_mode: lowest_energy` for CHE so every site references
the same lowest-energy converged clean slab. Use `per_site` only when the clean
state is intentionally site-conditioned.

| Key | Meaning |
| --- | --- |
| `enabled` | If true, write site-conditioned OER CHE summary CSVs. |
| `clean_reference_mode` | `per_site` uses each site's own `00_clean`; `lowest_energy` uses the lowest-energy converged `00_clean` across all sites as a shared `G(*)` reference. |
| `use_vibrational_free_energies` | If true, CHE uses `harmonic_free_energy_eV` from `vibration_summary.csv` when a ready row exists for that relaxed candidate; otherwise it falls back to the electronic relaxed energy. |
| `vibration_summary_csv` | Optional override path for the vibration summary. If omitted, CHE looks for `vibration_summary.csv` under `output_dir`. |
| `h2_energy_eV`, `h2o_energy_eV`, `o2_energy_eV`, `h2_thermo`, `h2o_thermo`, `o2_thermo` | Internal/legacy fields. Prefer `reference_thermo.h2.energy_eV`, `reference_thermo.h2o.energy_eV`, and `reference_thermo.o2.energy_eV`; the reference-thermo stage fills these CHE fields automatically. |
| `h2_correction_eV`, `h2o_correction_eV`, `o2_correction_eV` | Internal/legacy fields. Prefer molecule-specific `reference_thermo.<molecule>.correction_eV` when an extra molecular offset is needed. |
| `clean_correction_eV`, `oh_correction_eV`, `o_correction_eV`, `ooh_correction_eV` | Optional extra offsets on top of the slab/adsorbate free energies. Leave omitted in normal use when `use_vibrational_free_energies: true`. |
| `oer_reference_mode` | `auto` uses explicit O2 when `o2_energy_eV` is provided, otherwise closure mode. `closure` always uses `total_oer_free_energy_eV`; `explicit_o2` requires `o2_energy_eV`. |
| `total_oer_free_energy_eV` | Total four-step OER Gibbs free energy used in closure mode to avoid explicit O2; default `4.92` eV. In closure mode `DeltaG4` is computed from the already selected `DeltaG1-3`, so when `use_vibrational_free_energies: true` the closed cycle is a Gibbs free-energy cycle, not an electronic-energy cycle. |
| `equilibrium_potential_V` | Equilibrium potential used for overpotential; default `1.23` V. |
| `potential_V` | Optional applied potential; reported as shifted step free energies. |
| `use_converged_only` | If true, choose only converged relaxed candidates for each state. |
| `allow_unconverged_fallback` | If true, fall back to the lowest finite-energy unconverged candidate when no converged candidate exists. |
| `boltzmann_weight_states` | If true, also fill Boltzmann route columns in `oer_routes.csv` where OH/O/OOH state energies are finite-temperature log-sum-exp free energies over relaxed candidates. |
| `boltzmann_temperature_K` | Temperature used for Boltzmann weighting. Use the target OER analysis temperature, e.g. `303.0`. |
| `boltzmann_energy_cluster_tol_eV` | Optional energy clustering tolerance before Boltzmann weighting. Use a small value, e.g. `0.01`, to reduce artificial degeneracy from trial orientations that relax to the same basin. |

Aggregate outputs:

- `summary.csv`
- `representatives.csv`
- `representatives.traj`
- `site_manifest.csv`

Structured per-site outputs:

```text
<output_dir>/
  summary.csv
  representatives.csv
  representatives.traj
  site_manifest.csv
  sites/
    fcc_320-380-385/
      site.yaml
      representatives.csv
      representatives.traj
      reactions/
        README.md
        00_clean/
          README.md
          candidates.csv
          candidates.traj
          energies.csv
          relaxed.traj
          vibrations/
            <candidate_id>/
              result.csv
              vib.*
        01_OH/
          README.md
          candidates.csv
          candidates.traj
          energies.csv
          relaxed.traj
          vibrations/
            <candidate_id>/
              result.csv
              vib.*
        02_O/
          README.md
          candidates.csv
          candidates.traj
          energies.csv
          relaxed.traj
          vibrations/
            <candidate_id>/
              result.csv
              vib.*
        03_OOH/
          README.md
          candidates.csv
          candidates.traj
          energies.csv
          relaxed.traj
          vibrations/
            <candidate_id>/
              result.csv
              vib.*
```

The site directory name is derived from the canonical parent `site_id`
(`site_type + sorted support_indices`; for example `fcc:320-380-385` becomes
`fcc_320-380-385`), not from the transient local registry index. This keeps
downstream OER calculations tied to the physical support atoms selected from
the parent OH* ensemble.

The reaction directories are state directories, not arrow directories:

- `00_clean/` stores the clean parent-conditioned slab after removing the parent OH* adsorbate.
- `01_OH/` stores relaxation of the selected parent OH* representative(s).
- `02_O/` stores O* candidates derived from that parent site. Include nearby
  or site-shifted O* basins if deprotonation relaxes away from the original
  OH* anchor.
- `03_OOH/` stores OOH* candidates derived from the parent-conditioned O*
  ensemble. Include nearby or site-shifted OOH* basins when relevant.

Candidate generation also writes:

- `candidate_manifest.csv`
- `reactions/<state>/candidates.csv`
- `reactions/<state>/candidates.traj`

If state relaxation is enabled:

- `state_relaxation.log`
- `state_relaxation_manifest.csv`
- `reactions/<state>/energies.csv`
- `reactions/<state>/relaxed.traj`

If harmonic vibrations are enabled:

- `vibration.log`
- `vibration_summary.csv`
- `reactions/<state>/vibrations/<candidate_id>/result.csv`

If molecular reference thermo is enabled:

- `references/h2/summary.yaml`
- `references/h2/che_snippet.yaml`
- `references/h2o/summary.yaml`
- `references/h2o/che_snippet.yaml`

During a long state relaxation, `energies.csv`, `relaxed.traj`, and the progress
log are updated after each candidate, so an interrupted run can be inspected
without waiting for the whole state block to finish.

If CHE is enabled:

- `oer_routes.csv`
- `oer_states.csv`
- `oer_ensemble.csv` when `boltzmann_weight_states: true`

`oer_routes.csv` is the primary site-resolved OER table. It has one row per
parent site and reports the single-minimum route columns
(`DeltaG*_min_eV`, `overpotential_min_V`) side-by-side with the Boltzmann
route columns (`DeltaG*_boltzmann_eV`, `overpotential_boltzmann_V`) when
Boltzmann weighting is enabled.
It also writes electronic-only route columns without vibrational/ideal-gas
thermo contributions, such as `DeltaG*_min_electronic_eV` and
`DeltaG*_boltzmann_electronic_eV`, so the with- and without-vibration profiles
can be compared without reconstructing them from lower-level files.

`oer_states.csv` is the compact provenance table. It has one row per
`site_id + state_label` and records the selected minimum candidate, effective
state free energy, energy source (`electronic` or `harmonic`), convergence
metadata, source `energies.csv`, and the Boltzmann state free energy/count
when enabled.
The state table also records electronic-only state free energies
(`min_electronic_free_energy_eV` and
`boltzmann_electronic_free_energy_eV`) for provenance.

When `use_vibrational_free_energies: true`, the state energies in these OER
tables are the harmonic free energies for the relaxed candidates whenever a
ready `vibration_summary.csv` row exists.

`oer_ensemble.csv` is the single top-level ensemble descriptor. It uses the
site-resolved `DeltaG1_boltzmann_eV` values from `oer_routes.csv` to build a
dilute-limit OH adsorption weighting,
`w_p ∝ exp[-DeltaG1,p / (k_B T)]`, with uniform site degeneracy, and reports the
route-weighted mean of the site-specific overpotentials rather than the
overpotential of an averaged free-energy profile.

The CHE expressions are:

```text
DeltaG1 = G(OH*)  - G(*)   - G(H2O) + 0.5 G(H2)
DeltaG2 = G(O*)   - G(OH*)          + 0.5 G(H2)
DeltaG3 = G(OOH*) - G(O*)  - G(H2O) + 0.5 G(H2)
DeltaG4 = total_oer_free_energy_eV - DeltaG1 - DeltaG2 - DeltaG3
eta = max(DeltaG1, DeltaG2, DeltaG3, DeltaG4) - 1.23 V
```

Here `DeltaG1-3` use the active state/reference energy model. With
`use_vibrational_free_energies: true`, they are harmonic Gibbs free-energy
changes, and the closure expression forces the Gibbs free-energy sum to
`total_oer_free_energy_eV`.

With explicit O2 enabled, the last step is instead:

```text
DeltaG4 = G(*) + G(O2) - G(OOH*) + 0.5 G(H2)
```
