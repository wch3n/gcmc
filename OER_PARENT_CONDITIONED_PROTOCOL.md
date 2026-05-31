# Parent-Conditioned OER Ensemble Protocol

This note describes the recommended protocol for using the current MC/PT
adsorbate calculations to build an OER free-energy analysis on HE-MXenes.

The central idea is:

- sample `OH*` first, because it is the first adsorbed OER intermediate and
  defines the populated parent active-site ensemble;
- for each important `OH*` parent basin, generate local `O*` and `OOH*`
  child basins rather than independently choosing their clean-surface global
  minima;
- compute OER free-energy changes from these parent-conditioned ensembles.

This keeps the sequential nature of OER while retaining configurational
entropy from multiple accessible basins.

## 1. Physical Problem

The oxygen evolution reaction (OER) is commonly represented by four
proton-coupled electron-transfer steps:

```text
* + H2O      -> OH*  + H+ + e-
OH*         -> O*   + H+ + e-
O* + H2O    -> OOH* + H+ + e-
OOH*        -> *    + O2 + H+ + e-
```

The overpotential is obtained from the largest uphill free-energy step:

```text
eta = max(DeltaG1, DeltaG2, DeltaG3, DeltaG4) / e - 1.23 V
```

On a high-entropy MXene, there is not one unique active site. Different local
metal motifs and neighboring terminations can stabilize different
intermediates. Therefore, the key problem is to evaluate a connected OER
free-energy profile over a relevant ensemble of local basins, not just one
relaxed structure.

## 2. Why Parent-Conditioned Sampling

Independent global scans of `OH*`, `O*`, and `OOH*` answer this question:

```text
Where would each isolated intermediate prefer to adsorb on the surface?
```

That is useful as a diagnostic, but it is not the exact OER question. The OER
sequence asks:

```text
Given that OH* is already formed at parent basin p,
which O* and OOH* basins are locally accessible from that state?
```

The parent-conditioned protocol uses `OH*` as the parent state and constructs
`O*` and `OOH*` locally from each relevant `OH*` basin. Independent `O*` and
`OOH*` PT runs are still useful, but mainly as proposal generators for likely
nearby child basins.

## 3. Definitions

### Parent Basin

A parent basin `p` is an occupied `OH*` site/motif extracted from the `OH*`
finite-temperature trajectory. It should include:

- the adsorbate geometry;
- the support site assignment;
- the local metal motif;
- the surrounding termination/background structure.

In the current code, parent basins can be identified with
`analyze_local_adsorption_motifs.py` or the reusable
`ReactionPostProcessingWorkflow`.

### Child Basins

For each parent `OH*_p`, define candidate child basins:

```text
O*_{j|p}
OOH*_{k|p}
```

where `j` and `k` label same-site and nearby shifted basins accessible from
the parent `OH*` environment.

### Nearby Site

A nearby candidate site is one that satisfies at least one of:

- same registry site as the parent;
- shares at least one support metal atom with the parent site;
- lies within the first local metal-neighbor shell in xy;
- appears as a top site in the `O*` or `OOH*` PT run and overlaps the same
  local metal region.

This definition should be kept local. The child basin should represent a
plausible rearrangement from the parent `OH*`, not a global migration to a
remote clean-surface minimum.

## 4. Required Input Data

Recommended calculations:

- `OH_pt`: primary parent active-site ensemble;
- `O_pt`: auxiliary source of likely `O*` child sites;
- `OOH_pt`: auxiliary source of likely `OOH*` child sites;
- clean or parent-specific `*` reference slabs for CHE.

For the current TiZrMo MXene case, the relevant run directories are:

```text
/gpfs/scratch/acad/htbase/wchen/mxene/cmc/TiZrMo/X2CO2/ads/OH_pt
/gpfs/scratch/acad/htbase/wchen/mxene/cmc/TiZrMo/X2CO2/ads/O_pt
/gpfs/scratch/acad/htbase/wchen/mxene/cmc/TiZrMo/X2CO2/ads/OOH_pt
```

Use the fixed-temperature trajectories near operating temperature, for
example:

```text
results/replica_295K.traj
results/replica_303K.traj
results/replica_312K.traj
```

## 5. Step-By-Step Workflow

### Step 1: Extract OH Parent Basins

Run local motif analysis on the production window of the `OH*` PT trajectory:

```bash
PYTHONPATH=. python3 examples/adsorbate/analysis/analyze_local_adsorption_motifs.py \
  --traj /gpfs/scratch/acad/htbase/wchen/mxene/cmc/TiZrMo/X2CO2/ads/OH_pt/results/replica_303K.traj \
  --site-elements Ti Zr Mo \
  --substrate-elements Ti Zr Mo C \
  --functional-elements O \
  --site-types atop fcc hcp \
  --surface-side top \
  --anchor-element O \
  --start PRODUCTION_FRAME_START \
  --out-dir analysis/oer_parent_conditioned/OH \
  --write-representatives \
  --representative-group-by site \
  --representative-top-k 10 \
  --representative-n-per-group 1
```

The reusable workflow interface is:

```bash
PYTHONPATH=. python3 examples/adsorbate/runners/run_reaction_postprocess.py \
  --config analysis/oer_parent_conditioned/OH_pt/reaction_postprocess.yaml
```

Choose `PRODUCTION_FRAME_START` after checking equilibration. The output
provides:

- site/motif populations;
- support indices;
- representative full-slab frames for follow-up construction.

Select the top parent basins by population. A practical starting point is
top 3-5 parent `OH*` sites plus any chemically distinct lower-population basin.

### Step 2: Identify Candidate Child Sites

For each selected parent `OH*_p`, collect local candidate sites for `O*` and
`OOH*`.

Recommended candidate set:

```text
O* candidates for parent p:
  1. same site as OH*_p, produced by removing H
  2. top nearby O-preferred site sharing support atoms or local region
  3. second nearby O-preferred site
  4. third nearby O-preferred site

OOH* candidates for parent p:
  1. same site as the selected O* basin
  2. top nearby OOH-preferred site sharing support atoms or local region
  3. second nearby OOH-preferred site
  4. third nearby OOH-preferred site
```

The number four is not fundamental. It is a tractable starting shortlist. Add
more candidates if:

- several sites have comparable PT populations;
- relaxation collapses different initial guesses into distinct minima;
- `O_pt` or `OOH_pt` shows important low-energy sites nearby.

Use the `O_pt` and `OOH_pt` motif summaries as proposal lists, not as final
clean-surface references.

### Step 3: Construct O Child Basins From Each OH Parent

For each parent representative frame:

1. Same-site `O*`:
   - remove the H atom from `OH*`;
   - keep the adsorbate O at the parent anchor position;
   - relax.

2. Site-shifted `O*`:
   - start from the same parent slab/background;
   - remove the parent `OH*`;
   - insert atomic `O*` at nearby candidate registry sites;
   - relax each candidate.

After relaxation:

- assign the relaxed adsorbate to a registry site;
- discard duplicates that relax into the same site/basin;
- keep distinct basins with their energies and support assignments.

### Step 4: Construct OOH Child Basins

For each retained `O*_{j|p}` basin:

1. Same-site `OOH*`:
   - attach an `OH` group to the adsorbed O to form `OOH*`;
   - generate several initial orientations;
   - relax and keep intact minima.

2. Site-shifted `OOH*`:
   - start from the same parent local environment;
   - insert `OOH*` at nearby `OOH`-preferred registry sites;
   - try multiple orientations;
   - relax and keep distinct intact minima.

Because `OOH*` is orientation-sensitive, use multiple initial orientations.
Reject or separately label dissociated structures; they should not enter the
standard associative OER free-energy ladder unless intentionally studying a
different mechanism.

### Step 5: Compute Basin Free Energies

For each retained basin, compute a Gibbs free energy:

```text
G_X,i = E_X,i + ZPE_X,i - T S_X,i + correction_X
```

where `X` is `OH`, `O`, or `OOH`.

At minimum, keep the energy bookkeeping explicit:

- MLIP-relaxed energy for screening;
- DFT-relaxed or DFT single-point energy for final reporting;
- ZPE/entropy corrections from a consistent source;
- CHE references for `H2O`, `H2`, and `O2`.

If only MLIP energies are used, report the result as a screening-level OER
profile, not a final DFT-quality overpotential.

## 6. Parent-Conditioned Basin Routes

For one parent `OH*_p`, first group raw local samples into distinct retained
basins:

```text
O*_{j|p}
OOH*_{k|p}
```

where `j` and `k` label unique adsorption configurations after local CMC/PT,
quenching, and geometry/energy de-duplication. The current workflow keeps a
representative free energy for each basin rather than reducing the whole child
ensemble to one log-sum-exp state free energy.

The explicit route is:

```text
r = (p, j, k)
```

OOH* candidates generated from a specific O* candidate carry
`parent_o_candidate_id`; in that case the workflow pairs only compatible
`O*_{j|p}` and `OOH*_{k|p}` basins. If no such provenance is available, all
local O* x OOH* basin combinations for the parent are allowed.

The route weight is based on the sampled parent population and child basin
counts:

```text
W_r(raw) = P(OH*_p) P(O*_j | p) P(OOH*_k | p, O*_j)
```

and is normalized over all ready routes before ensemble reporting.

## 7. Parent-Conditioned OER Free-Energy Steps

For each explicit route `r=(p,j,k)`, define:

```text
DeltaG1_r = G_OH,p - G_* + CHE_reference_1
DeltaG2_r = G_O,j|p - G_OH,p + CHE_reference_2
DeltaG3_r = G_OOH,k|p - G_O,j|p + CHE_reference_3
DeltaG4_r = G_* - G_OOH,k|p + CHE_reference_4
```

The exact CHE reference terms depend on the chosen convention. In the common
four-step CHE convention, these correspond to:

```text
* + H2O      -> OH*  + H+ + e-
OH*         -> O*   + H+ + e-
O* + H2O    -> OOH* + H+ + e-
OOH*        -> *    + O2 + H+ + e-
```

At applied potential `U` versus RHE:

```text
DeltaG_i(U) = DeltaG_i(0) - e U
```

for each one-electron PCET step under the standard CHE convention.

The route-specific overpotential is:

```text
eta_r = max(DeltaG1_r, DeltaG2_r, DeltaG3_r, DeltaG4_r) / e - 1.23 V
```

This gives a distribution of OER profiles per populated `OH*` parent basin
instead of one profile from averaged child states.

## 8. Full Parent Ensemble

For the current workflow, the top-level ensemble summary should be based on a
dilute-limit model for the first `OH*` adsorption step, not on direct
population-weighting of the site-specific overpotentials.

The reason is that the canonical/PT `OH*` run gives a conditional distribution:

```text
P(parent site p | one OH* is present somewhere)
```

This is useful for identifying parent basins and for preserving the local
background structure, but it is not yet the same as the catalyst-level
probability that `OH*` forms at site `p` under operating conditions.

### Recommended Ensemble Weight

Use the site-resolved `DeltaG1_p` values from the parent-conditioned CHE table
to define a dilute-limit `OH*` occupation weight:

```text
w_p(U) ∝ g_p exp[-beta DeltaG1,p(U)]
```

where:

- `p` labels the parent `OH*` basin;
- `g_p` is a site degeneracy factor;
- `beta = 1 / (k_B T)`.

In the current code, the default choice is:

```text
g_p = 1
```

for each retained parent basin, i.e. uniform site degeneracy.

Because every `OH*` formation step contains one proton-electron transfer, the
relative site weights on the RHE scale are already determined by `DeltaG1_p` at
`U = 0`:

```text
DeltaG1,p(U) = DeltaG1,p(0) - eU
```

and the common `exp(beta eU)` factor cancels after normalization. Therefore the
current dilute-limit site weights can be built directly from the `U = 0`
`DeltaG1_p` values already present in the CHE summary.

### Route-Weighted Ensemble Summary

After normalizing the dilute-limit weights,

```text
W_p = w_p / sum_q w_q
```

the recommended top-level descriptor is the route-weighted mean overpotential:

```text
eta_ens = sum_p W_p eta_p
```

where each `eta_p` already comes from one parent-conditioned OER route:

```text
eta_p = max(DeltaG1_p, DeltaG2_p, DeltaG3_p, DeltaG4_p) / e - 1.23 V
```

This is the current top-level ensemble descriptor written by the workflow.

### What Not To Use As The Main Ensemble Summary

Do not use the overpotential of an averaged free-energy profile as the main
ensemble result:

```text
max(sum_p W_p DeltaG1_p, ..., sum_p W_p DeltaG4_p) / e - 1.23 V
```

This mixes different parent routes before applying the nonlinear `max(...)`
operation, so it does not correspond to one physical OER pathway. It can still
be reported as a secondary thermodynamic profile diagnostic, but it should not
replace the route-weighted summary when the concern is that some otherwise
promising channels are unlikely to host the first `OH*` adsorbate.

### Current Shared Clean Reference

This dilute-limit ensemble treatment is valid only when all parent-conditioned
states are referenced to one consistent clean slab. For a fixed alloy slab, use
one shared relaxed `00_clean` reference, selected from the lowest-energy
converged per-site `00_clean` structures, and keep the other per-site clean
slabs only as diagnostics for adsorbate-induced hysteresis.

### Recommended Reporting

- route-resolved `DeltaG_i,r` and `eta_r` from the explicit grouped-basin
  parent-conditioned table;
- the route-weighted ensemble overpotential `eta_ens`;
- the dominant weighted route and its parent basin;
- the sensitivity of the ensemble result to the assumed degeneracy model;
- the distribution of `eta_r` values across parent basins and child basins.

## 9. Role Of Independent O and OOH PT Runs

Independent `O_pt` and `OOH_pt` runs are not required as clean-surface
thermodynamic references in this protocol. Their role is:

- propose likely nearby child basins;
- reveal whether `O*` or `OOH*` prefers a different local site type;
- help avoid missing obvious low-energy local minima;
- provide sanity checks on candidate construction.

Do not interpret an isolated global `O*` or `OOH*` clean-surface minimum as
the mandatory child state for every `OH*` parent. That would break the
sequential parent-conditioned picture.

## 10. Practical Acceptance Criteria

A candidate basin is suitable for OER analysis if:

- the adsorbate remains intact when molecular integrity is required;
- the relaxed structure maps to a well-defined local site;
- no unphysical detachment or floating adsorbate is present;
- repeated orientations/initial guesses either converge to the same minimum or
  are retained as distinct basins;
- the local slab/background is consistent with the parent environment.

For `OOH*`, always check O-O and O-H connectivity after relaxation or MD.

## 11. What To Report

For each selected parent `OH*` basin:

```text
parent_id
OH support/site/motif
OH population
retained O child basins
retained OOH child basins
DeltaG1_p, DeltaG2_p, DeltaG3_p, DeltaG4_p
eta_p
limiting step
```

For the material as a whole:

- top parent-conditioned profiles;
- distribution of `eta_p`;
- dilute-limit `OH*`-weighted route overpotential summary `eta_ens`;
- dominant parent basin under the dilute-limit `OH*` weighting model;
- whether `OH*`, `O*`, and `OOH*` stay in the same local region or require
  site shifts;
- sensitivity to including/excluding nearby child basins.

Avoid reporting only one overpotential unless one parent basin clearly
dominates and all child basins collapse to one thermodynamic path.

For the current code outputs, the recommended hierarchy is:

- `oer_routes.csv`: primary site-resolved parent-conditioned route table,
  with one row per explicit grouped-basin route;
- `oer_ensemble.csv`: single top-level ensemble descriptor using normalized
  route weights from parent populations and child basin counts;
- `oer_states.csv`: compact state-level provenance and free-energy table;
- `vibration_summary.csv`: optional harmonic corrections for every relaxed
  clean/`OH*`/`O*`/`OOH*` candidate, with the local slab mask defined from the
  corresponding clean structure and reused exactly for the adsorbate states;
  CHE can then consume these harmonic free energies directly instead of the
  bare relaxed electronic energies;
- `references/h2/che_snippet.yaml` and `references/h2o/che_snippet.yaml`:
  optional gas-phase molecular reference energies and IdealGasThermo settings
  generated by the integrated `reference_thermo` stage before CHE.

## 12. Minimal Next Actions For The Current Project

1. Confirm all `OH_pt`, `O_pt`, and `OOH_pt` runs have completed cleanly.
2. Run motif analysis for `OH_pt` at `295K`, `303K`, and `312K`.
3. Select top `OH*` parent basins by population and chemical diversity.
4. Use `O_pt` and `OOH_pt` motif summaries to identify nearby child proposals.
5. Generate same-site and nearby shifted `O*` and `OOH*` candidates.
6. Relax candidates and remove duplicates.
7. Compute local harmonic corrections for the selected route minima when the
   final CHE table needs adsorbate/slab ZPE and thermal terms. Define the slab
   vibration mask from the clean state and reuse that exact slab-atom set for
   the adsorbate states.
8. Compute gas-phase `H2` and `H2O` vibrational/IdealGasThermo reference terms
   with the same calculator when the CHE table should avoid manual reference
   corrections.
9. Group parent-conditioned child samples into unique O* and OOH* basins.
10. Build explicit grouped-basin CHE routes and the route-weighted ensemble
   summary.
