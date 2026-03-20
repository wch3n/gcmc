# gcmc

Monte Carlo workflows for alloy and adsorbate sampling on ASE-compatible atomistic models.

The codebase supports:

- canonical alloy Monte Carlo on substitutional sublattices
- fixed-loading canonical adsorbate Monte Carlo with replica exchange support
- site-based grand-canonical adsorbate MC (`AdsorbateGCMC`)
- legacy grand-canonical adsorbate MC (`GCMC`)
- semi-grand alloy MC (`SemiGrandAlloyMC`)
- local relaxation and short hybrid MC/MD proposals
- local multiprocessing and Ray-backed distributed replica exchange
- YAML-driven workflow wrappers for alloy and adsorbate runs
- MXene-oriented analysis utilities for ordering, adsorption sites, surface motifs, and SRO

## Installation

Base install:

```bash
pip3 install .
```

Optional extras:

```bash
pip3 install .[speedups]
pip3 install ray
```

Notes:

- `mace-torch` is a package dependency and enables the MACE examples.
- The repository also works with other ASE-compatible calculators. Several production examples use `Symmetrix`, which is not bundled here.

## Main Components

### MC engines

- `AlloyCMC`
  - canonical swap MC for substitutional alloys
  - neighbor-aware swap proposals
  - optional local relaxation
  - optional hybrid MC/MD proposals

- `AdsorbateCMC`
  - fixed-loading canonical MC for adsorbates on slabs
  - monoatomic and rigid molecular adsorbates
  - move modes:
    - `displacement`
    - `site_hop`
    - `reorientation`
    - `hybrid` (mixture of MC move types)
  - optional hybrid MC/MD proposals
  - instantaneous site registry for `atop`, `bridge`, `fcc`, and `hcp` sites on distorted MXene surfaces

- `AdsorbateGCMC`
  - site-based grand-canonical MC for monoatomic adsorbates
  - insertion / deletion / canonical moves on an instantaneous site registry
  - single-run and `mu`-scan workflow wrappers

- `GCMC`
  - legacy variable-loading grand-canonical adsorbate MC

- `SemiGrandAlloyMC`
  - semi-grand alloy MC for composition control through chemical-potential differences

### Replica exchange

- `ReplicaExchange`
  - ensemble-neutral PT driver
  - works with `AlloyCMC` and `AdsorbateCMC`
  - backends:
    - `execution_backend="multiprocessing"`
    - `execution_backend="ray"`

### Analysis

- `MXeneOrderingAnalyzer`
  - layer composition and ordering summaries
- `MXeneSurfaceMotifAnalyzer`
  - local surface motif statistics
- `MXeneAdsorptionSiteAnalyzer`
  - adsorption-site candidate generation on MXene surfaces
- `MXeneSROAnalyzer`
  - Warren-Cowley SRO analysis and summary export

## Repository Layout

```text
gcmc/
  alloy_cmc.py
  adsorbate_cmc.py
  adsorbate_gcmc.py
  gcmc.py
  sgcmc.py
  replica.py
  workflows.py
  execution_backends.py
  analysis/
examples/
  alloy/
  adsorbate/
```

The example entrypoints are now thin `--config` wrappers around YAML files. In practice, you edit the YAML and keep the Python launcher unchanged.

## YAML-Driven Workflows

The recommended frontends are:

- `examples/alloy/run_cmc_alloy.py --config examples/alloy/alloy_cmc.yaml`
- `examples/alloy/run_pt_cmc_alloy.py --config examples/alloy/alloy_pt.yaml`
- `examples/alloy/run_pt_cmc_alloy_ray.py --config examples/alloy/alloy_pt_ray.yaml`
- `examples/adsorbate/run_adsorbate_cmc.py --config examples/adsorbate/adsorbate_cmc.yaml`
- `examples/adsorbate/run_adsorbate_gcmc.py --config examples/adsorbate/adsorbate_gcmc.yaml`
- `examples/adsorbate/run_adsorbate_gcmc_scan.py --config examples/adsorbate/adsorbate_gcmc_scan.yaml`

The YAML files are grouped by section:

- `system`
  - input structure, frame selection, supercell/repeat, alloy initialization, adsorbate setup
- `cmc` / `gcmc` / `pt`
  - engine-specific Monte Carlo settings
- `backend`
  - multiprocessing or Ray execution settings
- `calculator`
  - calculator type and model/device settings
- `output`
  - output prefix or output directory

The workflow loaders also accept the older `interval` key and map it to `write_interval`.

## Alloy Workflows

### Single-temperature alloy MC

See:

- `examples/alloy/run_cmc_alloy.py`
- `examples/alloy/alloy_cmc.yaml`

Typical invocation:

```bash
python3 examples/alloy/run_cmc_alloy.py --config examples/alloy/alloy_cmc.yaml
```

Core YAML sections:

- `system`
  - `snapshot`, `frame`, `site_element`, `composition`, `initialization_seed`
- `cmc`
  - `temperature`, `swap_elements`, `swap_mode`, relaxation / MD controls, `nsweeps`, `write_interval`
- `calculator`
  - `calculator`, `model_file`, `device`
- `output`
  - `output_prefix`

Underlying engine construction remains equivalent to:

```python
from ase.build import make_supercell
from ase.io import read
from gcmc import AlloyCMC
from gcmc.utils import initialize_alloy_sublattice

pristine = read("POSCAR.Ti2CO2")
atoms = make_supercell(pristine, [[8, 0, 0], [5, 10, 0], [0, 0, 1]])
atoms = initialize_alloy_sublattice(
    atoms=atoms,
    site_element="Ti",
    composition={"Ti": 0.5, "Mo": 0.5},
    seed=67,
)

mc = AlloyCMC(
    atoms=atoms,
    calculator=calculator,
    T=600.0,
    swap_elements=["Ti", "Mo"],
    swap_mode="hybrid",
    hybrid_neighbor_prob=0.8,
    relax=True,
    local_relax=True,
    relax_steps=10,
    fmax=0.1,
)
stats = mc.run(nsweeps=200, traj_file="alloy.traj")
```

### Replica exchange for alloys

See:

- `examples/alloy/run_pt_cmc_alloy.py`
- `examples/alloy/alloy_pt.yaml`
- `examples/alloy/run_pt_cmc_alloy_ray.py`
- `examples/alloy/alloy_pt_ray.yaml`
- `examples/alloy/run_pt_cmc_alloy_ray.slurm`

The Ray example demonstrates:

- nonuniform temperature grids
- multi-node GPU resource partitioning
- placement groups for actor spreading

Typical invocations:

```bash
python3 examples/alloy/run_pt_cmc_alloy.py --config examples/alloy/alloy_pt.yaml
python3 examples/alloy/run_pt_cmc_alloy_ray.py --config examples/alloy/alloy_pt_ray.yaml
```

The PT YAML is split into:

- `system`
  - reference snapshot, frame, optional `repeat` or `supercell_matrix`, alloy initialization
- `pt`
  - temperature ladder, swap/report/checkpoint cadence, cycle counts
- `mc`
  - per-replica `AlloyCMC` settings
- `backend`
  - `backend`, `n_gpus`, `workers_per_gpu`, and Ray-specific options
- `calculator`
  - calculator type and model
- `output`
  - `output_dir`, stats/results/checkpoint files

## Adsorbate Workflows

### Canonical fixed-loading adsorbate MC

See:

- `examples/adsorbate/run_adsorbate_cmc.py`
- `examples/adsorbate/adsorbate_cmc.yaml`

The recommended initialization path is `AdsorbateCMC.from_clean_surface(...)`. It builds an instantaneous site registry from the current slab geometry and places adsorbates on allowed sites.

Typical invocation:

```bash
python3 examples/adsorbate/run_adsorbate_cmc.py --config examples/adsorbate/adsorbate_cmc.yaml
```

Example:

```python
from ase.io import read
from gcmc import AdsorbateCMC

atoms = read("POSCAR.Ti2CO2")
cmc = AdsorbateCMC.from_clean_surface(
    atoms=atoms,
    calculator=calculator,
    T=300.0,
    adsorbate="OH",
    adsorbate_anchor_index=0,
    substrate_elements=("Ti", "C"),
    functional_elements=("O",),
    site_elements=("Ti",),
    surface_side="top",
    site_type=("atop", "bridge", "fcc", "hcp"),
    coverage=0.25,
    move_mode="hybrid",
    min_clearance=0.9,
    site_match_tol=0.6,
    surface_layer_tol=0.5,
    termination_clearance=0.8,
)
stats = cmc.run(nsweeps=200, traj_file="adsorbate.traj")
```

### Adsorbate move modes

`AdsorbateCMC` supports:

- `displacement`
  - continuous random translation of one adsorbate/group
- `site_hop`
  - discrete hop between instantaneous high-symmetry sites
- `reorientation`
  - rigid rotation of a molecular adsorbate about its anchor atom
- `hybrid`
  - mixture of `site_hop`, `reorientation`, and `displacement`

Important distinction:

- `move_mode="hybrid"` mixes MC move types.
- `enable_hybrid_md=True` adds short MD proposals on top of the MC move mode.

### Adsorbate tolerances

The public adsorbate/site tolerances were simplified to four main knobs:

- `min_clearance`
  - minimum full 3D separation between trial adsorbate atoms and the slab
- `site_match_tol`
  - in-plane tolerance used to recognize distorted `atop` / `bridge` / `fcc` / `hcp` sites
- `surface_layer_tol`
  - z-clustering tolerance used to identify the exposed surface layer
- `termination_clearance`
  - minimum 3D clearance from surface terminations when placing or hopping adsorbates

An optional `bridge_cutoff` override is also available. Other internal thresholds are derived automatically.

### Sweep definition in `AdsorbateCMC`

The adsorbate sweep definition now follows the active degrees of freedom:

- `site_hop` / `hybrid`
  - one sweep = number of eligible adsorption sites in the current registry
- `displacement` / `reorientation`
  - one sweep = number of adsorbates

This mirrors `AlloyCMC`, where one sweep scales with the active swappable sites.

### Replica exchange for fixed-loading adsorbates

See:

- `examples/adsorbate/run_pt_adsorbate_cmc.py`

This is appropriate for:

- fixed adsorbate loading
- temperature-assisted equilibration across adsorption basins

It is not a grand-canonical adsorption/desorption workflow. For variable loading, use `GCMC`.

### Grand-canonical adsorbate MC

See:

- `examples/adsorbate/run_adsorbate_gcmc.py`
- `examples/adsorbate/adsorbate_gcmc.yaml`
- `examples/adsorbate/run_adsorbate_gcmc_scan.py`
- `examples/adsorbate/adsorbate_gcmc_scan.yaml`

Recommended single-run invocation:

```bash
python3 examples/adsorbate/run_adsorbate_gcmc.py --config examples/adsorbate/adsorbate_gcmc.yaml
```

Recommended `mu`-scan invocation:

```bash
python3 examples/adsorbate/run_adsorbate_gcmc_scan.py --config examples/adsorbate/adsorbate_gcmc_scan.yaml
```

The scan workflow supports:

- multiprocessing scans via:
  - `backend.n_workers`
  - `backend.devices`
  - optional `backend.gpu_ids`
- Ray scans via:
  - `backend.ray_address`
  - `backend.ray_log_to_driver`
  - `backend.ray_num_cpus_per_task`
  - `backend.ray_num_gpus_per_task`
- per-task output directories and `summary.csv` aggregation

The legacy `examples/adsorbate/run_gcmc.py` script remains in the repository, but the site-based `AdsorbateGCMC` workflows are the preferred route for current adsorbate free-energy work.

## Hybrid MC/MD

Both `AlloyCMC` and `AdsorbateCMC` support short MD proposals.

Relevant options include:

- `enable_hybrid_md`
- `md_move_prob`
- `md_steps`
- `md_timestep_fs`
- `md_ensemble`
- `md_accept_mode`
- `md_planar`
- `md_planar_axis`

Use cases:

- local finite-temperature accommodation around accepted chemical moves
- improved local exploration beyond pure rigid MC proposals

Caution:

- high-temperature unconstrained MD can drive 2D slabs off the intended manifold
- for ordering studies on a metastable 2D scaffold, pure swap/site MC plus local relaxation is often easier to interpret

## Replica Exchange Backends

`ReplicaExchange` supports two execution backends:

- `execution_backend="multiprocessing"`
  - local backend for one host
- `execution_backend="ray"`
  - distributed backend for multi-host, multi-GPU runs

### Backend semantics

For `ReplicaExchange.from_auto_config(...)`:

- `n_gpus`
  - number of logical GPU slots available to PT workers
- `workers_per_gpu`
  - number of workers per logical GPU slot

For Ray actors:

- if `backend_kwargs["actor_options"]["num_gpus"]` is not set, the default is:
  - `1 / workers_per_gpu`

### Ray example

```python
pt = ReplicaExchange.from_auto_config(
    atoms_template=atoms,
    T_start=300.0,
    T_end=3000.0,
    n_replicas=32,
    grid_space="beta",
    calculator_class=Symmetrix,
    calc_kwargs=calc_kwargs,
    mc_class=AlloyCMC,
    mc_kwargs=mc_kwargs,
    n_gpus=8,
    workers_per_gpu=2,
    execution_backend="ray",
    backend_kwargs={
        "init_kwargs": {"address": "auto", "log_to_driver": False},
        "use_placement_group": True,
        "placement_group_strategy": "SPREAD",
    },
)
pt.run(n_cycles=200, equilibration_cycles=20)
```

### Placement group controls

Supported Ray placement-group options in `backend_kwargs`:

- `use_placement_group`
- `placement_group_strategy`
- `placement_group_bundles`
- `placement_group_name`
- `placement_group_lifetime`
- `placement_group_capture_child_tasks`
- `remove_placement_group_on_stop`

If placement groups are enabled and bundles are not provided, bundles are generated automatically from actor CPU/GPU requirements.

## Analysis Workflows

### Ordering and site analysis

See:

- `examples/alloy/analyze_mxene_ordering.py`
- `examples/alloy/analyze_surface_motifs.py`
- `examples/alloy/generate_adsorption_sites.py`

### MXene SRO analysis

See:

- `examples/alloy/analyze_mxene_sro.py`
- `examples/alloy/plot_mxene_sro_summary.py`

The SRO workflow supports:

- reference-supercell reconstruction from primitive cells
- trajectory-wise CSV export
- combined phase summaries
- layer-resolved ordering summaries for MXene alloy slabs

## Checkpoints and Outputs

All MC engines write trajectory / thermo / checkpoint state files.

Common outputs include:

- `*.traj`
  - saved structures
- `*.dat`
  - simple thermo logs
- `*.pkl`
  - MC checkpoint state

Replica exchange also writes:

- `*_results.csv`
- `*_stats.csv`
- master PT checkpoint state

## Notes on Calculators

This package expects ASE-compatible calculators.

Examples in the repository use:

- `mace.calculators.MACECalculator`
- `Symmetrix`
- ASE `LennardJones` for smoke tests and examples

When using GPU-backed calculators under Ray:

- keep calculator execution inside actors that request explicit GPU resources
- avoid relying on implicit GPU visibility behavior in zero-GPU tasks

## Current Version

`pyproject.toml` currently declares:

- version `0.9.0`
