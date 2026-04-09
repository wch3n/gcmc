# gcmc

Monte Carlo workflows for alloy and adsorbate sampling on ASE-compatible atomistic models.

## Documentation

- Configuration overview: `docs/configuration.md`
- Shared workflow keys: `docs/configuration/shared.md`
- Adsorbate workflow keys: `docs/configuration/adsorbate.md`
- Alloy workflow keys: `docs/configuration/alloy.md`

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

## Capabilities

- `AlloyCMC`
  - canonical swap MC for substitutional alloys
  - neighbor-aware proposals
  - optional relaxation and hybrid MC/MD
- `ReplicaExchange`
  - parallel tempering for alloy and fixed-loading adsorbate MC
  - `multiprocessing` and `ray` backends
- `AdsorbateCMC`
  - fixed-loading MC for atomic and rigid molecular adsorbates
  - `displacement`, `site_hop`, `reorientation`, and `hybrid` move modes
- `AdsorbateGCMC`
  - site-based grand-canonical MC
  - single-`mu`, `mu`-scan, and fixed-`T` `mu`-replica-exchange workflows
- `gcmc.analysis`
  - MXene ordering, motif, adsorption-site, and SRO analysis utilities

## Repository Layout

```text
gcmc/
  workflows.py
  replica.py
  execution_backends.py
  alloy_cmc.py
  adsorbate_cmc.py
  adsorbate_gcmc.py
  analysis/
docs/
  configuration.md
  configuration/
examples/
  alloy/
  adsorbate/
scripts/
  slurm/
tests/
```

## Quick Start

The recommended entry points are YAML-driven runners. In practice, edit the YAML and keep the Python launcher unchanged.

Alloy CMC:

```bash
python3 examples/alloy/runners/run_cmc_alloy.py \
  --config examples/alloy/configs/alloy_cmc.yaml
```

Alloy replica exchange:

```bash
python3 examples/alloy/runners/run_pt_cmc_alloy.py \
  --config examples/alloy/configs/alloy_pt.yaml
```

Ray-backed alloy replica exchange:

```bash
python3 examples/alloy/runners/run_pt_cmc_alloy_ray.py \
  --config examples/alloy/configs/alloy_pt_ray.yaml
```

Adsorbate CMC:

```bash
python3 examples/adsorbate/runners/run_adsorbate_cmc.py \
  --config examples/adsorbate/configs/adsorbate_cmc.yaml
```

Adsorbate GCMC:

```bash
python3 examples/adsorbate/runners/run_adsorbate_gcmc.py \
  --config examples/adsorbate/configs/adsorbate_gcmc.yaml
```

Adsorbate GCMC scan or `mu` exchange:

```bash
python3 examples/adsorbate/runners/run_adsorbate_gcmc_scan.py \
  --config examples/adsorbate/configs/adsorbate_gcmc_scan.yaml
```

## Configuration

The workflow loaders accept both:

- nested YAML sections: `system`, `cmc` / `gcmc` / `pt`, `backend`, `calculator`, `output`
- flat legacy YAML for backward compatibility

Use the docs for the actual keyword reference:

- `docs/configuration.md`
- `docs/configuration/shared.md`
- `docs/configuration/adsorbate.md`
- `docs/configuration/alloy.md`

## Examples and Analysis

- example layouts:
  - `examples/alloy/README.md`
  - `examples/adsorbate/README.md`
- shared Slurm helper for Ray jobs:
  - `scripts/slurm/ray_slurm_common.sh`

## Calculator Notes

The workflows expect ASE-compatible calculators.

Repository examples use:

- ASE `LennardJones`
- `mace.calculators.MACECalculator`
- `Symmetrix`

`Symmetrix` is not bundled with this repository.

## Version

`pyproject.toml` currently declares version `0.9.0`.
