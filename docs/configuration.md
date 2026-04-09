# Configuration Reference

This repository supports both nested and flat YAML layouts.

- Nested layout matches the workflow sections:
  - `system`
  - `cmc` / `gcmc` / `pt`
  - `backend`
  - `calculator`
  - `output`
- Flat layout is still accepted for backward compatibility.

The authoritative defaults live in `gcmc/workflows.py`.

## Workflow-to-config mapping

| Workflow | Entry point | Main sections |
| --- | --- | --- |
| Alloy CMC | `AlloyCMCWorkflow` | `system`, `mc`, `calculator`, `output` |
| Alloy replica exchange | `AlloyReplicaExchangeWorkflow` | `system`, `pt`, `mc`, `backend`, `calculator`, `output` |
| Adsorbate CMC | `AdsorbateCMCWorkflow` | `system`, `cmc`, `calculator`, `output` |
| Adsorbate GCMC | `AdsorbateGCMCWorkflow` | `system`, `gcmc`, `calculator`, `output` |
| Adsorbate GCMC scan | `AdsorbateGCMCScanWorkflow` | `system`, `gcmc`, `backend`, `calculator`, `output` |

## Important parsing rules

- `snapshot`, `model`, `model_file`, and `output_dir` are resolved relative to the YAML file location when given as relative paths.
- `output_prefix` is used as written; it is not rewritten relative to the config path.
- The older key `interval` is accepted and mapped to `write_interval`.
- `mu_range` + `n_mu_points` can be used instead of `mu_values` for scan workflows.
- `support_xy_tol` defaults to `max(1.2, 2.5 * site_match_tol)` when omitted.
- `termination_site_xy_tol` defaults to `support_xy_tol` when omitted.

## Section reference

- Shared keys used across multiple workflows: `docs/configuration/shared.md`
- Adsorbate CMC / GCMC / mu-exchange keys: `docs/configuration/adsorbate.md`
- Alloy CMC / replica-exchange keys: `docs/configuration/alloy.md`

## Typical layouts

Alloy replica exchange:

```yaml
system:
  snapshot: POSCAR
  frame: 0
  site_element: Ti
  composition:
    Ti: 0.5
    Mo: 0.5

pt:
  T_start: 1800
  T_end: 100
  n_replicas: 64
  n_cycles: 800

mc:
  swap_elements: [Ti, Mo]
  swap_mode: hybrid

backend:
  backend: ray
  n_gpus: 8

calculator:
  calculator: symmetrix
  model_file: model.json

output:
  output_dir: out
```

Adsorbate GCMC scan with `mu` exchange:

```yaml
system:
  snapshot: slab.traj
  frame: -1
  adsorbate: H2O.xyz
  adsorbate_anchor_index: 0
  site_elements: [Ti, Zr, Mo]
  site_type: [atop, fcc, hcp]
  substrate_elements: [Ti, Zr, Mo, C]

gcmc:
  temperature: 300
  mu_values: [-13.4, -13.2, -13.0]
  seeds: [67, 68]
  use_mu_exchange: true
  swap_interval: 20
  mu_exchange_cycles: 50
  move_mode: hybrid
  site_hop_prob: 0.3
  reorientation_prob: 0.4

backend:
  backend: ray
  ray_num_gpus_per_task: 0.5

calculator:
  calculator: symmetrix
  model_file: model.json

output:
  output_dir: out_mu_re
```
