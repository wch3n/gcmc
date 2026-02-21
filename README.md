# Grand Canonical Monte Carlo (GCMC) Simulations

This repository provides Monte Carlo and replica-exchange workflows for alloy and adsorbate systems with ASE-compatible calculators (including MLIPs).

## Installation

```bash
pip3 install .
```

To enable Ray-based distributed replica exchange:

```bash
pip3 install ray
```

## Replica Exchange Backends

`ReplicaExchange` now supports two execution backends:

- `execution_backend="multiprocessing"`: default local backend (single host).
- `execution_backend="ray"`: distributed backend for multi-host, multi-GPU runs.

### Backend Semantics

For `ReplicaExchange.from_auto_config(...)`:

- `n_gpus` means the number of logical GPU slots available to PT workers.
  - Local multiprocessing: normally physical GPUs on one host.
  - Ray: total slots across the Ray cluster.
- `workers_per_gpu` means number of workers (actors/processes) per logical GPU slot.

Ray backend GPU assignment behavior:

- If `backend_kwargs["actor_options"]["num_gpus"]` is not set, it defaults to:
  - `num_gpus_per_actor = 1 / workers_per_gpu`
- This automatically supports `workers_per_gpu > 1` without GPU over-request by default.

## Ray Usage Example

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
    n_gpus=8,                  # total logical GPU slots across cluster
    workers_per_gpu=2,         # two workers per slot
    execution_backend="ray",
    backend_kwargs={
        "init_kwargs": {"address": "auto", "log_to_driver": False},
        # Optional override; default is 1/workers_per_gpu.
        # "actor_options": {"num_gpus": 0.5},
    },
)
pt.run(n_cycles=200, equilibration_cycles=20)
```

## Placement Group Controls (Ray)

For stricter actor placement, pass these in `backend_kwargs`:

- `use_placement_group`: `True`/`False` (default `False`)
- `placement_group_strategy`: e.g. `"SPREAD"`, `"PACK"`
- `placement_group_bundles`: optional explicit bundles list
- `placement_group_name`: optional name
- `placement_group_lifetime`: optional lifetime
- `placement_group_capture_child_tasks`: default `True`
- `remove_placement_group_on_stop`: default `True`

If `use_placement_group=True` and bundles are not provided, bundles are auto-generated from actor CPU/GPU requirements.

## Slurm Template for 2 Nodes x 4 GPUs

```bash
#!/bin/bash
#SBATCH -J gcmc-ray-pt
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -o slurm-%j.out

set -euo pipefail

source ~/.bashrc
# conda activate your_env

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_ip=$(srun -N1 -n1 -w "$head_node" hostname -I | awk '{print $1}')
port=6379

cleanup () {
  for n in "${nodes[@]}"; do
    srun -N1 -n1 -w "$n" ray stop --force >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT

srun -N1 -n1 -w "$head_node" \
  ray start --head \
    --node-ip-address="$head_ip" \
    --port="$port" \
    --num-gpus=4 \
    --num-cpus="$SLURM_CPUS_PER_TASK" \
    --disable-usage-stats

for n in "${nodes[@]:1}"; do
  srun -N1 -n1 -w "$n" \
    ray start \
      --address "${head_ip}:${port}" \
      --num-gpus=4 \
      --num-cpus="$SLURM_CPUS_PER_TASK" \
      --disable-usage-stats
done

srun -N1 -n1 -w "$head_node" python run_pt_ray.py
```
