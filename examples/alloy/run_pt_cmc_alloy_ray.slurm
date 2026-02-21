#!/bin/bash
#SBATCH -J gcmc-ray-pt
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH -p gpu
#SBATCH -o slurm-%j.out

set -euo pipefail

# Activate your environment containing gcmc + ray + calculator deps.
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

echo "Head node: $head_node ($head_ip:$port)"
echo "Nodes: ${nodes[*]}"

# Start Ray head.
srun -N1 -n1 -w "$head_node" \
  ray start --head \
    --node-ip-address="$head_ip" \
    --port="$port" \
    --num-gpus=4 \
    --num-cpus="$SLURM_CPUS_PER_TASK" \
    --disable-usage-stats

# Start Ray workers.
for n in "${nodes[@]:1}"; do
  srun -N1 -n1 -w "$n" \
    ray start \
      --address "${head_ip}:${port}" \
      --num-gpus=4 \
      --num-cpus="$SLURM_CPUS_PER_TASK" \
      --disable-usage-stats
done

# Export topology for the Python example.
export PT_NODES="${SLURM_NNODES}"
export PT_GPUS_PER_NODE="4"
export PT_WORKERS_PER_GPU="2"

# Launch driver on head node.
srun -N1 -n1 -w "$head_node" python examples/alloy/run_pt_cmc_alloy_ray.py
