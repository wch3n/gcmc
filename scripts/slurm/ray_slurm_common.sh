#!/bin/bash

ray_slurm_require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    return 1
  }
}

ray_slurm_init() {
  ray_slurm_require_cmd ray || return 1
  ray_slurm_require_cmd scontrol || return 1

  mapfile -t RAY_SLURM_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
  if [[ ${#RAY_SLURM_NODES[@]} -eq 0 ]]; then
    echo "ERROR: could not resolve nodes from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-<unset>}" >&2
    return 1
  fi

  RAY_SLURM_HEAD_NODE="${RAY_SLURM_NODES[0]}"
  RAY_PORT="${RAY_PORT:-6379}"
  RAY_HEAD_IP=$(getent ahostsv4 "$RAY_SLURM_HEAD_NODE" | awk 'NR==1{print $1}')
  if [[ -z "${RAY_HEAD_IP}" ]]; then
    echo "ERROR: failed to resolve head node IP for ${RAY_SLURM_HEAD_NODE}" >&2
    return 1
  fi

  RAY_HEAD_ADDR="${RAY_HEAD_IP}:${RAY_PORT}"
  export RAY_ADDRESS="${RAY_HEAD_ADDR}"
  export RAY_GPUS_PER_NODE="${RAY_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
  export RAY_CPUS_PER_NODE="${RAY_CPUS_PER_NODE:-${SLURM_CPUS_PER_TASK:-1}}"
  export RAY_STARTUP_DELAY="${RAY_STARTUP_DELAY:-5}"
  export RAY_READY_RETRIES="${RAY_READY_RETRIES:-30}"
  export RAY_READY_DELAY="${RAY_READY_DELAY:-2}"

  RAY_SLURM_PIDS=()
}

ray_slurm_stop() {
  if [[ ${#RAY_SLURM_NODES[@]:-} -gt 0 ]]; then
    for n in "${RAY_SLURM_NODES[@]}"; do
      srun -N1 -n1 -w "$n" ray stop --force >/dev/null 2>&1 || true
    done
  fi

  if [[ ${#RAY_SLURM_PIDS[@]:-} -gt 0 ]]; then
    for pid in "${RAY_SLURM_PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
}

ray_slurm_start() {
  ray_slurm_init || return 1
  trap ray_slurm_stop EXIT

  for n in "${RAY_SLURM_NODES[@]}"; do
    srun -N1 -n1 -w "$n" ray stop --force >/dev/null 2>&1 || true
  done

  echo "Starting Ray head at ${RAY_HEAD_ADDR}"
  srun --overlap -N1 -n1 -w "$RAY_SLURM_HEAD_NODE" \
    ray start --head \
      --node-ip-address="$RAY_HEAD_IP" \
      --port="$RAY_PORT" \
      --num-gpus="$RAY_GPUS_PER_NODE" \
      --num-cpus="$RAY_CPUS_PER_NODE" \
      --disable-usage-stats \
      --block &
  RAY_SLURM_PIDS+=("$!")

  sleep "$RAY_STARTUP_DELAY"

  for n in "${RAY_SLURM_NODES[@]:1}"; do
    echo "Starting Ray worker on ${n}"
    srun --overlap -N1 -n1 -w "$n" \
      ray start \
        --address "$RAY_HEAD_ADDR" \
        --num-gpus="$RAY_GPUS_PER_NODE" \
        --num-cpus="$RAY_CPUS_PER_NODE" \
        --disable-usage-stats \
        --block &
    RAY_SLURM_PIDS+=("$!")
  done

  for ((i=1; i<=RAY_READY_RETRIES; i++)); do
    if ray status --address "$RAY_HEAD_ADDR" >/dev/null 2>&1; then
      echo "Ray cluster is ready at ${RAY_HEAD_ADDR}"
      return 0
    fi
    sleep "$RAY_READY_DELAY"
  done

  echo "ERROR: Ray GCS did not become reachable at ${RAY_HEAD_ADDR}" >&2
  return 1
}
