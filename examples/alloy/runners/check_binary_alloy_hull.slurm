#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug-gpu
#SBATCH --job-name=binary_hull
#SBATCH --output=hull_check.log
#SBATCH --account=htbase
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb

set -euo pipefail

module --force purge
module load EasyBuild/2024a Python LAMMPS
ulimit -s unlimited

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="/gpfs/home/acad/ucl-modl/wchen/mxene_proj/gcmc${PYTHONPATH:+:${PYTHONPATH}}"

CODE_DIR="${CODE_DIR:-/gpfs/home/acad/ucl-modl/wchen/mxene_proj/gcmc}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the alloy run directory, for example /path/to/X2CO2/slab/xl}"

ELEMENT_A="${ELEMENT_A:?Set ELEMENT_A, for example Ti}"
ELEMENT_B="${ELEMENT_B:?Set ELEMENT_B, for example Zr}"

RESULTS_DIR="${RESULTS_DIR:-${RUN_DIR}/results}"
CONFIG="${CONFIG:-${RUN_DIR}/config.yaml}"
TRAJ="${TRAJ:-${RESULTS_DIR}/replica_${TARGET_TEMPERATURE_K:-300}K.traj}"
DAT="${DAT:-${RESULTS_DIR}/replica_${TARGET_TEMPERATURE_K:-300}K.dat}"
OUT_DIR="${OUT_DIR:-${RESULTS_DIR}/hull_check_${ELEMENT_A}${ELEMENT_B}}"

# Endpoint defaults:
#   1. ENDPOINT_A/ENDPOINT_B if explicitly supplied.
#   2. POSCAR.<element> in SEARCH_ROOT.
#   3. POSCAR.<element> one or two directories above RUN_DIR.
SEARCH_ROOT="${SEARCH_ROOT:-$(cd "${RUN_DIR}/../.." && pwd)}"
ENDPOINT_A="${ENDPOINT_A:-}"
ENDPOINT_B="${ENDPOINT_B:-}"

if [[ -z "${ENDPOINT_A}" ]]; then
  if [[ -f "${SEARCH_ROOT}/POSCAR.${ELEMENT_A}" ]]; then
    ENDPOINT_A="${SEARCH_ROOT}/POSCAR.${ELEMENT_A}"
  elif [[ -f "${RUN_DIR}/../POSCAR.${ELEMENT_A}" ]]; then
    ENDPOINT_A="${RUN_DIR}/../POSCAR.${ELEMENT_A}"
  elif [[ -f "${RUN_DIR}/../../POSCAR.${ELEMENT_A}" ]]; then
    ENDPOINT_A="${RUN_DIR}/../../POSCAR.${ELEMENT_A}"
  fi
fi
if [[ -z "${ENDPOINT_B}" ]]; then
  if [[ -f "${SEARCH_ROOT}/POSCAR.${ELEMENT_B}" ]]; then
    ENDPOINT_B="${SEARCH_ROOT}/POSCAR.${ELEMENT_B}"
  elif [[ -f "${RUN_DIR}/../POSCAR.${ELEMENT_B}" ]]; then
    ENDPOINT_B="${RUN_DIR}/../POSCAR.${ELEMENT_B}"
  elif [[ -f "${RUN_DIR}/../../POSCAR.${ELEMENT_B}" ]]; then
    ENDPOINT_B="${RUN_DIR}/../../POSCAR.${ELEMENT_B}"
  fi
fi

if [[ -z "${ENDPOINT_A}" || ! -f "${ENDPOINT_A}" ]]; then
  echo "ERROR: endpoint for ${ELEMENT_A} not found. Set ENDPOINT_A=/path/to/POSCAR.${ELEMENT_A}" >&2
  exit 1
fi
if [[ -z "${ENDPOINT_B}" || ! -f "${ENDPOINT_B}" ]]; then
  echo "ERROR: endpoint for ${ELEMENT_B} not found. Set ENDPOINT_B=/path/to/POSCAR.${ELEMENT_B}" >&2
  exit 1
fi

START="${START:-0}"
STEP="${STEP:-1}"
TEMPERATURE_K="${TEMPERATURE_K:-${TARGET_TEMPERATURE_K:-300}}"
FORMULA_COUNT_ELEMENT="${FORMULA_COUNT_ELEMENT:-C}"
METALS_PER_FORMULA="${METALS_PER_FORMULA:-2}"
ENDPOINT_FMAX="${ENDPOINT_FMAX:-0.03}"
ENDPOINT_RELAX_STEPS="${ENDPOINT_RELAX_STEPS:-300}"
ALLOY_FMAX="${ALLOY_FMAX:-0.03}"
ALLOY_RELAX_STEPS="${ALLOY_RELAX_STEPS:-300}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-1}"

stop_args=()
if [[ -n "${STOP:-}" ]]; then
  stop_args=(--stop "${STOP}")
fi

max_frames_args=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  max_frames_args=(--max-frames "${MAX_FRAMES}")
fi

relax_args=()
if [[ "${RELAX_ENDPOINTS:-0}" == "1" ]]; then
  relax_args=(--relax-endpoints --endpoint-fmax "${ENDPOINT_FMAX}" --endpoint-relax-steps "${ENDPOINT_RELAX_STEPS}")
fi

coherent_args=()
if [[ "${COHERENT_ENDPOINTS:-1}" == "1" ]]; then
  coherent_args=(--coherent-endpoints)
fi

free_args=()
if [[ "${NO_FREE_ENDPOINTS:-0}" == "1" ]]; then
  free_args=(--no-free-endpoints)
fi

recompute_args=()
dat_args=(--dat "${DAT}")
if [[ "${QUENCH_ALLOY:-0}" == "1" ]]; then
  recompute_args=(--quench-alloy --alloy-fmax "${ALLOY_FMAX}" --alloy-relax-steps "${ALLOY_RELAX_STEPS}")
elif [[ "${RECOMPUTE_ALLOY:-0}" == "1" ]]; then
  recompute_args=(--recompute-alloy)
else
  if [[ ! -f "${DAT}" ]]; then
    echo "ERROR: DAT not found: ${DAT}. Set DAT=... or use RECOMPUTE_ALLOY=1." >&2
    exit 1
  fi
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: CONFIG not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${TRAJ}" ]]; then
  echo "ERROR: TRAJ not found: ${TRAJ}" >&2
  exit 1
fi

echo "Running binary alloy hull check"
echo "  RUN_DIR=${RUN_DIR}"
echo "  ELEMENTS=${ELEMENT_A} ${ELEMENT_B}"
echo "  CONFIG=${CONFIG}"
echo "  TRAJ=${TRAJ}"
echo "  DAT=${DAT}"
echo "  ENDPOINT_A=${ENDPOINT_A}"
echo "  ENDPOINT_B=${ENDPOINT_B}"
echo "  OUT_DIR=${OUT_DIR}"
echo "  COHERENT_ENDPOINTS=${COHERENT_ENDPOINTS:-1}"
echo "  RELAX_ENDPOINTS=${RELAX_ENDPOINTS:-0}"
echo "  RECOMPUTE_ALLOY=${RECOMPUTE_ALLOY:-0}"
echo "  QUENCH_ALLOY=${QUENCH_ALLOY:-0}"
echo "  MAX_FRAMES=${MAX_FRAMES:-}"
echo "  PROGRESS_INTERVAL=${PROGRESS_INTERVAL}"

python3 -u "${CODE_DIR}/examples/alloy/analysis/check_binary_alloy_hull.py" \
  --config "${CONFIG}" \
  --traj "${TRAJ}" \
  "${dat_args[@]}" \
  --elements "${ELEMENT_A}" "${ELEMENT_B}" \
  --endmember "${ELEMENT_A}" "${ENDPOINT_A}" \
  --endmember "${ELEMENT_B}" "${ENDPOINT_B}" \
  --formula-count-element "${FORMULA_COUNT_ELEMENT}" \
  --metals-per-formula "${METALS_PER_FORMULA}" \
  --temperature-K "${TEMPERATURE_K}" \
  --start "${START}" \
  "${stop_args[@]}" \
  --step "${STEP}" \
  "${max_frames_args[@]}" \
  --progress-interval "${PROGRESS_INTERVAL}" \
  --out-dir "${OUT_DIR}" \
  "${coherent_args[@]}" \
  "${free_args[@]}" \
  "${relax_args[@]}" \
  "${recompute_args[@]}"
