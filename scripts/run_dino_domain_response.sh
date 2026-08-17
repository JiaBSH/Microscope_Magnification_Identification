#!/bin/bash
#SBATCH --job-name=dino-domain-response
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/scvi576/run/JiaBSH/mmdetection_para/outputs/dino_window_supplement/02_domain_response/logs/slurm_%j.out
#SBATCH --error=/data/home/scvi576/run/JiaBSH/mmdetection_para/outputs/dino_window_supplement/02_domain_response/logs/slurm_%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load cuda/13.0
fi

CONDA_BASE="${CONDA_BASE:-/data/apps/miniforge/25.3.0-3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mmdetection_para}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/home/scvi576/run/JiaBSH/Microscope_Magnification_Identification_confusion}"
DATA_ROOT="${DATA_ROOT:-/data/home/scvi576/run/JiaBSH/mmdetection_para/data/syn_multimag/coco_rotation}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/home/scvi576/run/JiaBSH/mmdetection_para/outputs/dino_window_supplement/02_domain_response}"
HF_HOME="${HF_HOME:-/data/run01/scvi576/JiaBSH/.hf_cache}"
TMPDIR="${TMPDIR:-/data/run01/scvi576/JiaBSH/.tmp}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TMPDIR
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" "${MPLCONFIGDIR}" "${TMPDIR}"
cd "${PROJECT_ROOT}"

python -u -m rate_identification.domain_response \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-name vit_small_patch14_dinov2 \
  --resize-long-edge 1024 \
  --input-size 518 \
  --patch-size 14 \
  --seed 42 \
  --reuse-cache

