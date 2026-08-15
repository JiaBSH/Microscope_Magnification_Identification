#!/bin/bash
#SBATCH --job-name=scale-confusion
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/data/home/scvi576/run/JiaBSH/mmdetection_para/outputs/dino_window_supplement/01_scale_ablation/logs/slurm_%j.out
#SBATCH --error=/data/home/scvi576/run/JiaBSH/mmdetection_para/outputs/dino_window_supplement/01_scale_ablation/logs/slurm_%j.err

set -euo pipefail

# Some Slurm compute nodes do not initialize Environment Modules for batch
# shells. The conda environment already ships the CUDA runtime required by
# PyTorch, so loading the cluster module is optional rather than a hard
# requirement.
if command -v module >/dev/null 2>&1; then
  module purge 2>/dev/null || true
  module load cuda/13.0
fi

CONDA_BASE="${CONDA_BASE:-/data/apps/miniforge/25.3.0-3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mmdetection_para}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/home/scvi576/run/JiaBSH/Microscope_Magnification_Identification_confusion}"
DATA_ROOT="${DATA_ROOT:-/data/home/scvi576/run/JiaBSH/mmdetection_para/data/syn_multimag/coco_rotation}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/home/scvi576/run/JiaBSH/mmdetection_para/outputs/dino_window_supplement/01_scale_ablation}"
TORCH_HOME="${TORCH_HOME:-/data/home/scvi576/run/JiaBSH/.torch_cache}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TORCH_HOME
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export MPLCONFIGDIR="${OUTPUT_DIR}/.matplotlib"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs" "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"

python -u -m rate_identification.rotation_confusion \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods raw_pca handcrafted_pca dino_no_pca dino_pca \
  --pca-components 32 \
  --knn-neighbors 5 \
  --cluster-count 6 \
  --resize-long-edge 1024 \
  --splits val test \
  --seed 42 \
  --reuse-cache
