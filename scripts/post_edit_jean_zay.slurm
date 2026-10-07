#!/bin/bash
# Submit from the repository root; override the account with sbatch --account if needed.
# Create logs/ before submission: mkdir -p logs
#SBATCH --job-name=min-postedit
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/postedit-%j.out
#SBATCH --error=logs/postedit-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

export LC_ALL=C
export LANG=C
export PYTHONUTF8=1
module purge
module load arch/h100
module load python/3.11.5
module load cuda/12.8.0

source /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-postedit-311-clean/bin/activate
NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8

MODEL=/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3.5-27B-FP8
srun python -u scripts/post_edit_translations.py --model "$MODEL" --tensor-parallel-size 1 --batch-size 32 "$@"
