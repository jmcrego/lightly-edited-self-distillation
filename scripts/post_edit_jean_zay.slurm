#!/bin/bash
# Submit from the repository root; override the account with sbatch --account if needed.
# Create logs/ before submission: mkdir -p logs
#SBATCH --job-name=minimal-postedit
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/postedit-%j.out
#SBATCH --error=logs/postedit-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

module load arch/h100
# Activate your prepared CUDA-compatible environment before sbatch, or set
# POSTEDIT_PYTHON to its absolute Python path. No downloads happen in this job.
POSTEDIT_PYTHON="${POSTEDIT_PYTHON:-python3}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8

MODEL="${POSTEDIT_MODEL:-/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3.5-122B-A10B-FP8}"
# One Slurm task owns all four GPUs; vLLM launches tensor-parallel workers.
srun --ntasks=1 "$POSTEDIT_PYTHON" -u scripts/post_edit_translations.py \
    --model "$MODEL" --tensor-parallel-size 4 --batch-size 32 "$@"
