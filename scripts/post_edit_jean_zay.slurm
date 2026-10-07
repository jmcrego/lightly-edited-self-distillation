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

export LC_ALL=C
export LANG=C
export PYTHONUTF8=1
module load arch/h100
module load python/3.11.5
# Activate your prepared CUDA-compatible environment before sbatch, or set
# POSTEDIT_PYTHON to its absolute Python path. No downloads happen in this job.
POSTEDIT_PYTHON="${POSTEDIT_PYTHON:-/lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-postedit-py311/bin/python}"
if ! "$POSTEDIT_PYTHON" -c 'import sys, sqlite3; print("Python:", sys.executable, "SQLite:", sqlite3.sqlite_version)'; then
    echo "SQLite import failed before inference. Resolved extension dependencies:" >&2
    SQLITE_EXTENSION="$("$POSTEDIT_PYTHON" -c 'import glob, sysconfig; print(next(iter(glob.glob(sysconfig.get_path("stdlib") + "/lib-dynload/_sqlite3*.so")), ""))')"
    if [[ -n "$SQLITE_EXTENSION" ]]; then
        ldd "$SQLITE_EXTENSION" >&2
    fi
    exit 1
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8

MODEL="${POSTEDIT_MODEL:-/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3.5-122B-A10B-FP8}"
# One Slurm task owns all four GPUs; vLLM launches tensor-parallel workers.
srun --ntasks=1 "$POSTEDIT_PYTHON" -u scripts/post_edit_translations.py \
    --model "$MODEL" --tensor-parallel-size 4 --batch-size 32 "$@"
