#!/bin/bash
# Submit from the repository root; create logs/ before submission.
# Override the account with sbatch --account if needed.
#SBATCH --job-name=transgemma12b
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/translate-%j.out
#SBATCH --error=logs/translate-%j.err

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
export CUDA_PATH="$CUDA_HOME"
export FLASHINFER_NVCC="$NVCC_PATH"
export FLASHINFER_WORKSPACE_BASE="$PWD/logs/flashinfer-cuda128"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8

MODEL=/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it
srun python -u scripts/translate_tsv.py \
  --input data/roche.random.10K.enfr.gz \
  --source-language en \
  --target-language fr \
  --model "$MODEL" \
  --prompt prompts/base_translation.txt \
  --tensor-parallel-size 1 \
  --batch-size 32 \
  --output data/roche.random.10K.transgemma12b.jsonl.gz \
  --references-output data/roche.random.10K.references.jsonl.gz \
  "$@"
