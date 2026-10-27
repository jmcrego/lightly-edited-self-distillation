#!/bin/bash
# Submit from the repository root; override the account with sbatch --account if needed.
# Create logs/ before submission: mkdir -p logs
#SBATCH --job-name=postedit-122b
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/postedit-122b-%j.out
#SBATCH --error=logs/postedit-122b-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

if [[ $# -eq 0 || -z "$1" ]]; then
  echo "Usage: sbatch $0 FILE [post-editing options...]" >&2
  exit 1
fi
FILE="$1"
shift
if [[ ! -f "$FILE" || ! -r "$FILE" ]]; then
  echo "Input file does not exist or is not readable: $FILE" >&2
  exit 1
fi
PREFIX="${FILE%.gz}"
PREFIX="${PREFIX%.jsonl}"
PREFIX="${PREFIX%.json}"
FOUT="${PREFIX}.postedited122b.full.jsonl.gz"
FALL="${PREFIX}.postedited122b.full.src-human-base-corrected.tsv"

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
echo "CUDA_HOME=$CUDA_HOME; FLASHINFER_NVCC=$FLASHINFER_NVCC"
"$FLASHINFER_NVCC" --version
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8

MODEL=/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3.5-122B-A10B-FP8

srun python -u scripts/post_edit_translations.py \
  --model "$MODEL" \
  --tensor-parallel-size 4 \
  --batch-size 32 \
  --synthetic "$FILE" \
  --output "$FOUT" \
  "$@"

gunzip -cd "$FOUT" | \
    jq -r '[.seg, .tgts[0].human, .tgts[0].base, .tgts[0].corrected] | select(all(.[]; . != null and . != "")) | @tsv' \
    > "$FALL"
