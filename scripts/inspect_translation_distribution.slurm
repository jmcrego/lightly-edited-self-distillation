#!/bin/bash
# Submit from the repository root after creating logs/.
# Uses the training environment and frozen original model, without adapters.
# Usage: sbatch scripts/inspect_translation_distribution.slurm FILE [scoring options...]
#SBATCH --job-name=transgemma-score
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/distribution-%j.out
#SBATCH --error=logs/distribution-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

if [[ $# -eq 0 || -z "$1" ]]; then
  echo "Usage: sbatch $0 FILE [scoring options...]" >&2
  exit 1
fi
FILE="$1"
shift
if [[ ! -f "$FILE" || ! -r "$FILE" ]]; then
  echo "Input file does not exist or is not readable: $FILE" >&2
  exit 1
fi

export LC_ALL=C
export LANG=C
export PYTHONUTF8=1
# Slurm may inherit Conda variables without its interactive shell function.
# Initialize the hook before module cleanup can call `conda deactivate`.
CONDA_COMMAND="${CONDA_EXE:-$(type -P conda || true)}"
if [[ -n "$CONDA_COMMAND" ]] && ! declare -F conda >/dev/null; then
  CONDA_HOOK="$("$CONDA_COMMAND" shell.bash hook)"
  set +u
  eval "$CONDA_HOOK"
  set -u
fi
if declare -F conda >/dev/null; then
  set +u
  while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate
  done
  set -u
fi
printf 'Loading training modules\n'
module purge
module load arch/h100
module load python/3.11.5
module load cuda/12.8.0

# Edit this path or export TRAIN_VENV before submitting.
TRAIN_VENV="${TRAIN_VENV:-/lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311}"
if [[ ! -f "$TRAIN_VENV/bin/activate" ]]; then
  printf 'Training environment not found: %s\nSet TRAIN_VENV to your training venv.\n' "$TRAIN_VENV" >&2
  exit 1
fi
source "$TRAIN_VENV/bin/activate"
NVCC_PATH="$(readlink -f "$(command -v nvcc)")"
export CUDA_HOME="$(dirname "$(dirname "$NVCC_PATH")")"
export CUDA_PATH="$CUDA_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

srun --gres=gpu:1 python -u scripts/inspect_translation_distribution.py \
  --input "$FILE" \
  "$@"
