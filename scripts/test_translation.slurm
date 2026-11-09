#!/bin/bash
# Submit from the repository root after creating logs/:
#   mkdir -p logs
#   sbatch scripts/test_translation.slurm [translation options...]
# Runs all three adapters sequentially on one H100.
#SBATCH --job-name=transgemma-test
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/test-translation-%j.out
#SBATCH --error=logs/test-translation-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

INPUT=/linkhome/rech/genata01/ujt99zo/work/josep/lightly-edited-self-distillation/data/test17.en
CHECKPOINTS=(
  Roche-81k-sft-base-1638031
  Roche-81k-sft-corrected-1638029
  Roche-81k-sft-human-1638030
)
[[ -r "$INPUT" ]] || { echo "Input not readable: $INPUT" >&2; exit 1; }
for checkpoint in "${CHECKPOINTS[@]}"; do
  [[ -d "../training/$checkpoint/final-adapter" ]] || {
    echo "Adapter missing: ../training/$checkpoint/final-adapter" >&2
    exit 1
  }
done

export LC_ALL=C
export LANG=C
export PYTHONUTF8=1
# Match training's cleanup of Conda environments inherited by Slurm.
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

module purge
module load arch/h100
module load python/3.11.5
module load cuda/12.8.0
TRAIN_VENV="${TRAIN_VENV:-/lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311}"
if [[ ! -f "$TRAIN_VENV/bin/activate" ]]; then
  echo "Training environment missing: $TRAIN_VENV; set TRAIN_VENV before submitting." >&2
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

for checkpoint in "${CHECKPOINTS[@]}"; do
  OUTPUT="inference/test17-$checkpoint"
  printf 'Translating with %s; output: %s\n' "$checkpoint" "$OUTPUT"
  srun --gres=gpu:1 python -u scripts/test_translation.py \
    --input "$INPUT" \
    --model /lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it \
    --adapter "../training/$checkpoint/final-adapter" \
    --prompt prompts/base_translation.txt \
    --output "$OUTPUT" \
    --source-language en \
    --target-language fr \
    --max-length 2048 \
    --max-new-tokens 512 \
    --eval-batch-size 1 \
    --no-comet \
    "$@"
  printf 'Translations saved under %s\n' "$OUTPUT"
done
