#!/bin/bash
# Submit from the repository root after creating logs/.
# Use a separate training environment with the dependencies in train_translation.py.
# Override arguments by appending options to sbatch scripts/train_translation.slurm.
#SBATCH --job-name=transgemma-lora
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=18:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

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

# Experiment settings. Input files must already use src/tgt[/reject], not tgts.
MODEL=/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it
PROMPT=prompts/base_translation.txt
TRAIN=data/roche.50K.postedited122b.full.src-human-base-corrected.tsv
DEV=data/roche.test.enfr.gz

METHOD="sft"
OUTPUT="training/50K-${METHOD}-corrected-${SLURM_JOB_ID}"
COLS="--src_col 1 --tgt_col 4"

# METHOD="sft"
# OUTPUT="training/50K-${METHOD}-human-${SLURM_JOB_ID}"
# COLS="--src_col 1 --tgt_col 2"

### exp 3
# METHOD="dpo"
# OUTPUT="training/${METHOD}-corrected-base-${SLURM_JOB_ID}"
# COLS="--src_col 1 --chosen_col 4 --rejected_col 3"



# Learning rate uses the script's method-specific default unless overridden.
srun --gres=gpu:1 python -u scripts/train_translation.py \
  --method "$METHOD" \
  --model "$MODEL" \
  --train "$TRAIN" \
  $COLS \
  --dev "$DEV" \
  --prompt "$PROMPT" \
  --output "$OUTPUT" \
  --source-language en \
  --target-language fr \
  --epochs 1.0 \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 2048 \
  --max-new-tokens 512 \
  --eval-steps 100 \
  --eval-batch-size 1 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --dpo-beta 0.1 \
  --dpo-loss sigmoid \
  "$@"
