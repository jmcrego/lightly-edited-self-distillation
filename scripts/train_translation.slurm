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
#SBATCH --time=04:00:00
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
METHOD=sft
MODEL=/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it
TRAIN=data/roche.10K.postedited122b.full.shuf.trn.jsonl.train.seg-corrected.jsonl
DEV=data/roche.10K.postedited122b.full.shuf.dev.jsonl.train.seg-corrected.jsonl
TEST=""  # Optional: set a path to enable final test evaluation.
PROMPT=prompts/base_translation.txt
OUTPUT="training/${METHOD}-corrected-${SLURM_JOB_ID}"

srun --gres=gpu:1 python -u - <<'PY'
import torch
import transformers
import trl
import peft
import datasets
import sacrebleu

print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
print("Transformers:", transformers.__version__, "TRL:", trl.__version__,
      "PEFT:", peft.__version__, flush=True)
torch.cuda.init()
if torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one visible GPU")
probe = torch.empty(1, device="cuda:0")
print("GPU usable:", torch.cuda.get_device_name(0), flush=True)
PY

TEST_ARGS=()
if [[ -n "$TEST" ]]; then
  TEST_ARGS=(--test "$TEST")
fi

# Learning rate uses the script's method-specific default unless overridden.
srun --gres=gpu:1 python -u scripts/train_translation.py \
  --method "$METHOD" \
  --model "$MODEL" \
  --train "$TRAIN" \
  --dev "$DEV" \
  "${TEST_ARGS[@]}" \
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
