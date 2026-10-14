#!/bin/bash
# Usage: sbatch scripts/cometkiwi.slurm SRC HYP
# Create logs/ before submission. Outputs: HYP.cometkiwi (sentence/global scores)
# and HYP.cometkiwi.details (JSON sentence scores; no XCOMET error spans).
# Cache facebook/xlm-roberta-xxl config/tokenizer on a login node first.
#SBATCH --job-name=cometkiwi-xxl
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/cometkiwi-%j.out
#SBATCH --error=logs/cometkiwi-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ $# -ne 2 ]]; then
  printf 'Usage: sbatch scripts/cometkiwi.slurm SRC HYP\n' >&2
  exit 1
fi
SRC="$1"
HYP="$2"
OUTPUT="${HYP}.cometkiwi"
DETAILS="${OUTPUT}.details"
for file in "$SRC" "$HYP"; do
  if [[ ! -r "$file" ]]; then
    printf 'Input file not readable: %s\n' "$file" >&2
    exit 1
  fi
done
for file in "$OUTPUT" "$DETAILS"; do
  if [[ -e "$file" ]]; then
    printf 'Output already exists: %s\n' "$file" >&2
    exit 1
  fi
done

export LC_ALL=C
export LANG=C
export PYTHONUTF8=1
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
source "$TRAIN_VENV/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
MODEL="${COMETKIWI_MODEL:-/lustre/fsmisc/dataset/HuggingFace_Models/Unbabel/wmt23-cometkiwi-da-xxl/checkpoints/model.ckpt}"
BATCH_SIZE="${COMETKIWI_BATCH_SIZE:-1}"
if [[ ! -f "$MODEL" ]]; then
  printf 'COMET checkpoint not found: %s\n' "$MODEL" >&2
  exit 1
fi

srun --gres=gpu:1 python -u - "$SRC" "$HYP" <<'PY'
import sys
import torch
import comet
from transformers import AutoConfig, XLMRobertaTokenizerFast

# Check offline encoder assets before allocating the large metric model.
AutoConfig.from_pretrained("facebook/xlm-roberta-xxl", local_files_only=True)
XLMRobertaTokenizerFast.from_pretrained("facebook/xlm-roberta-xxl", local_files_only=True)

counts = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        counts.append(sum(1 for _ in stream))
if not counts[0] or len(set(counts)) != 1:
    raise ValueError(f"Expected equal, nonzero source/hypothesis counts; got {counts}")
torch.cuda.init()
if torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one visible GPU")
print(f"Scoring {counts[0]} aligned sentences on {torch.cuda.get_device_name(0)}", flush=True)
PY

srun --gres=gpu:1 comet-score \
  -s "$SRC" \
  -t "$HYP" \
  --model "$MODEL" \
  --gpus 1 \
  --batch_size "$BATCH_SIZE" \
  --to_json "$DETAILS" > "$OUTPUT"
printf 'CometKiwi scores: %s\nSentence details: %s\n' "$OUTPUT" "$DETAILS"
