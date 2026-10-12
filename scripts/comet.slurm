#!/bin/bash
# Usage: sbatch scripts/comet.slurm SRC TGT HYP
# Create logs/ before submission. Output: HYP.comet (COMET JSON scores).
# Cache facebook/xlm-roberta-xl config/tokenizer on a login node first.
#SBATCH --job-name=xcomet
#SBATCH --account=eut@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/comet-%j.out
#SBATCH --error=logs/comet-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ $# -ne 3 ]]; then
  printf 'Usage: sbatch scripts/comet.slurm SRC TGT HYP\n' >&2
  exit 1
fi
SRC="$1"
TGT="$2"
HYP="$3"
OUTPUT="${HYP}.comet"
for file in "$SRC" "$TGT" "$HYP"; do
  if [[ ! -r "$file" ]]; then
    printf 'Input file not readable: %s\n' "$file" >&2
    exit 1
  fi
done
if [[ -e "$OUTPUT" ]]; then
  printf 'Output already exists: %s\n' "$OUTPUT" >&2
  exit 1
fi

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
MODEL="${COMET_MODEL:-/lustre/fsmisc/dataset/HuggingFace_Models/Unbabel/XCOMET-XL/checkpoints/model.ckpt}"
BATCH_SIZE="${COMET_BATCH_SIZE:-4}"
if [[ ! -f "$MODEL" ]]; then
  printf 'COMET checkpoint not found: %s\n' "$MODEL" >&2
  exit 1
fi

srun --gres=gpu:1 python -u - "$SRC" "$TGT" "$HYP" <<'PY'
import sys
import torch
import comet

counts = []
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as stream:
        counts.append(sum(1 for _ in stream))
if not counts[0] or len(set(counts)) != 1:
    raise ValueError(f"Expected equal, nonzero source/reference/hypothesis counts; got {counts}")
torch.cuda.init()
if torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one visible GPU")
print(f"Scoring {counts[0]} aligned sentences on {torch.cuda.get_device_name(0)}", flush=True)
PY

srun --gres=gpu:1 comet-score \
  -s "$SRC" \
  -r "$TGT" \
  -t "$HYP" \
  --model "$MODEL" \
  --gpus 1 \
  --batch_size "$BATCH_SIZE" \
  --to_json "$OUTPUT"
printf 'COMET results: %s\n' "$OUTPUT"
