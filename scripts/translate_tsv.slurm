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
# Check the GPU inside an allocated step before opening the translation checkpoint.
srun --gres=gpu:1 python -u - <<'PY'
import os
import socket
import subprocess
import torch

print("GPU preflight on", socket.gethostname(), flush=True)
for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES"):
    print(f"{name}={os.environ.get(name, '<unset>')}", flush=True)
subprocess.run(["nvidia-smi"], check=True)
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True)
torch.cuda.init()
probe = torch.empty(1, device="cuda:0")
print("GPU usable:", torch.cuda.get_device_name(0), flush=True)
PY

srun --gres=gpu:1 python -u scripts/translate_tsv.py \
  --input data/roche.random.10K.enfr.gz \
  --source-language en \
  --target-language fr \
  --model "$MODEL" \
  --prompt prompts/base_translation.txt \
  --tensor-parallel-size 1 \
  --batch-size 32 \
  --output data/roche.random.10K.transgemma12b.jsonl.gz \
  "$@"
