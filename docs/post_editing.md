# Minimal Post-Editing on Jean Zay

The teacher sees the source, TranslateGemma translation, and human reference.
It infers a domain only when supported, corrects actual errors, and preserves
valid student wording. The reference is evidence, not a wording template.
Unchanged translations remain in the training dataset.

The default teacher candidate is `Qwen/Qwen3.5-122B-A10B-FP8`, the official
FP8 checkpoint of the multilingual Qwen3.5 mixture-of-experts model (122B total,
10B active parameters per token). Its size does not establish
that it is a better translator than TranslateGemma-12B: audit a pilot before
processing the whole corpus. Change the teacher with `--model` or
`POSTEDIT_MODEL`. Use an instruction/chat model that supports a system message
and JSON output; thinking is disabled through the chat template. Text-only
loading skips the vision encoder. vLLM reads weight quantization from the model
configuration; `dtype="bfloat16"` specifies the compute dtype, not conversion
of FP8 weights to BF16.

Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B-FP8),
[Qwen3.5 deployment recipe](https://recipes.vllm.ai/Qwen/Qwen3.5-122B-A10B),
[vLLM structured outputs](https://docs.vllm.ai/en/v0.18.1/examples/offline_inference/structured_outputs/),
[IDRIS H100 job examples](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf).

## Preparation

Run these installation commands from the repository root on the Jean Zay
frontend; no GPU allocation is needed for installation. Use a dedicated
environment in your project storage. Do not load the site `vllm` or
`pytorch-gpu` modules into this environment: pip installs the matching
dependencies. The available `vllm/0.7.1` module is too old for Qwen3.5.
GPU execution and driver compatibility must be checked on an allocated node.

```bash
module load arch/h100
module load python/3.11.5
module load cuda/12.8.0
python3 --version
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version; print("Base Python:", sys.executable)'

python3 -c 'import sqlite3; print("SQLite:", sqlite3.sqlite_version)'
python3 -m venv /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-postedit-311-clean
source /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-postedit-311-clean/bin/activate
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version; print("Venv Python:", sys.executable)'

python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements-postedit.txt
python -m pip check
ninja --version
python -c 'import sqlite3; print("SQLite:", sqlite3.sqlite_version)'
```

The Jean Zay launcher defaults to the shared model directory you located:

```text
/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3.5-27B-FP8/
```

No model download is needed. The compute job enables offline mode and activates
`venv-postedit-311-clean` directly. No `POSTEDIT_PYTHON` export is needed for
submission. For manual environment checks, use:

```bash
export POSTEDIT_PYTHON="/lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-postedit-311-clean/bin/python"
```

The launcher purges inherited modules, loads `arch/h100`, `python/3.11.5`, and
`cuda/12.8.0`, then explicitly activates the project venv. It runs the 27B-FP8
model on one H100 with tensor parallelism set to 1.

The launcher uses the portable `C` locale with Python
UTF-8 mode. If `_sqlite3` fails with `undefined symbol: sqlite3_deserialize`,
the runtime SQLite library lacks a symbol required by this Python build. The
`ldd` output for the extension identifies the resolved library.
Check that path before changing library search paths or rebuilding the venv.

The site `python/3.12.2` environment failed this check: its resolved SQLite
library did not export `sqlite3_deserialize`. The user verified that
`python/3.11.5` imports SQLite successfully (version 3.51.1), so preparation now
uses a separate `venv-postedit-311-clean` environment, verified with Python 3.11.
An earlier directory named `venv-postedit-py311` still contained Python 3.9;
directory names do not determine the interpreter version. Keep the original
`venv-postedit` intact; do not reuse a Python 3.12 venv with Python 3.11.
Run preparation in a fresh session without the old venv activated, and replace
any previously exported `POSTEDIT_PYTHON` with the path above.

FlashInfer invokes `ninja` to compile GPU kernels. It is included in
`requirements-postedit.txt`, and the launcher adds the selected Python's `bin`
directory to `PATH` through activation. Selecting a venv
Python alone does not expose the venv's command-line executables to workers.
For `FileNotFoundError: ... 'ninja'`, install it using that exact interpreter:

```bash
"$POSTEDIT_PYTHON" -m pip install 'ninja>=1.11'
export PATH="$(dirname "$POSTEDIT_PYTHON"):$PATH"
ninja --version
```

The installed PyTorch reports CUDA 12.8. The launcher therefore loads the
available `cuda/12.8.0` toolkit and sets `CUDA_HOME` from the resolved `nvcc`
path. FlashInfer needs that compiler for runtime kernel compilation; PyTorch's
CUDA runtime packages alone did not provide it. Verify the toolkit with:

```bash
module load cuda/12.8.0
command -v nvcc
nvcc --version
```

The launcher also exports `CUDA_PATH` and `FLASHINFER_NVCC` explicitly and uses
`logs/flashinfer-cuda128` as a separate FlashInfer workspace. This avoids reusing
the earlier cache containing commands for the nonexistent `/usr/local/cuda`.
The log prints the compiler path and version before inference. This kernel cache
is separate from the translation audit checkpoint and does not affect `--resume`.

Override the shared directory with `--model /path/to/model` in the job arguments.
Direct Python execution defaults to
the Hugging Face model ID, so pass `--model` explicitly when running inference
without the Jean Zay launcher.

For reproducible experiments, keep local model snapshots immutable and record
the shared snapshot version if available. For a Hub model instead, download a
specific commit and pass that commit using `--revision COMMIT`. Resume checks input hashes,
prompt, script, model selection, and generation settings; it cannot detect
replaced weights inside a local model directory.

## Validate and Run

From the repository root, validate all source/language alignment and print one
prompt without importing vLLM or loading a model:

```bash
python3 scripts/post_edit_translations.py --dry-run
```

Run a 100-sentence pilot with the 27B-FP8 model on one H100 80 GB GPU.
One Slurm task launches the engine; no GPU-allocation overrides are needed.
The project allocation is `eut`, so use `--account=eut@h100`.
Select any required site-specific QoS
at submission. Four hours is a starting wall-time budget, not a throughput
estimate; measure the pilot before scheduling the full corpus.

Create `logs/` in the submission directory before calling `sbatch`: Slurm opens
its log files before executing the script. This directory is ignored by Git.

```bash
mkdir -p logs
sbatch --account=eut@h100 scripts/post_edit_jean_zay.slurm \
  --limit 100 --output data/postedited.pilot.jsonl.gz
```

Review pilot edits for correctness and unnecessary rewriting. Then run the
complete corpus into a different output:

```bash
mkdir -p logs
sbatch --account=eut@h100 scripts/post_edit_jean_zay.slurm
```

Resume an interrupted full run with the same inputs, prompt, model, and options:

```bash
mkdir -p logs
sbatch --account=eut@h100 scripts/post_edit_jean_zay.slurm --resume
```

Use `--no-reference` for a source/student-only ablation. The default input names
match the existing paired files. `--synthetic`, `--references`, `--prompt`, and
`--output` accept alternative paths. `--limit` restricts inference but still
checks alignment throughout both files.

## Outputs and Validation

- `logs/postedit-<JOB_ID>.out`: Slurm stdout, progress, and summary.
- `logs/postedit-<JOB_ID>.err`: Slurm stderr, warnings, and errors.
- `data/postedited.10_data.jsonl.gz`: original source and target schema, with
  target `seg` replaced by the corrected synthetic translation. No teacher
  explanations enter the SFT targets. Original extra fields are preserved.
- `*.audit.jsonl`: durable checkpoint, one record per source line, containing
  the corrected training record, original translations, teacher domain labels,
  literal edit spans, brief reasons, raw JSON responses, and review flags.
- `*.manifest.json`: input hashes, prompt, script hash, and run settings.
- `*.stats.json`: changed/unchanged counts, percentages, and review counts.
- `*.failure.json`: rejected teacher response and source line, if inference
  stops on a validation failure. This diagnostic is never used as training data.

The output dataset is exported atomically after all selected rows finish.
Interrupted jobs resume from the audit checkpoint; only an incomplete final
journal line is discarded. Existing outputs are not overwritten without
`--resume`. A completed resume can re-export without loading a GPU model.

JSON-constrained decoding and local checks reject malformed, empty, truncated,
or inconsistent teacher outputs. A failed batch stops the run before committing
that batch, instead of silently using a reference or student fallback. Increase
token/context budgets for truncation, or inspect the prompt/model for invalid
edits. Changed generation settings require a new output path.

Edits above `--review-edit-fraction` (default 0.30) are flagged, not clipped or
automatically rejected. The fraction is punctuation-sensitive token changes
from sequence alignment divided by the longer token count; it is an audit
heuristic, not a semantic quality score. Teacher ambiguity flags also trigger
review. Flagged translations ARE included in the export: inspect them before
using the corpus for training. Domain labels and reasons are unverified teacher
judgments. Structural validation cannot establish translation correctness.

Local checks:

```bash
python3 -m unittest discover -s tests -v
```
