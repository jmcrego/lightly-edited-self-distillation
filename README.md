# Lightly Edited Self-Distillation

## Translation and post-editing

Generate base translations with [TranslateGemma 12B](docs/translation.md), then
post-edit them with either Qwen launcher below. Both launchers use the same
`scripts/post_edit_translations.py`, prompt, and input/output format.

| Launcher | Post-editing model | H100 GPUs | Tensor parallelism |
| --- | --- | --- | --- |
| `post_edit_translations_1GPU.slurm` | Qwen3.5-27B-FP8 | 1 | 1 |
| `post_edit_translations_4GPU.slurm` | Qwen3.5-122B-A10B-FP8 | 4 | 4 |

Submit from the repository root, passing the input JSONL file first:

```bash
mkdir -p logs
sbatch scripts/post_edit_translations_1GPU.slurm data/Roche-train.enfr.translategemma12b.jsonl.gz
# Or use the larger post-editor:
sbatch scripts/post_edit_translations_4GPU.slurm data/Roche-train.enfr.translategemma12b.jsonl.gz
```

Each prompt includes the source, base translation, and human target reference
(unless `--no-reference` is passed). Outputs stay alongside the input, with
`.jsonl.gz` replaced by `.postedited27b.full.jsonl.gz` or
`.postedited122b.full.jsonl.gz`. Each job also exports a matching
`.src-human-base-corrected.tsv` for training, so no manual conversion is needed.
Additional options follow the input path, for example `--resume`.

See [post-editing instructions](docs/post_editing.md) for output examples,
checkpointing, and pilot runs, and [training setup](docs/training_environment.md)
for using the exported TSV.

To inspect whether corrections stay close to the frozen TranslateGemma model,
see [distribution inspection](docs/distribution_inspection.md). The scorer
reports sentence-level edit distances and NLL for base/human/corrected
translations, plus aggregate comparisons, using a one-GPU launcher.

## Training Environment

See [Jean Zay training setup](docs/training_environment.md) for environment
creation and Slurm submission instructions.

To install COMET alongside SFT/DPO training dependencies in the training venv:

```bash
source /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311/bin/activate
python -m pip install \
  'unbabel-comet==2.2.7' \
  'numpy==1.26.4' \
  'pyarrow==20.0.0' \
  'setuptools>=77.0.3,<82'

python -m pip check
python -c "import numpy, pandas, pyarrow, torch, transformers, trl, peft, datasets; print('Training imports OK')"
python -c "import comet; print('COMET import OK')"
```

These import checks passed on Jean Zay after applying the pins above, as
reported on 2026-10-10. COMET requires NumPy below 2; the PyArrow pin avoids
the NumPy 2 requirement of the previously installed build. The Setuptools
upper bound retains `pkg_resources`, used by COMET's TorchMetrics dependency.
Do not modify the environment while training jobs are using it.

The shared XCOMET-XL checkpoint is:

```text
/lustre/fsmisc/dataset/HuggingFace_Models/Unbabel/XCOMET-XL/checkpoints/model.ckpt
```

Installing COMET does not automatically add scoring to
`scripts/train_translation.py`, which currently reports BLEU and chrF.
Model loading and scoring must still be verified, including availability of
the encoder/tokenizer when running offline.
