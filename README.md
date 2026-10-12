# Lightly Edited Self-Distillation

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
