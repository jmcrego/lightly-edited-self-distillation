# Jean Zay Training Environment

Setup instructions recorded on 2026-10-10 for single-H100 TranslateGemma LoRA
training with `scripts/train_translation.py`. Creation and GPU functionality
must be verified on Jean Zay; they have not been verified locally.

## Environment

Use a separate environment from `venv-postedit-311-clean` to avoid changing the
dependencies of the working vLLM translation/postediting installation.

Default path used by `scripts/train_translation.slurm`:

```text
/lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311
```

## Creation

Run on Jean Zay:

```bash
module purge
module load arch/h100 python/3.11.5 cuda/12.8.0

python -m venv /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311
source /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311/bin/activate

python -m pip install --upgrade pip
python -m pip install \
  'transformers==4.57.6' 'trl==0.19.1' 'peft==0.17.1' \
  'accelerate>=1.4,<2' 'datasets>=3,<4' \
  'sacrebleu>=2.5,<3' sentencepiece

python -m pip check
```

PyTorch is installed as a dependency rather than explicitly pinned. A CUDA
build is required. Loading the CUDA module alone does not establish that the
installed PyTorch build can use the GPU.

The default attention backend is `sdpa`; `flash-attn` is not required unless
selecting `--attention-implementation flash_attention_2`.

## COMET Installation

COMET was subsequently installed in this same training environment. See the
[README installation commands](../README.md#training-environment) for the
verified NumPy, PyArrow, and Setuptools compatibility pins and import checks.
Package installation must not be performed while training jobs use the venv.

## Activation

For a later interactive session, load the modules above and activate:

```bash
source /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311/bin/activate
```

The Slurm launcher loads the modules and activates this environment
automatically. To use a different environment:

```bash
export TRAIN_VENV=/path/to/training-venv
sbatch scripts/train_translation.slurm sft-corrected
```

## Job Verification

Submit from the repository root after updating the launcher input paths:

```bash
mkdir -p logs
LC_ALL=C LANG=C sbatch scripts/train_translation.slurm sft-corrected
```

Before training, the launcher checks dependency imports, prints PyTorch,
Transformers, TRL, and PEFT versions, initializes CUDA, and allocates a tensor
on the single visible GPU. Results appear in `logs/train-JOBID.out`; failures
appear in `logs/train-JOBID.err`. Successful environment verification does not
guarantee that a full training run will fit in GPU memory.

## Shell Startup Troubleshooting

If the job reports `CondaError: Run 'conda init' before 'conda deactivate'`,
the batch shell may have inherited Conda environment variables without its
shell function. The launcher initializes the Conda Bash hook when an executable
is available, then deactivates inherited environments before module cleanup.
It does not run `conda init` or modify shell startup files.

Submit with `LC_ALL=C LANG=C` as above to avoid inheriting an unavailable
`en_US.UTF-8` locale. The launcher also sets these variables internally, but
that happens after Bash starts and cannot suppress an earlier startup warning.
If the Conda error persists, inspect the full job log to identify whether it
comes from a startup file or module hook before the launcher setup runs.

### CPO training

`train_translation.py` also supports contrastive preference optimization (CPO)
with the pinned TRL 0.19.1 environment. It combines a reference-free sigmoid
preference loss with supervised NLL on the chosen translations:

```bash
python scripts/train_translation.py --method cpo \
  --train data/train.tsv --dev data/dev.tsv \
  --prompt prompts/base_translation.txt --output training/cpo-corrected-base \
  --src_col 1 --chosen_col 4 --rejected_col 3 \
  --cpo-beta 0.1 --cpo-alpha 1.0
```

Training TSV columns are source/human/base/corrected; dev columns are
source/human. The [post-editing launchers](post_editing.md) automatically create
this four-column TSV after running either the 27B (one GPU) or 122B (four GPUs)
post-editor. Set `TRAIN` in `scripts/train_translation.slurm` to the resulting
`*.postedited27b.full.src-human-base-corrected.tsv` or
`*.postedited122b.full.src-human-base-corrected.tsv` path. For SFT on corrections,
select `--method sft --src_col 1 --tgt_col 4`; for CPO/DPO, prefer column 4 over
column 3 as shown above. Identical preference pairs and oversized examples are excluded.
CPO preserves the template's prompt tokens and Gemma end-of-turn token, masks
prompt tokens in both losses, and uses no reference policy or reference-score
precomputation. `--cpo-beta` scales the preference logits; `--cpo-alpha` weights
the chosen-response NLL (zero disables it). The default learning rate is 5e-6,
as for DPO. Greedy BLEU/chrF evaluation and best-adapter selection are unchanged.

### Selecting the SFT target

The first launcher argument selects the experiment; it is consumed before
remaining options are forwarded to Python:

```bash
sbatch scripts/train_translation.slurm sft-corrected
sbatch scripts/train_translation.slurm sft-human
sbatch scripts/train_translation.slurm sft-base
```

All three use source column 1 and target columns 4, 2, and 3 respectively,
with the same training settings. Outputs are saved under
`training/Roche-81k-EXPERIMENT-JOBID`. Missing or invalid experiment names
produce a usage error. Additional options follow the experiment name, e.g.
`sbatch scripts/train_translation.slurm sft-corrected --learning-rate 5e-5`.

### Selecting preference targets for CPO/DPO

Use `METHOD-CHOSEN-REJECTED` as the first argument. The chosen target comes
first and the rejected target second:

```bash
sbatch scripts/train_translation.slurm cpo-corrected-human
sbatch scripts/train_translation.slurm cpo-corrected-base
sbatch scripts/train_translation.slurm dpo-human-base
```

Both CPO and DPO accept all six ordered pairings of distinct targets from
`corrected`, `human`, and `base`. These map to columns 4, 2, and 3 respectively.
For example, `cpo-corrected-human` uses chosen column 4 and rejected column 2.
SFT shortcuts remain `sft-corrected`, `sft-human`, and `sft-base`.

Source defaults to column 1; override with a trailing `--src_col` if required.
Output directories retain the experiment name, e.g.
`training/Roche-81k-cpo-corrected-human-JOBID`. The launcher uses learning rate
`2e-5` for SFT and `5e-6` for CPO/DPO. Additional Python options follow the
experiment name:

```bash
sbatch scripts/train_translation.slurm cpo-corrected-human --cpo-beta 0.1 --cpo-alpha 1.0
sbatch scripts/train_translation.slurm dpo-corrected-base --learning-rate 2e-6
```
