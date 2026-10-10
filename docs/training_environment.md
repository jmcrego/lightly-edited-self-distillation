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

## Activation

For a later interactive session, load the modules above and activate:

```bash
source /lustre/fsn1/projects/rech/eut/ujt99zo/josep/venv-train-311/bin/activate
```

The Slurm launcher loads the modules and activates this environment
automatically. To use a different environment:

```bash
export TRAIN_VENV=/path/to/training-venv
sbatch scripts/train_translation.slurm
```

## Job Verification

Submit from the repository root after updating the launcher input paths:

```bash
mkdir -p logs
LC_ALL=C LANG=C sbatch scripts/train_translation.slurm
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
