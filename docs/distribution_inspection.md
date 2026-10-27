# Inspecting translations under the frozen TranslateGemma model

`scripts/inspect_translation_distribution.py` compares the `base`, `human`, and
`corrected` target fields in post-edited JSONL (optionally `.gz`). It loads the
original TranslateGemma-12B-IT checkpoint without adapters, disables gradients,
and performs no training or generation. Use the CUDA training environment from
[the training setup](training_environment.md), with Transformers 4.57.6,
Accelerate, and PyTorch. The Slurm launcher requests one H100 and uses
`TRAIN_VENV`, defaulting to the same environment as the training launcher.

## Running a pilot and full inspection

Submit from the repository root:

```bash
mkdir -p logs
sbatch scripts/inspect_translation_distribution.slurm \
  data/Roche-test.enfr.translategemma12b.postedited27b.full.jsonl.gz \
  --limit 100 --output-prefix data/Roche-test.27b.distribution-pilot
```

For the full file, omit `--limit` and use a different output prefix, or let the
script derive it by stripping `.gz` and `.jsonl`/`.json` and appending
`.distribution`:

```bash
sbatch scripts/inspect_translation_distribution.slurm \
  data/Roche-test.enfr.translategemma12b.postedited27b.full.jsonl.gz
```

Use `--model` to select the original local checkpoint and `--prompt` to select
exactly the translation template used for base generation. Defaults match the
current TranslateGemma translation launcher. No human reference or base
translation is included in the scoring prompt: each candidate is independently
scored given the source and language codes only.

Input is line-delimited JSON, with one target per record. If records have
multiple targets, select one using `--target-language fr`. Missing, null, or
empty candidate fields remain visible in the reports, and available candidates
are still scored. The script does not accept four-column training TSV input.

A GPU-free dry run validates records and computes edit distances, without
loading the model or tokenizer:

```bash
python3 scripts/inspect_translation_distribution.py \
  --input data/Roche-test.enfr.translategemma12b.postedited27b.full.jsonl.gz \
  --limit 10 --dry-run --output-prefix /tmp/roche-distribution-dry-run
```

## Reports

Each run produces three new files; existing report paths are rejected:

- `PREFIX.sentences.jsonl`: source and all three candidates, likelihoods,
  pairwise edit distances, NLL differences, and scoring statuses.
- `PREFIX.sentences.tsv`: the same measurements in a flat, headered table for
  sentence-by-sentence inspection. Embedded tabs/newlines use standard CSV
  quoting; this is an inspection report, not a training TSV.
- `PREFIX.summary.json`: aggregate scores, status counts, input/prompt provenance,
  and scoring conventions. The summary is also printed to the job log.

Reports are written after scoring completes. There is no checkpoint/resume;
start with a pilot to establish runtime. Slurm logs use
`logs/distribution-JOBID.out` and `.err`.

For all three pairs (base/human, base/corrected, human/corrected), edit scores
include character and punctuation-aware word-token Levenshtein distances.
Normalized distance divides by the longer candidate length. Character distance
preserves casing, punctuation, and whitespace; word distance uses regex tokens
and ignores whitespace differences. These are not model-token edit distances
or TER. Summary scores include unchanged counts, changed fractions, sentence
means/medians/percentiles, and pooled normalized distances.

## Likelihood convention and interpretation

Teacher forcing scores each actual candidate token under its preceding prompt
and candidate tokens. The model is in evaluation mode. Only candidate text
contributes to the score: prompt tokens and the end-of-turn token are excluded.
Candidate whitespace is preserved, and special tokens are not added when
encoding the candidate. The implementation uses Gemma's selected-logit output
to avoid retaining vocabulary logits for prompt positions; see the
[Transformers Gemma3 documentation](https://huggingface.co/docs/transformers/model_doc/gemma3).

Per candidate, the report includes:

- `tokens`: number of candidate tokens under the frozen model's tokenizer.
- `total_logprob`: sum of token log-probabilities, using natural logarithms.
- `nll`: negative total log-probability divided by candidate token count.
- `perplexity`: `exp(nll)`; numerically overflowing exponentials are null.

Lower NLL means the candidate is more predictable under the original model,
not necessarily more accurate. A positive `corrected_minus_base` NLL difference
means the correction is more surprising on average. Correct domain terms may
increase NLL: inspect such cases rather than rejecting them automatically.
Length normalization helps compare different translations but does not remove
all length or tokenization effects. These likelihood differences are not KL
divergence and do not measure model drift after fine-tuning.

The summary separates `all_available` scores from `complete_triplets` scores.
Compare variants using the latter, where the same records have all three
likelihoods. It reports both the mean sentence NLL and token-weighted corpus
NLL; corpus perplexity is the exponential of corpus NLL, not the average of
sentence perplexities. Paired NLL-difference summaries use complete triplets.

`--max-length` defaults to 4096 prompt-plus-candidate tokens. Longer candidates
are marked `context_budget_exceeded`, never truncated, and excluded from NLL
aggregates. Edit distances are still available. Check status counts and paired
sample sizes before interpreting aggregate comparisons.
