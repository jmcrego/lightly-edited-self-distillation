# Domain corpora

Download JRC-Acquis, EMEA, KDE4, and ECB separately with OPUS Tools:

```bash
python3.11 -m venv .venv-opus
.venv-opus/bin/python -m pip install opustools
PYTHON=.venv-opus/bin/python bash scripts/download_corpora.sh en fr
```

Supply the actual source and target language codes. The optional third argument
changes the output directory (default: repository `corpora/`). Set `PYTHON` to use
a virtual environment. Downloads can be large; this command downloads all four
corpora, rather than sampling them.

For sentence pairs only, use `OPUS_FORMATS=moses` before the command. This skips
XML downloads, but document-based splitting requires the XML archives. Progress
updates are limited to once per second; archive validation can also take time
after the network transfer finishes.

Each `corpora/CORPUS/LANGUAGE-PAIR/manifest.json` records the resolved OPUS release,
language direction, OPUS Tools version, resource URLs, and SHA-256 checksums.
Reruns reuse the recorded release. Keep the manifest alongside the raw archives.
XML archives retain document identities and sentence alignment links; Moses
archives contain convenient parallel text. Archive integrity checks do not check
translation quality. An interrupted run can leave a partial archive; remove that
archive before retrying if validation fails.

The downloader only retrieves and verifies raw data. Prepare the experimental
datasets separately using this protocol:

1. Preserve each corpus separately, with its OPUS release and manifest. Extract
   records with corpus, document IDs on both sides, alignment IDs, source text,
   and reference translation. Retain provenance when concatenating aligned spans.
2. Filter empty pairs, incorrect languages, extreme length ratios, and corrupted
   alignments. Use a language identifier on both sides; record its model/version,
   confidence thresholds, length limits, ratio definition and threshold, and
   rejection counts per corpus. Check missing sentence IDs, invalid links,
   encoding errors, and malformed documents. Set thresholds before evaluation
   and inspect a sample of accepted and rejected records.
3. Identify exact and near duplicates within and across all four corpora before
   splitting. Record normalization and similarity rules. Group duplicate pairs
   and repeated source sentences, including those with different targets, so
   their variants cannot cross training, development, and test boundaries.
   Review near duplicates before removing examples that differ in meaning.
4. Create held-out document groups wherever document IDs are available. Connect
   documents sharing duplicate groups and assign each connected group wholly to
   one split, including across corpora. Otherwise split deduplicated sentence
   groups. Keep corpus labels so each domain has its own test set; record split
   sizes and any limitations caused by missing document metadata.
5. Use a fixed random seed (for example, `42`) and stable record ordering. Save
   the exact train/dev/test records, group assignments, preparation configuration,
   code revision, and checksums. Freeze test sets before the first model
   evaluation and never modify them after evaluating models. Select thresholds,
   training settings, and checkpoints using training/development data only.
6. Evaluate the base model and every fine-tuned model on all four frozen test
   sets. Use the same prompts, decoding settings, references, and metric versions
   for every model. Save translations and scores with model/checkpoint IDs and
   test-set checksums. Report a model-by-corpus table, including in-domain gains
   and changes in the other domains. Keep test sources and references out of
   training, teacher post-editing data, and retrieval examples.

OPUS Tools interface: https://github.com/Helsinki-NLP/OpusTools/blob/master/opustools_pkg/README.md

## Prepare the downloaded data

```bash
.venv-opus/bin/python -m pip install -r requirements-corpora.txt
.venv-opus/bin/python scripts/prepare_corpora.py en fr
```

The default output is `corpora/prepared/`. The command refuses to reuse any
existing output directory, including an incomplete run. Use `--output` with a
new directory when changing preparation settings before evaluation. Once models
have been evaluated, keep the original tests and manifests unchanged.

Run a pilot first to inspect accepted/rejected examples and group sizes:

```bash
.venv-opus/bin/python scripts/prepare_corpora.py en fr \
  --limit-per-corpus 1000 --output corpora/prepared-pilot
```

Defaults: seed 42, train/dev/test fractions 90/5/5, 3-2000 characters per side,
maximum longer/shorter character ratio 3, and minimum language confidence 0.6.
Lingua checks both sides against all its supported languages in high-accuracy
mode. Wrong or uncertain languages are rejected, including ambiguous short UI
labels; inspect the rejection audit before deciding whether to adjust thresholds.
`--language-mode low` saves resources but reduces short-text detection accuracy.

Exact deduplication uses Unicode NFKC, case folding, and whitespace normalization,
preserving original accepted text. Identical pairs are removed within each corpus;
identical sources (even with different targets) are grouped across all corpora.
Cross-corpus copies retain their domain labels but always receive the same split.
Near duplicates use source character 5-grams, 64-permutation MinHash LSH, and an
exact Jaccard check at 0.85. This approximate candidate search can miss matches;
it does not detect semantic paraphrases. Near matches are grouped rather than
deleted, and their IDs and similarities are saved for inspection.

Moses XML/IDS metadata supplies source/target document IDs and alignment IDs.
The pipeline checks text/metadata row counts, invalid links, replacement characters,
control characters, and lengths. It cannot check sentence IDs against the original
document XML or detect every semantically incorrect alignment from Moses text alone.
For the current downloads, JRC-Acquis and KDE4 have XML links, EMEA has no document
metadata, and ECB uses a single document ID. Corpora with no metadata or a single
source document use sentence-group splitting, recorded in `report.json`.
`--sentence-corpora JRC-Acquis KDE4` explicitly selects sentence splitting if
document-connected components prevent useful held-out sets; this weakens document
independence and should only be decided before evaluation.

Documents sharing duplicate sources are connected before splitting. Large
components can prevent the requested fractions or produce an empty split. The
report records actual counts and the largest components; any empty split makes
`manifest.json` report `evaluation_ready: false`. Pilots are also never marked
evaluation-ready. Inspect these results before using the tests.

Outputs:

- `CORPUS.train.jsonl.gz`, `.dev.jsonl.gz`, and `.test.jsonl.gz`: source text in
  `seg`, references in `tgts[].human`, language codes, and provenance.
  Existing prepared files with references in `tgts[].seg` or
  `tgts[].human_reference` remain readable.
- `rejections.jsonl.gz`: rejected texts, reasons, metadata, and language checks.
- `duplicate_links.jsonl.gz`: exact-source and verified near-source connections.
- `assignments.jsonl.gz`: input row IDs, connected groups, splits, and retention flags.
- `config.json`, `report.json`, `manifest.json`: settings, counts, dependency versions,
  code/script identity, input/output checksums, and readiness status.
- `preparation.sqlite`: disk-backed accepted rows and duplicate-search index, retained
  for inspecting group members and matching audit IDs to original texts.

The pipeline creates data splits; model generation and scoring are separate.
Evaluate every model on the same four `*.test.jsonl.gz` files and save each model's
predictions and metric settings with the test manifest.
