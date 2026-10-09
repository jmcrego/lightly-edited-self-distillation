# Translate TSV pairs before post-editing

`scripts/translate_tsv.py` reads UTF-8 `source<TAB>human_reference` pairs, one
pair per line. It accepts plain TSV or gzip-compressed TSV; use `--skip-header`
if the file has a header. Fields are literal text, not CSV-quoted: quotes are
preserved, and embedded tabs or multiline fields are not supported. Empty fields
and malformed rows stop the run instead of silently changing alignment.

Only source text and the source/target language codes enter the model prompt.
Human references are embedded in the output and never shown to the translation
model. A separate reference export is optional. The editable prompt is
`prompts/base_translation.txt`, a Jinja chat
template with Gemma turn markers and structured `source_lang_code` and
`target_lang_code` fields. It replaces the model's bundled instruction text;
use `--prompt` to choose another template. The default model is the shared
Jean Zay directory `/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it`.

On a GPU node with the existing post-editing venv activated:

```bash
python -u scripts/translate_tsv.py \
  --input data/domain.en-fr.tsv \
  --source-language en --target-language fr \
  --model /lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it \
  --output data/domain.student.jsonl.gz
```

Use the actual local model directory on Jean Zay, or a Hugging Face model ID
where Hub access is available. The installed `requirements-postedit.txt`
environment supplies vLLM. The default tensor parallelism is 1, appropriate
for starting with one H100 for this 12B model. The script does not submit a job
or allocate GPUs; run it inside your GPU allocation.

Validate the TSV and inspect the source-only payload without loading a model:

```bash
python scripts/translate_tsv.py \
  --input data/domain.en-fr.tsv \
  --source-language en --target-language fr \
  --output data/domain.student.jsonl.gz --dry-run
```

For a pilot, add `--limit 100` and choose a separate pilot output. The entire TSV
is still validated. Default batch size is 32, context limit 2048, generation
limit 512, seed 42, and temperature 0. Inputs are never truncated; empty or
truncated generations stop the run. Increase the token limits as needed within
the model's supported context length.

Outputs use gzip JSONL, compatible with the post-editor:

```json
{"language":"en","seg":"Hello.","tgts":[{"language":"fr","seg":"Bonjour.","human_reference":"Salut."}]}
```

The target's `seg` contains the generated translation; `human_reference` contains
the original second TSV column. The post-editor automatically reads the embedded
reference. Use `--references-output` only if you also want a separate reference file.
The script writes `OUTPUT.audit.jsonl` after each successful batch and records
input hashes, prompt contents, and model/settings in `OUTPUT.manifest.json`. Existing outputs
require `--resume`; resuming requires the same input, script, model, and options.
Pin `--revision` for a Hub model, or keep local model snapshots immutable.

Feed the self-contained file to the correction job:

```bash
sbatch scripts/post_edit_translations.slurm \
  --synthetic data/domain.student.jsonl.gz \
  --output data/domain.corrected.jsonl.gz
```

TranslateGemma input contract:
https://huggingface.co/google/translategemma-12b-it#usage

## Jean Zay submission

The launcher requests one H100 GPU on one node and activates the existing
`venv-postedit-311-clean` environment in offline mode. It defaults to the Roche
10K TSV, the shared TranslateGemma-12B model, and `prompts/base_translation.txt`.
Submit from the repository root:

```bash
mkdir -p logs
sbatch scripts/translate_tsv.slurm
```

Arguments after the script override its Python defaults. Use separate paths
for a pilot so its outputs do not conflict with the full run:

```bash
sbatch scripts/translate_tsv.slurm --limit 100 \
  --output data/roche.pilot.transgemma12b.jsonl.gz
```

Resume a full run with `sbatch scripts/translate_tsv.slurm --resume`. Logs are
`logs/translate-JOBID.out` and `.err`. The initial time limit is four hours;
adjust it based on pilot throughput and your allocation's limits.
