#!/usr/bin/env python3
"""Translate source/reference TSV with TranslateGemma for minimal post-editing."""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it"

if __package__:
    from .post_edit_translations import (
        atomic_json, digest, export_dataset, open_text, read_checkpoint,
    )
else:
    from post_edit_translations import (
        atomic_json, digest, export_dataset, open_text, read_checkpoint,
    )


def load_tsv(path, source_language, target_language, limit=None, skip_header=False):
    rows = []
    with open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            if skip_header and number == 1:
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{number}: expected exactly two TAB-separated fields")
            source, reference = fields
            if not source.strip() or not reference.strip():
                raise ValueError(f"{path}:{number}: empty source or human reference")
            if "\ufffd" in source or "\ufffd" in reference:
                raise ValueError(f"{path}:{number}: invalid replacement character")
            # Validate even rows beyond a pilot limit; never silently skip bad pairs.
            if limit is None or len(rows) < limit:
                rows.append({"language": source_language, "seg": source,
                             "tgts": [{"language": target_language, "human": reference}]})
    if not rows:
        raise ValueError("no input sentence pairs")
    return rows


def translation_messages(source, source_language, target_language):
    return [{"role": "user", "content": [{
        "type": "text", "source_lang_code": source_language,
        "target_lang_code": target_language, "text": source,
    }]}]


def gemma3_rope_compatibility(config):
    """Express Gemma3's nested RoPE settings in its equivalent legacy format."""
    text = config.get_text_config()
    if getattr(text, "model_type", None) != "gemma3_text":
        return config
    parameters = getattr(text, "rope_parameters", None)
    if not isinstance(parameters, dict) or "full_attention" not in parameters:
        return config
    if set(parameters) != {"full_attention", "sliding_attention"}:
        raise ValueError("unsupported Gemma3 nested RoPE layout")
    full, sliding = parameters["full_attention"], parameters["sliding_attention"]
    if (not isinstance(full, dict) or "rope_type" not in full
            or not isinstance(sliding, dict) or sliding.get("rope_type") != "default"
            or set(sliding) - {"rope_type", "rope_theta"}):
        raise ValueError("Gemma3 RoPE compatibility requires default sliding-attention RoPE")
    full_theta = full.get("rope_theta", getattr(text, "rope_theta", None))
    local_theta = sliding.get("rope_theta", getattr(text, "rope_local_base_freq", None))
    if full_theta is None or local_theta is None:
        raise ValueError("Gemma3 RoPE compatibility requires explicit full/local base frequencies")
    # Transformers 4 + vLLM 0.18.1 inserts rope_theta at the top of a nested
    # dictionary and then fails validation. Gemma3's legacy path uses the full
    # parameters globally and default RoPE + rope_local_base_freq locally.
    text.rope_parameters = deepcopy(full)
    text.rope_parameters["rope_theta"] = full_theta
    text.rope_theta = full_theta
    text.rope_local_base_freq = local_theta
    text.rope_scaling = deepcopy(text.rope_parameters)
    print("Applied Gemma3 RoPE compatibility: "
          f"full={text.rope_parameters}; sliding=default, theta={local_theta}", flush=True)
    return config


def generate_batch(llm, tokenizer, sampling, batch, args, start=0):
    prompts = []
    for offset, reference in enumerate(batch):
        token_ids = tokenizer.apply_chat_template(
            translation_messages(reference["seg"], args.source_language, args.target_language),
            chat_template=args.prompt.read_text(encoding="utf-8"),
            tokenize=True, add_generation_prompt=True)
        if len(token_ids) + args.max_new_tokens > args.max_model_len:
            raise ValueError(f"input row {start + offset + 1}: context budget exceeded; "
                             "increase --max-model-len (sources are never truncated)")
        # Token IDs avoid re-tokenizing the rendered template and duplicating BOS.
        prompts.append({"prompt_token_ids": token_ids})
    responses = llm.generate(prompts, sampling, use_tqdm=False)
    if len(responses) != len(batch):
        raise ValueError("model returned an unexpected number of translations")
    entries = []
    for offset, (reference, response) in enumerate(zip(batch, responses)):
        line = start + offset + 1
        if len(response.outputs) != 1:
            raise ValueError(f"input row {line}: expected one translation")
        answer = response.outputs[0]
        if answer.finish_reason != "stop":
            raise ValueError(f"input row {line}: generation ended with {answer.finish_reason!r}; "
                             "increase --max-new-tokens if truncated")
        if not isinstance(answer.text, str) or not answer.text.strip():
            raise ValueError(f"input row {line}: empty model translation")
        record = deepcopy(reference)
        record["tgts"][0]["base"] = answer.text.rstrip()
        entries.append({"line": line, "record": record})
    return entries


def check_token_budget_resume(previous, current):
    allowed = {"max_model_len", "max_new_tokens", "script_sha256"}
    if ({key: value for key, value in previous.items() if key not in allowed}
            != {key: value for key, value in current.items() if key not in allowed}):
        raise ValueError("token-budget resume cannot change input, model, prompt, or other settings")
    for key in ("max_model_len", "max_new_tokens"):
        if current[key] < previous[key]:
            raise ValueError(f"token-budget resume cannot decrease {key}")
    if all(current[key] == previous[key] for key in ("max_model_len", "max_new_tokens")):
        raise ValueError("token-budget resume requires increased token limits")


def run(args, references):
    output = args.output.resolve()
    reference_output = args.references_output.resolve() if args.references_output else None
    audit = Path(str(output) + ".audit.jsonl")
    manifest = Path(str(output) + ".manifest.json")
    paths = tuple(path for path in (output, reference_output, audit, manifest) if path is not None)
    if len(set(paths)) != len(paths) or args.input.resolve() in paths:
        raise ValueError("input, outputs, and checkpoint paths must be distinct")
    config = {"format_version": 4, "input_sha256": digest(args.input),
              "script_sha256": digest(Path(__file__)), "model": args.model,
              "revision": args.revision, "source_language": args.source_language,
              "target_language": args.target_language, "skip_header": args.skip_header,
              "limit": args.limit, "seed": args.seed, "temperature": 0,
              "max_model_len": args.max_model_len, "max_new_tokens": args.max_new_tokens,
              "tensor_parallel_size": args.tensor_parallel_size, "batch_size": args.batch_size,
              "gpu_memory_utilization": args.gpu_memory_utilization,
              "references_output": str(reference_output) if reference_output else None, "dtype": "bfloat16"}
    config["prompt"] = args.prompt.read_text(encoding="utf-8")
    if not args.resume and any(path.exists() for path in paths):
        raise ValueError("output/checkpoint exists; use --resume or a new --output")
    if args.resume and not manifest.exists():
        raise ValueError("--resume requires an existing manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    if reference_output:
        reference_output.parent.mkdir(parents=True, exist_ok=True)
    with open(audit, "a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another job is using this output checkpoint") from error
        previous_config = None
        if args.resume:
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous != config:
                if not getattr(args, "resume_with_larger_token_budget", False):
                    raise ValueError("resume configuration or input differs from checkpoint")
                check_token_budget_resume(previous, config)
                previous_config = previous
        else:
            if (manifest.exists() or output.exists()
                    or (reference_output and reference_output.exists()) or stream.tell()):
                raise ValueError("output/checkpoint was created by another job")
            atomic_json(manifest, config)
        completed = read_checkpoint(stream)
        if len(completed) > len(references):
            raise ValueError("checkpoint contains more rows than input")
        for index, entry in enumerate(completed):
            record = entry["record"]
            if (record["language"], record["seg"]) != (
                    references[index]["language"], references[index]["seg"]):
                raise ValueError("checkpoint source differs from input")
            targets = record.get("tgts", [])
            if (len(targets) != 1 or targets[0].get("language") != args.target_language
                    or not isinstance(targets[0].get("base"), str) or not targets[0]["base"].strip()):
                raise ValueError("invalid checkpoint translation")
            if targets[0].get("human") != references[index]["tgts"][0]["human"]:
                raise ValueError("checkpoint human reference differs from input")
        if previous_config is not None:
            history = Path(str(output) + ".resume-history.jsonl")
            with history.open("a", encoding="utf-8") as history_stream:
                history_stream.write(json.dumps({"completed_rows": len(completed),
                                                 "previous": previous_config, "next": config}) + "\n")
                history_stream.flush()
                os.fsync(history_stream.fileno())
            atomic_json(manifest, config)
            print(f"Preserving {len(completed)} completed translations; increasing token limits. "
                  f"Previous/current settings and script hashes recorded in {history}", flush=True)
        if len(completed) < len(references):
            from vllm import LLM, SamplingParams

            llm = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
                      hf_overrides=gemma3_rope_compatibility,
                      tensor_parallel_size=args.tensor_parallel_size,
                      distributed_executor_backend="mp", dtype="bfloat16",
                      max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                      gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
                      language_model_only=True, enable_prefix_caching=True,
                      generation_config="vllm")
            tokenizer = llm.get_tokenizer()
            sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=args.seed)
            for start in range(len(completed), len(references), args.batch_size):
                entries = generate_batch(llm, tokenizer, sampling,
                                         references[start:start + args.batch_size], args, start)
                for entry in entries:
                    stream.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
                completed.extend(entries)
                print(f"Checkpointed {len(completed)}/{len(references)} translations", flush=True)
        export_dataset(output, completed)
        if reference_output:
            export_dataset(reference_output, [{"record": reference} for reference in references])
    print(f"Student translations with human references: {output}")
    if reference_output:
        print(f"Separate human references: {reference_output}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source<TAB>reference TSV, optionally .gz")
    parser.add_argument("--output", type=Path, required=True, help="base translations, .jsonl.gz or .json.gz")
    parser.add_argument("--references-output", type=Path, help="optionally export a separate reference file")
    parser.add_argument("--source-language", required=True, help="TranslateGemma language code, e.g. en")
    parser.add_argument("--target-language", required=True, help="TranslateGemma language code, e.g. fr")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local directory")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/base_translation.txt",
                        help="editable Gemma chat template (Jinja)")
    parser.add_argument("--revision", help="model/tokenizer revision for reproducibility")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="translate first N pairs; validate the entire TSV")
    parser.add_argument("--skip-header", action="store_true", help="skip the first TSV line")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-with-larger-token-budget", action="store_true",
                        help="resume with increased token limits; records old/new settings and script hashes")
    parser.add_argument("--dry-run", action="store_true", help="validate TSV and show model input; no GPU needed")
    args = parser.parse_args()
    if args.resume_with_larger_token_budget:
        args.resume = True
    if args.source_language == args.target_language:
        parser.error("source and target languages must differ")
    if any(value <= 0 for value in (args.batch_size, args.tensor_parallel_size,
                                    args.max_model_len, args.max_new_tokens)):
        parser.error("batch, GPU, and token settings must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_new_tokens >= args.max_model_len or not 0 < args.gpu_memory_utilization < 1:
        parser.error("invalid context budget or GPU memory utilization")
    if not args.output.name.endswith(".gz"):
        parser.error("--output must end in .gz")
    if args.references_output and not args.references_output.name.endswith(".gz"):
        parser.error("--references-output must end in .gz")
    return args


def main():
    args = parse_args()
    try:
        references = load_tsv(args.input, args.source_language, args.target_language,
                              args.limit, args.skip_header)
        if not args.prompt.read_text(encoding="utf-8").strip():
            raise ValueError("empty translation prompt")
        if args.dry_run:
            print(f"Validated {len(references)} selected sentence pairs")
            print(f"Prompt template: {args.prompt}")
            print(json.dumps(translation_messages(references[0]["seg"], args.source_language,
                                                  args.target_language), ensure_ascii=False, indent=2))
        else:
            run(args, references)
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
