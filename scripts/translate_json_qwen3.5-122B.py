#!/usr/bin/env python3
"""Translate JSON/JSONL sources with Qwen3.5-122B-A10B-FP8.

Only top-level seg is sent to the model. Existing fields are preserved; output
translations are stored in tgts[].Qwen3.5-122B-A10B-FP8 for the target language.
Uses the checkpoint's native chat template with thinking disabled. Output and
resume/skip behavior are shared with translate_tsv.py. Requires four H100 GPUs
by default and the existing vLLM postediting environment.
"""

import json
from copy import deepcopy
from pathlib import Path

if __package__:
    from .translate_tsv import ROOT, open_text, parse_args, run
else:
    from translate_tsv import ROOT, open_text, parse_args, run


OUTPUT_FIELD = "Qwen3.5-122B-A10B-FP8"


def load_json(path, args):
    with open_text(path, "rt") as stream:
        try:
            values = json.load(stream)
        except json.JSONDecodeError:
            stream.seek(0)
            values = [json.loads(line) for line in stream]
        else:
            values = values if isinstance(values, list) else [values]
    records = []
    for index, original in enumerate(values, 1):
        if not isinstance(original, dict) or not isinstance(original.get("seg"), str) or not original["seg"].strip():
            raise ValueError(f"record {index}: expected a nonempty seg source")
        if original.get("language") != args.source_language:
            raise ValueError(f"record {index}: source language differs from --source-language")
        record = deepcopy(original)
        targets = record.setdefault("tgts", [])
        if not isinstance(targets, list) or any(not isinstance(t, dict) for t in targets):
            raise ValueError(f"record {index}: tgts must be a list of objects")
        matches = [t for t in targets if t.get("language") == args.target_language]
        if len(matches) > 1:
            raise ValueError(f"record {index}: multiple targets for {args.target_language}")
        if matches and OUTPUT_FIELD in matches[0]:
            raise ValueError(f"record {index}: {OUTPUT_FIELD} already exists")
        if not matches:
            targets.append({"language": args.target_language})
        if args.limit is None or index <= args.limit:
            records.append(record)
    if not records:
        raise ValueError("input contains no source records")
    return records


def qwen_messages(source, args):
    instruction = args.prompt.read_text(encoding="utf-8").format(
        source_language=args.source_language, target_language=args.target_language)
    return [{"role": "system", "content": instruction}, {"role": "user", "content": source}]


def prompt_builder(tokenizer, source, args):
    return tokenizer.apply_chat_template(qwen_messages(source, args), tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)


def main():
    args = parse_args(defaults={
        "model": "/lustre/fsmisc/dataset/HuggingFace_Models/Qwen/Qwen3.5-122B-A10B-FP8",
        "prompt": ROOT / "prompts/qwen3.5-122B.txt",
        "tensor_parallel_size": 4, "max_model_len": 4096,
    }, description=__doc__, prompt_help="Translation instruction with {source_language}/{target_language} placeholders",
        input_help="JSON object, JSON array, or JSONL with language/seg/tgts; optionally .gz", require_gzip=False)
    try:
        if args.skip_header or args.references_output:
            raise ValueError("--skip-header and --references-output are not supported for JSON input")
        references = load_json(args.input, args)
        if not args.prompt.read_text(encoding="utf-8").strip():
            raise ValueError("empty translation prompt")
        if args.dry_run:
            print(f"Validated {len(references)} selected source records")
            print(json.dumps(qwen_messages(references[0]["seg"], args), ensure_ascii=False, indent=2))
        else:
            run(args, references, prompt_builder=prompt_builder,
                engine_overrides={"gdn_prefill_backend": "triton"}, driver_path=Path(__file__),
                output_field=OUTPUT_FIELD)
    except (ValueError, KeyError, OSError, ImportError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
