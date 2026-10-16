#!/usr/bin/env python3
"""Minimally post-edit tgts[].translategemma-12b with Qwen, preserving input fields.

The same-language tgts[].seg supplies the human reference. Results are added
alongside the student as translategemma-12b+qwen3.5-122b. Supports JSONL and
JSONL.gz, checkpointed resume, and GPU-free --dry-run.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

if __package__:
    from . import post_edit_translations as core
else:
    import post_edit_translations as core

STUDENT_FIELD = "translategemma-12b"
OUTPUT_FIELD = "translategemma-12b+qwen3.5-122b"


def canonical_pair(row):
    if not isinstance(row.get("tgts"), list):
        raise ValueError("tgts must be a list")
    references = {}
    students = {}
    for target in row["tgts"]:
        if not isinstance(target, dict) or not isinstance(target.get("language"), str):
            raise ValueError("each target requires a string language")
        language = target["language"]
        for field, mapping in (("seg", references), (STUDENT_FIELD, students)):
            if field in target:
                if language in mapping:
                    raise ValueError(f"duplicate {field} for {language}")
                if not isinstance(target[field], str) or not target[field].strip():
                    raise ValueError(f"empty or invalid {field} for {language}")
                mapping[language] = target[field]
        if OUTPUT_FIELD in target:
            raise ValueError(f"{OUTPUT_FIELD} already exists; use the original input for --resume")
    if not students:
        raise ValueError(f"no {STUDENT_FIELD} translations")
    student = {"language": row.get("language"), "seg": row.get("seg"), "tgts": []}
    reference = {"language": row.get("language"), "seg": row.get("seg"), "tgts": []}
    for language, text in students.items():
        if language not in references:
            raise ValueError(f"missing human reference seg for {language}")
        student["tgts"].append({"language": language, "student": text})
        reference["tgts"].append({"language": language, "human": references[language]})
    core.targets(student)
    core.targets(reference, "human")
    return student, reference


def load_pairs(path, limit=None):
    pairs = []
    for number, row in enumerate(core.records(path), 1):
        try:
            canonical_pair(row)
        except ValueError as error:
            raise ValueError(f"input line {number}: {error}") from error
        if limit is None or number <= limit:
            pairs.append((row, None))
    if not pairs:
        raise ValueError("no input records")
    return pairs


def generate_batch(llm, tokenizer, sampling, batch, prompt, args):
    entries = core.generate_batch(llm, tokenizer, sampling,
                                  [canonical_pair(row) for row, _ in batch], prompt, args)
    for entry, (row, _) in zip(entries, batch):
        corrections = {target["language"]: target["corrected"]
                       for target in entry["record"]["tgts"]}
        entry["record"] = deepcopy(row)
        for target in entry["record"]["tgts"]:
            if STUDENT_FIELD in target:
                target[OUTPUT_FIELD] = corrections[target["language"]]
    return entries


def parse_args(argv=None):
    # Reuse the inference CLI so generation and resume options stay consistent.
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--input", required=True, type=Path)
    known, remaining = parser.parse_known_args(argv)
    if any(arg == "--synthetic" or arg.startswith("--synthetic=") or
           arg == "--references" or arg.startswith("--references=") or
           arg == "--no-reference" for arg in remaining):
        parser.error("use --input; references are read from same-language tgts[].seg")
    saved = sys.argv
    try:
        sys.argv = [saved[0], "--synthetic", str(known.input), *remaining]
        args = core.parse_args()
    finally:
        sys.argv = saved
    return args


def main():
    # Let the shared parser show all supported inference flags on --help.
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print(__doc__ + "\nUsage: --input INPUT --output OUTPUT [inference options]\n")
        print("--input replaces --synthetic; --references and --no-reference are unsupported.\n")
        core.parse_args()
    args = parse_args()
    try:
        prompt = args.prompt.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError("empty system prompt")
        pairs = load_pairs(args.synthetic, args.limit)
        if args.dry_run:
            student, reference = canonical_pair(pairs[0][0])
            print(f"Validated {len(pairs):,} selected source lines")
            print(json.dumps(core.messages(student, reference,
                             student["tgts"][0]["language"], prompt), ensure_ascii=False, indent=2))
        else:
            core.run(args, pairs, prompt, batch_generator=generate_batch, driver_path=Path(__file__))
    except (OSError, ValueError, ImportError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
