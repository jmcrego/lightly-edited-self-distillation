#!/usr/bin/env python3
"""Extract SFT pairs or DPO triplets from a postedited JSONL dataset.

Output keys are src/tgt and, when requested, reject. TSV uses the same order,
without a header. Selectors choose input fields independently of output names.
For example, --src tgts.human places the human translation in src.
Input and output support optional gzip compression (.gz suffix).
"""

import argparse
from contextlib import contextmanager
import gzip
import json
from pathlib import Path
import sys


SELECTORS = ("seg", "tgts.human", "tgts.base", "tgts.corrected")


@contextmanager
def open_text(path, mode):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode, encoding="utf-8") as stream:
        yield stream


def extract(record, selectors, target_language=None):
    target = None
    if any(selector.startswith("tgts.") for selector in selectors):
        targets = record.get("tgts", [])
        if not isinstance(targets, list) or any(not isinstance(t, dict) for t in targets):
            raise ValueError("tgts must be a list of objects")
        if target_language:
            targets = [t for t in targets if t.get("language") == target_language]
        if len(targets) > 1:
            raise ValueError("multiple matching targets; specify --target-language or remove duplicate targets")
        target = targets[0] if targets else {}
    values = [record.get("seg") if selector == "seg" else target.get(selector.split(".")[1])
              for selector in selectors]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        return None
    return values


def prepare(args):
    selectors = [args.src, args.tgt] + ([args.reject] if args.reject else [])
    keys = ["src", "tgt"] + (["reject"] if args.reject else [])
    output = args.output.resolve()
    if output == args.input.resolve():
        raise ValueError("input and output must differ")
    if output.exists():
        raise ValueError("output exists; choose a new output path")
    suffix = output.with_suffix("").suffix if output.suffix == ".gz" else output.suffix
    file_format = args.format or {".jsonl": "jsonl", ".tsv": "tsv"}.get(suffix)
    if file_format is None:
        raise ValueError("use a .jsonl/.tsv output suffix or specify --format")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(".tmp-" + output.name)
    counts = {"read": 0, "written": 0, "missing_or_empty": 0, "tsv_control_characters": 0}
    try:
        with open_text(args.input, "rt") as source, open_text(temporary, "wt") as destination:
            for line, text in enumerate(source, 1):
                counts["read"] += 1
                try:
                    record = json.loads(text)
                    if not isinstance(record, dict):
                        raise ValueError("expected a JSON object")
                    values = extract(record, selectors, args.target_language)
                except ValueError as error:
                    raise ValueError(f"{args.input}:{line}: {error}") from error
                if values is None:
                    counts["missing_or_empty"] += 1
                    continue
                if file_format == "tsv":
                    # The training reader expects literal tabs, not quoted CSV.
                    if any(any(char in value for char in "\t\r\n") for value in values):
                        counts["tsv_control_characters"] += 1
                        continue
                    destination.write("\t".join(values) + "\n")
                else:
                    destination.write(json.dumps(dict(zip(keys, values)), ensure_ascii=False) + "\n")
                counts["written"] += 1
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(output), **counts}, indent=2))
    return counts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Postedited JSONL file, optionally .gz")
    parser.add_argument("--output", type=Path, required=True, help="New JSONL or TSV file, optionally .gz")
    parser.add_argument("--format", choices=["jsonl", "tsv"], help="Output format; inferred from output suffix when omitted")
    parser.add_argument("--src", choices=SELECTORS, required=True, help="Input selector for output src / column 1")
    parser.add_argument("--tgt", choices=SELECTORS, required=True, help="Input selector for output tgt / column 2 (chosen for DPO)")
    parser.add_argument("--reject", choices=SELECTORS, help="Optional selector for output reject / column 3 (DPO)")
    parser.add_argument("--target-language", help="Language to select from tgts; unnecessary when there is only one target")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        prepare(parse_args())
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
