#!/usr/bin/env python3
"""Compare aligned gzip JSONL translations using light punctuation tokenization."""

import argparse
from difflib import SequenceMatcher
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys
import unicodedata


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
ROOT = Path(__file__).resolve().parents[1]


def tokenize(text, ignore_case_punct=False):
    if ignore_case_punct:
        text = "".join(" " if unicodedata.category(char).startswith("P") else char
                       for char in text).casefold()
    return TOKEN_PATTERN.findall(text)


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a JSON object")
                yield record
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error


def translations(record, path, line_number):
    if not isinstance(record.get("seg"), str):
        raise ValueError(f"{path}:{line_number}: missing source string 'seg'")
    targets = record.get("tgts")
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"{path}:{line_number}: missing translations 'tgts'")
    result = {}
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError(f"{path}:{line_number}: invalid translation")
        language = target.get("language")
        segment = target.get("seg")
        if not isinstance(language, str) or not isinstance(segment, str):
            raise ValueError(f"{path}:{line_number}: invalid translation language or seg")
        if language in result:
            raise ValueError(f"{path}:{line_number}: duplicate target language {language!r}")
        result[language] = segment
    return result


def compare(left_path, right_path, ignore_case_punct=False):
    stats = {"sentences": 0, "translation_pairs": 0, "left_tokens": 0,
             "right_tokens": 0, "equal_tokens": 0, "equal_lines": 0}
    pairs = zip_longest(read_records(left_path), read_records(right_path))
    for line_number, (left, right) in enumerate(pairs, 1):
        if left is None or right is None:
            raise ValueError(f"record counts differ at line {line_number}")
        left_targets = translations(left, left_path, line_number)
        right_targets = translations(right, right_path, line_number)
        if (left.get("language"), left["seg"]) != (right.get("language"), right["seg"]):
            raise ValueError(f"source sentences or languages differ at line {line_number}")
        if left_targets.keys() != right_targets.keys():
            raise ValueError(f"target languages differ at line {line_number}")
        stats["sentences"] += 1
        equal_line = True
        for language, segment in left_targets.items():
            left_tokens = tokenize(segment, ignore_case_punct)
            right_tokens = tokenize(right_targets[language], ignore_case_punct)
            equal_line = equal_line and left_tokens == right_tokens
            # Align within each translation; never match tokens across sentences.
            matches = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)
            stats["equal_tokens"] += sum(block.size for block in matches.get_matching_blocks())
            stats["left_tokens"] += len(left_tokens)
            stats["right_tokens"] += len(right_tokens)
            stats["translation_pairs"] += 1
        stats["equal_lines"] += int(equal_line)
    return stats


def percentage(count, total):
    return 100 * count / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", nargs="?", type=Path,
                        default=ROOT / "data/bitext.10_data.jsonl.gz")
    parser.add_argument("right", nargs="?", type=Path,
                        default=ROOT / "data/transgemma.10_data.jsonl.gz")
    parser.add_argument("--ignore-case-punct", action="store_true",
                        help="ignore case and Unicode punctuation in token and line statistics")
    args = parser.parse_args()
    try:
        stats = compare(args.left, args.right, args.ignore_case_punct)
    except (OSError, ValueError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Left:  {args.left}")
    print(f"Right: {args.right}")
    print(f"Source sentences: {stats['sentences']:,}")
    print(f"Translation pairs: {stats['translation_pairs']:,}")
    for label, count in (("Equal lines", stats["equal_lines"]),
                         ("Different lines", stats["sentences"] - stats["equal_lines"])):
        print(f"{label}: {count:,} ({percentage(count, stats['sentences']):.2f}%)")
    print("Line equality: identical token sequences in all target languages; whitespace ignored.")
    if args.ignore_case_punct:
        print("Tokens: case-folded with punctuation ignored; sequence alignment.")
    else:
        print("Tokens: words and separate punctuation; case-sensitive sequence alignment.")
    print(f"{'Metric':<24} {'Left':>22} {'Right':>22}")
    print(f"{'Total tokens':<24} {stats['left_tokens']:>22,} {stats['right_tokens']:>22,}")
    for label, different in (("Equal tokens", False), ("Different tokens", True)):
        cells = []
        for key in ("left_tokens", "right_tokens"):
            total = stats[key]
            count = total - stats["equal_tokens"] if different else stats["equal_tokens"]
            cells.append(f"{count:,} ({percentage(count, total):.2f}%)")
        print(f"{label:<24} {cells[0]:>22} {cells[1]:>22}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
