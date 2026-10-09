#!/usr/bin/env python3
"""Filter OPUS Moses archives and create immutable, leakage-aware domain splits."""

import argparse
from array import array
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
import gzip
import hashlib
import importlib.metadata
import io
from itertools import islice, zip_longest
import json
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ("JRC-Acquis", "EMEA", "KDE4", "ECB")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def shingles(text):
    # Character shingles also work for languages without whitespace tokenization.
    return {text[i:i + 5] for i in range(max(1, len(text) - 4))}


def jaccard(left, right):
    return len(left & right) / len(left | right)


class Groups:
    def __init__(self):
        self.parent = array("q")
        self.size = array("q")

    def add(self):
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def xml_links(stream):
    documents = None
    root = None
    for event, element in ET.iterparse(stream, events=("start", "end")):
        tag = element.tag.rsplit("}", 1)[-1]
        if root is None:
            root = element
        if event == "start" and tag == "linkGrp":
            documents = (element.get("fromDoc"), element.get("toDoc"))
        elif event == "end" and tag == "link":
            sides = element.get("xtargets", "").split(";")
            # opus_read/Moses omit null alignments; they occupy no text row.
            if len(sides) == 2 and not all(side.strip() for side in sides):
                element.clear()
                continue
            valid = (documents and all(documents) and len(sides) == 2
                     and all(side.strip() for side in sides))
            yield {"documents": documents, "sentence_ids": sides,
                   "valid": bool(valid)}
            element.clear()
        elif event == "end" and tag == "linkGrp":
            element.clear()
            root.clear()
            documents = None


def id_links(stream):
    for line in io.TextIOWrapper(stream, encoding="utf-8", errors="strict"):
        fields = line.rstrip("\r\n").split("\t")
        yield {"documents": fields[:2], "sentence_ids": fields[2:4],
               "valid": len(fields) == 4 and all(field.strip() for field in fields)}


def archive_members(archive, source, target):
    def one(suffix, required=False):
        matches = [name for name in archive.namelist() if name.endswith(suffix)]
        if len(matches) > 1 or (required and len(matches) != 1):
            raise ValueError(f"Expected one archive member ending in {suffix}: {matches}")
        return matches[0] if matches else None
    return one("." + source, True), one("." + target, True), one(".xml"), one(".ids")


def input_records(path, source, target, limit=0):
    with zipfile.ZipFile(path) as archive, ExitStack() as stack:
        src_name, tgt_name, xml_name, ids_name = archive_members(archive, source, target)
        src = stack.enter_context(io.TextIOWrapper(archive.open(src_name),
                                                   encoding="utf-8", errors="replace"))
        tgt = stack.enter_context(io.TextIOWrapper(archive.open(tgt_name),
                                                   encoding="utf-8", errors="replace"))
        metadata = None
        if xml_name:
            metadata = xml_links(stack.enter_context(archive.open(xml_name)))
        elif ids_name:
            metadata = id_links(stack.enter_context(archive.open(ids_name)))
        streams = (src, tgt, metadata) if metadata is not None else (src, tgt)
        rows = zip_longest(*streams)
        if limit:
            rows = islice(rows, limit)
        for number, row in enumerate(rows, 1):
            if any(value is None for value in row):
                raise ValueError(f"{path}: text/alignment row counts differ at {number}")
            link = row[2] if metadata is not None else None
            if link and source > target:
                # OPUS alignment metadata follows the alphabetically sorted pair.
                link["documents"] = tuple(reversed(link["documents"]))
                link["sentence_ids"] = list(reversed(link["sentence_ids"]))
            yield number, row[0].strip(), row[1].strip(), link


def basic_rejection(source, target, metadata, args):
    if not source or not target:
        return "empty"
    if any("\ufffd" in text or any(unicodedata.category(c) == "Cc" and c != "\t"
                                  for c in text) for text in (source, target)):
        return "corrupt_text"
    if metadata and not metadata["valid"]:
        return "invalid_alignment"
    if max(len(source), len(target)) > args.max_chars:
        return "too_long"
    if min(len(source), len(target)) < args.min_chars:
        return "too_short"
    if max(len(source), len(target)) / min(len(source), len(target)) > args.max_ratio:
        return "length_ratio"
    if not any(c.isalpha() for c in source) or not any(c.isalpha() for c in target):
        return "no_letters"
    return None


def language_checker(args):
    from lingua import LanguageDetectorBuilder

    builder = LanguageDetectorBuilder.from_all_languages()
    if args.language_mode == "low":
        builder = builder.with_low_accuracy_mode()
    detector = builder.build()

    @lru_cache(maxsize=50000)
    def check(text):
        values = detector.compute_language_confidence_values(text)
        if not values:
            return None, 0.0
        return values[0].language.iso_code_639_1.name.lower(), values[0].value

    return check


def connect_documents(db, groups, sentence_corpora, report):
    for corpus in CORPORA:
        docs = db.execute("SELECT COUNT(DISTINCT src_doc) FROM records WHERE corpus=?",
                          (corpus,)).fetchone()[0]
        mode = "document" if docs > 1 and corpus not in sentence_corpora else "sentence"
        report[corpus]["split_unit"] = mode
        report[corpus]["source_document_count"] = docs
        if mode == "sentence":
            report[corpus]["limitation"] = (
                "Explicit sentence fallback" if corpus in sentence_corpora else
                "No document metadata or only one source document; document holdout unavailable")
            continue
        first = {}
        for index, src_doc, tgt_doc in db.execute(
                "SELECT id,src_doc,tgt_doc FROM records WHERE corpus=? ORDER BY id", (corpus,)):
            for side, document in (("source", src_doc), ("target", tgt_doc)):
                if document:
                    key = (side, document)
                    if key in first:
                        groups.union(index, first[key])
                    else:
                        first[key] = index


def connect_duplicates(db, groups, args, audit):
    from datasketch import MinHash, MinHashLSH

    # Store band buckets on disk rather than retaining millions of Python sets.
    try:
        layout = MinHashLSH(threshold=args.near_threshold, num_perm=args.num_perm)
    except ValueError as error:
        if "bands are too small" not in str(error):
            raise
        layout = MinHashLSH(threshold=args.near_threshold, num_perm=args.num_perm,
                            params=(2, args.num_perm // 2))
    ranges = layout.hashranges
    db.executescript("""
        CREATE TABLE sources (text TEXT PRIMARY KEY, record_id INTEGER NOT NULL);
        CREATE TABLE buckets (band INTEGER, signature BLOB, record_id INTEGER);
        CREATE INDEX bucket_lookup ON buckets (band, signature);
    """)
    counts = Counter()
    for index, normalized in db.execute("SELECT id,norm_source FROM records ORDER BY id"):
        previous = db.execute("SELECT record_id FROM sources WHERE text=?",
                              (normalized,)).fetchone()
        if previous:
            groups.union(index, previous[0])
            counts["exact_source_links"] += 1
            audit.write(json.dumps({"kind": "exact_source", "left": previous[0],
                                    "right": index}) + "\n")
            continue
        db.execute("INSERT INTO sources VALUES (?,?)", (normalized, index))
        if len(normalized) < args.near_min_chars:
            continue
        tokens = shingles(normalized)
        sketch = MinHash(num_perm=args.num_perm, seed=args.seed)
        sketch.update_batch([token.encode("utf-8") for token in sorted(tokens)])
        bands = [sketch.hashvalues[start:end].tobytes() for start, end in ranges]
        candidates = set()
        for band, signature in enumerate(bands):
            candidates.update(row[0] for row in db.execute(
                "SELECT record_id FROM buckets WHERE band=? AND signature=?", (band, signature)))
        for candidate in sorted(candidates):
            other = db.execute("SELECT norm_source FROM records WHERE id=?",
                               (candidate,)).fetchone()[0]
            score = jaccard(tokens, shingles(other))
            if score >= args.near_threshold:
                groups.union(index, candidate)
                counts["near_source_links"] += 1
                audit.write(json.dumps({"kind": "near_source", "left": candidate,
                                        "right": index, "jaccard": score}) + "\n")
        db.executemany("INSERT INTO buckets VALUES (?,?,?)",
                       ((band, signature, index) for band, signature in enumerate(bands)))
        if (index + 1) % 10000 == 0:
            print(f"Duplicate search: {index + 1:,} accepted records", flush=True)
    db.commit()
    return dict(counts)


def assign_splits(db, groups, args):
    components = defaultdict(Counter)
    for index, corpus in db.execute("SELECT id,corpus FROM records WHERE kept=1 ORDER BY id"):
        components[groups.find(index)][corpus] += 1
    totals = Counter()
    for sizes in components.values():
        totals.update(sizes)
    rng = random.Random(args.seed)
    order = list(components)
    rng.shuffle(order)
    # Allocate large connected groups first, using seed-shuffled order to break ties.
    order.sort(key=lambda key: sum(components[key].values()), reverse=True)
    fractions = {"train": 1 - args.dev_fraction - args.test_fraction,
                 "dev": args.dev_fraction, "test": args.test_fraction}
    counts = {split: Counter() for split in fractions}
    assignments = {}
    for group in order:
        sizes = components[group]
        def cost(split):
            # Increment in squared deviation from each domain's target size.
            return sum(((counts[split][corpus] + size - totals[corpus] * fractions[split]) ** 2
                        - (counts[split][corpus] - totals[corpus] * fractions[split]) ** 2)
                       / max(1, totals[corpus]) for corpus, size in sizes.items())
        split = min(fractions, key=cost)
        assignments[group] = split
        counts[split].update(sizes)
    return assignments, counts, components


def prepare(args):
    if args.output.exists():
        raise ValueError(f"Output already exists; frozen splits cannot be overwritten: {args.output}")
    check_language = language_checker(args)
    pair = "-".join(sorted((args.source, args.target)))
    inputs = []
    for corpus in CORPORA:
        folder = args.input / corpus / pair
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest["source"], manifest["target"]) != (args.source, args.target):
            raise ValueError(f"Language direction mismatch: {manifest_path}")
        files = [item for item in manifest["resources"] if item["preprocessing"] == "moses"]
        if len(files) != 1:
            raise ValueError(f"Expected one recorded Moses archive: {manifest_path}")
        archive = folder / files[0]["file"]
        actual_hash = checksum(archive)
        if actual_hash != files[0]["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        inputs.append((corpus, archive, manifest, actual_hash))
    # Exclusive directory creation protects frozen tests and incomplete runs alike.
    args.output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    write_json(args.output / "config.json", config)
    db = sqlite3.connect(args.output / "preparation.sqlite")
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE records (
            id INTEGER PRIMARY KEY, corpus TEXT, line INTEGER, source TEXT, target TEXT,
            norm_source TEXT, norm_target TEXT, src_doc TEXT, tgt_doc TEXT,
            metadata TEXT, language_checks TEXT, kept INTEGER DEFAULT 1);
        CREATE INDEX exact_pair ON records(corpus,norm_source,norm_target);
        CREATE TABLE assignments (record_id INTEGER PRIMARY KEY, group_id INTEGER, split TEXT);
    """)
    groups = Groups()
    report = {corpus: {"counts": Counter()} for corpus in CORPORA}
    try:
        with gzip.open(args.output / "rejections.jsonl.gz", "wt", encoding="utf-8") as rejected:
            for corpus, archive, manifest, archive_hash in inputs:
                stats = report[corpus]
                stats["release"] = manifest["release"]
                stats["archive"] = str(archive.resolve())
                stats["archive_sha256"] = archive_hash
                stats["download_manifest_sha256"] = checksum(archive.parent / "manifest.json")
                print(f"Filtering {corpus} ({manifest['release']})...", flush=True)
                for line, source, target, metadata in input_records(
                        archive, args.source, args.target, args.limit_per_corpus):
                    stats["counts"]["input"] += 1
                    reason = basic_rejection(source, target, metadata, args)
                    checks = {}
                    if reason is None:
                        for side, text, expected in (("source", source, args.source),
                                                     ("target", target, args.target)):
                            detected, confidence = check_language(text)
                            checks[side] = {"detected": detected, "confidence": confidence}
                            if confidence < args.language_confidence:
                                reason = side + "_uncertain_language"
                                break
                            if detected != expected:
                                reason = side + "_wrong_language"
                                break
                    if reason:
                        stats["counts"][reason] += 1
                        rejected.write(json.dumps({"corpus": corpus, "line": line,
                                                   "reason": reason, "source": source,
                                                   "target": target, "metadata": metadata,
                                                   "language_checks": checks},
                                                  ensure_ascii=False) + "\n")
                        continue
                    index = groups.add()
                    documents = metadata["documents"] if metadata else (None, None)
                    norm_source, norm_target = normalize(source), normalize(target)
                    duplicate = db.execute(
                        "SELECT id FROM records WHERE corpus=? AND norm_source=? AND norm_target=?",
                        (corpus, norm_source, norm_target)).fetchone()
                    db.execute("INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                               (index, corpus, line, source, target, norm_source, norm_target,
                                *documents, json.dumps(metadata), json.dumps(checks),
                                int(duplicate is None)))
                    stats["counts"]["accepted_before_dedup"] += 1
                    if duplicate:
                        stats["counts"]["exact_pair_removed"] += 1
                    else:
                        stats["counts"]["retained"] += 1
                    if line % 10000 == 0:
                        print(f"  {line:,} rows; {stats['counts']['retained']:,} retained", flush=True)
                db.commit()
        connect_documents(db, groups, set(args.sentence_corpora), report)
        with gzip.open(args.output / "duplicate_links.jsonl.gz", "wt", encoding="utf-8") as audit:
            duplicate_stats = connect_duplicates(db, groups, args, audit)
        assignments, counts, components = assign_splits(db, groups, args)
        with ExitStack() as stack:
            handles = {(corpus, split): stack.enter_context(gzip.open(
                args.output / f"{corpus}.{split}.jsonl.gz", "wt", encoding="utf-8"))
                for corpus in CORPORA for split in ("train", "dev", "test")}
            assignment_file = stack.enter_context(gzip.open(
                args.output / "assignments.jsonl.gz", "wt", encoding="utf-8"))
            for index, corpus, line, source, target, metadata, checks, kept in db.execute(
                    "SELECT id,corpus,line,source,target,metadata,language_checks,kept "
                    "FROM records ORDER BY id"):
                group = groups.find(index)
                split = assignments.get(group)
                db.execute("INSERT INTO assignments VALUES (?,?,?)", (index, group, split))
                assignment_file.write(json.dumps({"id": index, "corpus": corpus, "line": line,
                                                   "group": group, "split": split,
                                                   "kept": bool(kept)}) + "\n")
                if not kept:
                    continue
                record = {"id": f"{corpus}:{line}", "seg": source, "language": args.source,
                          "tgts": [{"human": target, "language": args.target}],
                          "corpus": corpus, "release": report[corpus]["release"],
                          "group": group, "alignment": json.loads(metadata),
                          "language_checks": json.loads(checks)}
                handles[corpus, split].write(json.dumps(record, ensure_ascii=False) + "\n")
        db.commit()
        warnings = []
        for corpus in CORPORA:
            report[corpus]["splits"] = {split: counts[split][corpus] for split in counts}
            total = sum(counts[split][corpus] for split in counts)
            report[corpus]["split_fractions"] = {
                split: counts[split][corpus] / total if total else 0.0 for split in counts}
            if any(counts[split][corpus] == 0 for split in counts):
                warnings.append(f"{corpus} has an empty split; inspect connected groups before evaluation")
        largest = sorted((sum(sizes.values()) for sizes in components.values()), reverse=True)[:10]
        write_json(args.output / "report.json", {"corpora": report, "duplicates": duplicate_stats,
                                                "largest_groups": largest, "warnings": warnings})
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                  capture_output=True, text=True)
        files = {path.name: checksum(path) for path in sorted(args.output.iterdir())
                 if path.suffix in (".gz", ".json")}
        write_json(args.output / "manifest.json", {
            "complete": True, "evaluation_ready": not warnings and not args.limit_per_corpus,
            "pilot": bool(args.limit_per_corpus), "seed": args.seed,
            "code_revision": revision.stdout.strip() or None,
            "script_sha256": checksum(Path(__file__)),
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("lingua-language-detector", "datasketch", "numpy", "scipy")},
            "python_version": sys.version, "files": files,
            "near_duplicates": "Source character-5-gram MinHash LSH candidates verified with exact "
                               "Jaccard; approximate search can miss similar pairs. Grouped, not removed.",
            "exact_duplicates": "NFKC/casefold/whitespace pair duplicates removed within each corpus; "
                                "source duplicates grouped globally, preserving domain labels.",
        })
        print(json.dumps({corpus: report[corpus]["splits"] for corpus in CORPORA}, indent=2))
        for warning in warnings:
            print("WARNING: " + warning, file=sys.stderr)
        print(f"Saved frozen splits and audit records in {args.output}", flush=True)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--input", type=Path, default=ROOT / "corpora")
    parser.add_argument("--output", type=Path, default=ROOT / "corpora/prepared")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--max-ratio", type=float, default=3.0)
    parser.add_argument("--language-confidence", type=float, default=0.6)
    parser.add_argument("--language-mode", choices=("high", "low"), default="high")
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--near-min-chars", type=int, default=30)
    parser.add_argument("--num-perm", type=int, default=64)
    parser.add_argument("--sentence-corpora", nargs="*", choices=CORPORA, default=[])
    parser.add_argument("--limit-per-corpus", type=int, default=0, help="pilot only; 0 processes all rows")
    args = parser.parse_args()
    if args.source == args.target:
        parser.error("source and target must differ")
    if not (0 < args.dev_fraction < 1 and 0 < args.test_fraction < 1
            and args.dev_fraction + args.test_fraction < 1):
        parser.error("dev/test fractions must be positive and sum to less than 1")
    if not (0 < args.near_threshold < 1 and 0 <= args.language_confidence <= 1):
        parser.error("invalid similarity or confidence threshold")
    if not (0 < args.min_chars <= args.max_chars and args.max_ratio >= 1
            and args.near_min_chars >= 5 and args.num_perm >= 16
            and args.limit_per_corpus >= 0 and 0 <= args.seed < 2 ** 32):
        parser.error("invalid length, MinHash, seed, or pilot settings")
    try:
        prepare(args)
    except (ValueError, OSError, ImportError, ET.ParseError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
