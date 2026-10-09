#!/usr/bin/env python3
"""Minimally post-edit aligned translation JSONL using an offline vLLM teacher."""

import argparse
from copy import deepcopy
from difflib import SequenceMatcher
import fcntl
import gzip
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["meaning", "omission", "addition", "terminology", "grammar", "naturalness", "formatting"]
RESULT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["corrected_translation", "edits"],
    "properties": {
        "corrected_translation": {"type": "string", "minLength": 1},
        "edits": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["before", "after", "category", "reason"],
            "properties": {
                "before": {"type": "string"}, "after": {"type": "string"},
                "category": {"type": "string", "enum": CATEGORIES},
                "reason": {"type": "string", "minLength": 1},
            },
        }},
    },
}


class TeacherOutputError(ValueError):
    def __init__(self, message, index, language, original, answer):
        super().__init__(message)
        self.details = {"batch_record": index + 1, "language": language,
                        "original_translation": original, "raw_response": answer.text,
                        "finish_reason": answer.finish_reason, "error": message}


def open_text(path, mode="rt"):
    opener = gzip.open if str(path).endswith(".gz") else open
    return opener(path, mode, encoding="utf-8")


def records(path):
    with open_text(path) as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                yield row
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def targets(row):
    if not isinstance(row.get("language"), str) or not isinstance(row.get("seg"), str):
        raise ValueError("source requires string language and seg")
    if not row["seg"].strip():
        raise ValueError("empty source")
    if not isinstance(row.get("tgts"), list) or not row["tgts"]:
        raise ValueError("expected nonempty tgts list")
    result = {}
    for target in row["tgts"]:
        if not isinstance(target, dict) or not isinstance(target.get("language"), str):
            raise ValueError("invalid target language")
        if not isinstance(target.get("seg"), str) or not target["seg"].strip():
            raise ValueError("expected nonempty target seg")
        if target["language"] in result:
            raise ValueError("duplicate target language")
        result[target["language"]] = target["seg"]
    return result


def load_pairs(synthetic, references=None, limit=None, use_embedded_reference=True):
    student_rows = records(synthetic)
    reference_rows = records(references) if references else None
    pairs = zip_longest(student_rows, reference_rows) if references else ((r, None) for r in student_rows)
    result = []
    for number, (student, reference) in enumerate(pairs, 1):
        try:
            if student is None or (references and reference is None):
                raise ValueError("input record counts differ")
            student_targets = targets(student)
            if references is None and use_embedded_reference:
                embedded = [target.get("human_reference") for target in student["tgts"]]
                if any("human_reference" in target for target in student["tgts"]):
                    if any(not isinstance(text, str) or not text.strip() for text in embedded):
                        raise ValueError("missing or empty embedded human reference")
                    reference = {"language": student["language"], "seg": student["seg"],
                                 "tgts": [{"language": target["language"], "seg": text}
                                          for target, text in zip(student["tgts"], embedded)]}
            if reference is not None:
                reference_targets = targets(reference)
                if (student["language"], student["seg"]) != (reference["language"], reference["seg"]):
                    raise ValueError("source or source language mismatch")
                if student_targets.keys() != reference_targets.keys():
                    raise ValueError("target language mismatch")
            if limit is None or number <= limit:
                result.append((student, reference))
        except ValueError as error:
            raise ValueError(f"input line {number}: {error}") from error
    if not result:
        raise ValueError("no input records")
    return result


def messages(student, reference, language, system_prompt):
    payload = {"source_language": student["language"], "target_language": language,
               "source": student["seg"], "student_translation": targets(student)[language]}
    if reference is not None:
        payload["human_reference"] = targets(reference)[language]
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def validate_result(text, original):
    result = json.loads(text)
    if not isinstance(result, dict) or set(result) != set(RESULT_SCHEMA["required"]):
        raise ValueError("unexpected teacher output fields")
    corrected = result["corrected_translation"]
    if not isinstance(corrected, str) or not corrected.strip():
        raise ValueError("empty or invalid corrected translation")
    if not isinstance(result["edits"], list):
        raise ValueError("invalid edits list")
    if (corrected != original) != bool(result["edits"]):
        raise ValueError("translation change and edits list disagree")
    for edit in result["edits"]:
        if not isinstance(edit, dict) or set(edit) != {"before", "after", "category", "reason"}:
            raise ValueError("invalid edit fields")
        if not all(isinstance(value, str) for value in edit.values()):
            raise ValueError("invalid edit strings")
        if edit["category"] not in CATEGORIES or not edit["reason"].strip():
            raise ValueError("invalid edit category or reason")
        if edit["before"] == edit["after"]:
            raise ValueError("edit does not change text")
        if edit["before"] not in original or edit["after"] not in corrected:
            raise ValueError("edit spans do not occur in translations")
    return result


def edit_fraction(original, corrected):
    left = re.findall(r"\w+|[^\w\s]", original)
    right = re.findall(r"\w+|[^\w\s]", corrected)
    alignment = SequenceMatcher(None, left, right, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in alignment.get_opcodes()
                  if tag != "equal")
    return changed / max(len(left), len(right), 1)


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def read_checkpoint(stream):
    """Discard only an unterminated final record left by an interrupted write."""
    stream.seek(0)
    completed = []
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line:
            break
        if not line.endswith(b"\n"):
            stream.seek(offset)
            stream.truncate()
            break
        entry = json.loads(line)
        if entry["line"] != len(completed) + 1:
            raise ValueError("nonsequential audit checkpoint")
        completed.append(entry)
    stream.seek(0, os.SEEK_END)
    return completed


def export_dataset(path, completed):
    # Keep .gz as the last suffix so the temporary file is also compressed.
    temporary = path.with_name(".tmp-" + path.name)
    with open_text(temporary, "wt") as stream:
        for entry in completed:
            stream.write(json.dumps(entry["record"], ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def generate_batch(llm, tokenizer, sampling, batch, prompt, args):
    jobs = []
    for index, (student, reference) in enumerate(batch):
        for language, original in targets(student).items():
            text = tokenizer.apply_chat_template(
                messages(student, reference, language, prompt), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            if len(tokenizer.encode(text, add_special_tokens=False)) + args.max_new_tokens > args.max_model_len:
                raise ValueError(f"batch record {index + 1}, {language}: context budget exceeded; "
                                 "increase --max-model-len (input is never truncated)")
            jobs.append((index, language, original, text))
    outputs = llm.generate([{"prompt_token_ids": tokenizer.encode(job[3], add_special_tokens=False)}
                            for job in jobs], sampling, use_tqdm=False)
    if len(outputs) != len(jobs):
        raise ValueError("teacher returned an unexpected number of outputs")
    entries = [{"record": deepcopy(student), "targets": []} for student, _ in batch]
    for (index, language, original, _), output in zip(jobs, outputs):
        answer = output.outputs[0]
        if answer.finish_reason != "stop":
            raise TeacherOutputError(
                f"teacher output did not finish normally ({answer.finish_reason}); "
                "increase --max-new-tokens if truncated", index, language, original, answer)
        try:
            result = validate_result(answer.text, original)
        except ValueError as error:
            raise TeacherOutputError(
                f"batch record {index + 1}, {language}: invalid teacher result: {error}",
                index, language, original, answer) from error
        fraction = edit_fraction(original, result["corrected_translation"])
        result.update({"language": language, "original_translation": original,
                       "edit_fraction": fraction, "large_edit": fraction > args.review_edit_fraction,
                       "raw_response": answer.text})
        entries[index]["targets"].append(result)
        for target in entries[index]["record"]["tgts"]:
            if target["language"] == language:
                target["seg"] = result["corrected_translation"]
    return entries


def run(args, pairs, prompt):
    output = args.output.resolve()
    audit = Path(str(output) + ".audit.jsonl")
    manifest = Path(str(output) + ".manifest.json")
    summary_path = Path(str(output) + ".stats.json")
    config = {"format_version": 1, "synthetic_sha256": digest(args.synthetic),
              "script_sha256": digest(Path(__file__)),
              "references_sha256": digest(args.references) if args.references else None,
              "prompt": prompt, "model": args.model, "revision": args.revision,
              "limit": args.limit, "max_new_tokens": args.max_new_tokens,
              "max_model_len": args.max_model_len, "seed": args.seed,
              "review_edit_fraction": args.review_edit_fraction, "temperature": 0,
              "tensor_parallel_size": args.tensor_parallel_size, "dtype": "bfloat16",
              "quantization": "from_model_config", "language_model_only": True,
              "enable_thinking": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(p.exists() for p in (output, audit, manifest, summary_path)):
        raise ValueError("output/checkpoint exists; use --resume or a new --output")
    if args.resume and not manifest.exists():
        raise ValueError("--resume requires an existing manifest")
    with open(audit, "a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another job is using this output checkpoint") from error
        if args.resume:
            if json.loads(manifest.read_text(encoding="utf-8")) != config:
                raise ValueError("resume configuration/input/prompt differs from checkpoint")
        else:
            if manifest.exists() or output.exists() or stream.tell():
                raise ValueError("output/checkpoint was created by another job; use --resume")
            atomic_json(manifest, config)
        completed = read_checkpoint(stream)
        if len(completed) > len(pairs):
            raise ValueError("checkpoint contains more rows than input")
        if len(completed) < len(pairs):
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import StructuredOutputsParams

            llm = LLM(model=args.model, revision=args.revision, tokenizer_revision=args.revision,
                      tensor_parallel_size=args.tensor_parallel_size,
                      distributed_executor_backend="mp", dtype="bfloat16",
                      max_model_len=args.max_model_len, max_num_seqs=args.batch_size,
                      gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
                      language_model_only=True,
                      gdn_prefill_backend="triton",
                      enable_prefix_caching=True, generation_config="vllm")
            tokenizer = llm.get_tokenizer()
            sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
                                      seed=args.seed,
                                      structured_outputs=StructuredOutputsParams(json=RESULT_SCHEMA))
            for start in range(len(completed), len(pairs), args.batch_size):
                try:
                    entries = generate_batch(llm, tokenizer, sampling,
                                             pairs[start:start + args.batch_size], prompt, args)
                except TeacherOutputError as error:
                    failure_path = Path(str(output) + ".failure.json")
                    atomic_json(failure_path, {**error.details,
                                              "line": start + error.details["batch_record"]})
                    raise ValueError(f"{error}; response saved to {failure_path}") from error
                for offset, entry in enumerate(entries):
                    entry["line"] = start + offset + 1
                    stream.write((json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
                completed.extend(entries)
                print(f"Checkpointed {len(completed):,}/{len(pairs):,} source lines", flush=True)
        export_dataset(output, completed)
        details = [target for entry in completed for target in entry["targets"]]
        changed = sum(t["original_translation"] != t["corrected_translation"] for t in details)
        summary = {"source_lines": len(completed), 
                   "translation_pairs": len(details),
                   "unchanged_translations": len(details) - changed, 
                   "changed_translations": changed,
                   "changed_percent": 100 * changed / len(details),
                   "large_edits": sum(t["large_edit"] for t in details),
                   "mean_edit_fraction": sum(t["edit_fraction"] for t in details) / len(details)}
        atomic_json(summary_path, summary)
        print(json.dumps(summary, indent=2))
        print(f"Training data: {output}\nAudit: {audit}\nSummary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=ROOT / "data/transgemma.10_data.jsonl.gz")
    parser.add_argument("--references", type=Path,
                        help="separate reference file; otherwise read embedded human_reference fields")
    parser.add_argument("--no-reference", action="store_true", help="use source and student only")
    parser.add_argument("--output", type=Path, default=ROOT / "data/postedited.10_data.jsonl.gz")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/minimal_post_edit.txt")
    parser.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B-FP8", help="HF model ID or local directory")
    parser.add_argument("--revision", help="model/tokenizer commit or tag; pin for reproducibility")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--review-edit-fraction", type=float, default=0.30, help="flag larger token edits for review; never clip or reject corrections")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="process first N lines; still validate all input alignment")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs and show one prompt; no GPU needed")
    args = parser.parse_args()
    if args.no_reference:
        args.references = None
    elif args.references is None and args.synthetic.resolve() == (ROOT / "data/transgemma.10_data.jsonl.gz").resolve():
        args.references = ROOT / "data/bitext.10_data.jsonl.gz"
    if any(n <= 0 for n in (args.tensor_parallel_size, args.batch_size, args.max_model_len, args.max_new_tokens)):
        parser.error("GPU, batch, and token counts must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or not 0 <= args.review_edit_fraction <= 1:
        parser.error("invalid GPU memory utilization or review edit fraction")
    if args.max_new_tokens >= args.max_model_len:
        parser.error("--max-new-tokens must be smaller than --max-model-len")
    if args.output.resolve() in {p.resolve() for p in (args.synthetic, args.references, args.prompt) if p}:
        parser.error("output must not overwrite an input or prompt")
    return args


def main():
    args = parse_args()
    try:
        prompt = args.prompt.read_text(encoding="utf-8").strip()
        if not prompt:
            raise ValueError("empty system prompt")
        pairs = load_pairs(args.synthetic, args.references, args.limit,
                           use_embedded_reference=not args.no_reference)
        if not args.no_reference and any(reference is None for _, reference in pairs):
            raise ValueError("human references are required; embed human_reference in each target "
                             "or supply --references")
        print(f"Validated {len(pairs):,} selected source lines", flush=True)
        if args.dry_run:
            student, reference = pairs[0]
            print(json.dumps(messages(student, reference, next(iter(targets(student))), prompt),
                             ensure_ascii=False, indent=2))
        else:
            run(args, pairs, prompt)
    except (OSError, ValueError, ImportError, EOFError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
