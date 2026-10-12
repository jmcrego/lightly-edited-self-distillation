#!/usr/bin/env python3
"""Greedy TranslateGemma translation, optionally with a LoRA adapter.

Input: JSON/JSONL with src and optional tgt, TSV with src[<TAB>tgt], or
plain .txt sources. All formats support .gz. Alternatively provide human
references as a separate plain-text file, one per original input record.
With references, compute BLEU, chrF, and COMET (unless --no-comet).
Use the training venv; see README.md for COMET compatibility pins.
"""

import argparse
import gc
import json
from pathlib import Path
import sys

if __package__:
    from .train_translation import evaluate, load_model, open_text, prompt_ids, read_examples, write_json
else:
    from train_translation import evaluate, load_model, open_text, prompt_ids, read_examples, write_json


DEFAULT_COMET = "/lustre/fsmisc/dataset/HuggingFace_Models/Unbabel/XCOMET-XL/checkpoints/model.ckpt"


def read_input(path):
    suffix = path.with_suffix("").suffix if path.suffix == ".gz" else path.suffix
    if suffix in {".json", ".jsonl"}:
        return list(read_examples(path, ["src", "tgt"]))
    if suffix not in {".txt", ".tsv"}:
        raise ValueError("input must be .json, .jsonl, .tsv, or .txt, optionally .gz")
    rows = []
    with open_text(path) as stream:
        for line, text in enumerate(stream, 1):
            parts = text.rstrip("\r\n").split("\t") if suffix == ".tsv" else [text.rstrip("\r\n")]
            if len(parts) not in {1, 2}:
                raise ValueError(f"{path}:{line}: expected src or src<TAB>tgt")
            rows.append((line, dict(zip(["src", "tgt"], parts))))
    return rows


def prepare_input(args, tokenizer, prompt):
    inputs = read_input(args.input)
    if args.references:
        with open_text(args.references) as stream:
            references = [text.rstrip("\r\n") for text in stream]
        if len(references) != len(inputs):
            raise ValueError("reference count must match the original input count before filtering")
        for (_, value), reference in zip(inputs, references):
            value["tgt"] = reference
    supplied = any("tgt" in value for _, value in inputs)
    if supplied and any(not isinstance(value.get("tgt"), str) or not value["tgt"].strip()
                        for _, value in inputs):
        raise ValueError("references must be nonempty strings for every input record, or absent for all")
    rows, excluded = [], []
    for line, value in inputs:
        if not isinstance(value.get("src"), str) or not value["src"].strip():
            raise ValueError(f"{args.input}:{line}: expected a nonempty src string")
        ids = prompt_ids(tokenizer, prompt, value["src"], args)
        if len(ids) + args.max_new_tokens > args.max_length:
            excluded.append({"line": line, "reason": "generation_context_budget"})
        else:
            rows.append({"line": line, "src": value["src"], "tgt": value.get("tgt"), "input_ids": ids})
    if not rows:
        raise ValueError("no input sentences fit the generation context budget")
    return rows, excluded, supplied


def score_comet(args, rows, hypotheses):
    from comet import load_from_checkpoint

    model = load_from_checkpoint(str(args.comet_model))
    data = [{"src": row["src"], "mt": hyp, "ref": row["tgt"]}
            for row, hyp in zip(rows, hypotheses)]
    result = model.predict(data, batch_size=args.comet_batch_size, gpus=args.comet_gpus)
    if len(result.scores) != len(rows):
        raise ValueError("COMET returned an unexpected number of scores")
    details = [{"line": row["line"], "score": float(score)} for row, score in zip(rows, result.scores)]
    write_json(args.output / "comet-scores.json", details)
    return {"comet": float(result.system_score), "comet_model": str(args.comet_model)}


def run(args):
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("translation requires exactly one visible CUDA GPU")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be new or empty")
    if args.adapter and not args.adapter.is_dir():
        raise ValueError("LoRA adapter must be an existing directory")
    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    turn_end = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if turn_end is None or turn_end == tokenizer.unk_token_id:
        raise ValueError("tokenizer has no Gemma <end_of_turn> token")
    args.stop_ids = list({tokenizer.eos_token_id, turn_end} - {None})
    rows, excluded, has_references = prepare_input(args, tokenizer, args.prompt.read_text(encoding="utf-8"))
    if has_references and not args.no_comet:
        if not args.comet_model.is_file():
            raise ValueError("COMET checkpoint missing; set --comet-model or use --no-comet")
        import comet  # Fail before expensive translation if its dependencies are broken.
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "excluded.json", excluded)
    print(f"Translating {len(rows)} sentences; excluded {len(excluded)}", flush=True)
    model = load_model(args)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    scores = evaluate(model, tokenizer, rows, args.output, args, score_references=has_references)
    # Keep the training-sized model and COMET off the GPU at the same time.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    details = [json.loads(line) for line in (args.output / "hyps.jsonl").read_text(encoding="utf-8").splitlines()]
    hypotheses = [entry["hyp"] for entry in details]
    for name, values in (("sources", [row["src"] for row in rows]),
                         ("references", [row["tgt"] for row in rows] if has_references else [])):
        if values:
            with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for row, value in zip(rows, values):
                    stream.write(json.dumps({"line": row["line"], "text": value}, ensure_ascii=False) + "\n")
    if has_references and not args.no_comet:
        print("Scoring with COMET", flush=True)
        scores.update(score_comet(args, rows, hypotheses))
    write_json(args.output / "metrics.json", scores)
    print(json.dumps(scores, indent=2), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Sources or source/reference pairs")
    parser.add_argument("--references", type=Path, help="Optional reference text file; overrides input tgt fields")
    parser.add_argument("--model", default="/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it", help="Base TranslateGemma checkpoint")
    parser.add_argument("--adapter", "--lora-checkpoint", type=Path, help="Optional saved LoRA adapter directory")
    parser.add_argument("--prompt", type=Path, default=Path("prompts/base_translation.txt"), help="Translation Jinja template")
    parser.add_argument("--output", type=Path, required=True, help="New or empty results directory")
    parser.add_argument("--source-language", default="en")
    parser.add_argument("--target-language", default="fr")
    parser.add_argument("--max-length", type=int, default=2048, help="Prompt plus reserved generation tokens; exclude oversized inputs")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--attention-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument("--bleu-tokenizer", default="13a")
    parser.add_argument("--comet-model", type=Path, default=Path(DEFAULT_COMET), help="Local COMET checkpoint; used only with references")
    parser.add_argument("--comet-batch-size", type=int, default=4)
    parser.add_argument("--comet-gpus", type=int, choices=[0, 1], default=1)
    parser.add_argument("--no-comet", action="store_true", help="With references, compute only BLEU/chrF")
    args = parser.parse_args(argv)
    if min(args.max_length, args.max_new_tokens, args.eval_batch_size, args.comet_batch_size) <= 0:
        parser.error("token limits and batch sizes must be positive")
    if args.max_new_tokens >= args.max_length:
        parser.error("max-new-tokens must be less than max-length")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
