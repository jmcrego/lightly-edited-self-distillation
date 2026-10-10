#!/usr/bin/env python3
"""Single-H100 TranslateGemma LoRA SFT/DPO with greedy BLEU/chrF evaluation.

Install in a separate CUDA training environment:
  pip install 'transformers==4.57.6' 'trl==0.19.1' 'peft==0.17.1' \
    'accelerate>=1.4,<2' 'datasets>=3,<4' 'sacrebleu>=2.5,<3' sentencepiece
PyTorch must be a CUDA build. Check --help for all experiment settings.
Train/dev/test accept .json (object or array), .jsonl, or .tsv, optionally .gz.
SFT: src/tgt fields or src<TAB>tgt columns.
DPO training: src/tgt/reject fields or src<TAB>tgt<TAB>reject columns;
tgt is chosen and reject is rejected. Dev/test use src/tgt for BLEU/chrF;
an additional DPO reject field/column is accepted but unused for decoding.
All options apply to both methods unless marked DPO-only. There are no
SFT-only options. Run once per target variant/LR/batch setting.
Adapters use the text-only model returned by load_model(), not the multimodal
wrapper. Reload them against that same text-only base model.
"""

import argparse
from collections import Counter
from contextlib import nullcontext
import gzip
import json
from pathlib import Path

if __package__:
    from .translate_tsv import gemma3_rope_compatibility
else:
    from translate_tsv import gemma3_rope_compatibility


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_examples(path, fields, optional_reject=False):
    suffix = path.with_suffix("").suffix if path.suffix == ".gz" else path.suffix
    if suffix not in {".json", ".jsonl", ".tsv"}:
        raise ValueError(f"{path}: expected .json, .jsonl, or .tsv (optionally .gz)")
    with open_text(path) as stream:
        if suffix == ".json":
            try:
                values = json.load(stream)
            except json.JSONDecodeError:
                # Also accept line-delimited objects with a .json suffix.
                stream.seek(0)
            else:
                values = values if isinstance(values, list) else [values]
                for index, value in enumerate(values, 1):
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:record {index}: expected a JSON object")
                    yield index, value
                return
        for line, text in enumerate(stream, 1):
            if suffix == ".tsv":
                parts = text.rstrip("\r\n").split("\t")
                columns = fields + ["reject"] if optional_reject and len(parts) == len(fields) + 1 else fields
                if len(parts) != len(columns):
                    expected = "<TAB>".join(fields)
                    raise ValueError(f"{path}:{line}: expected {expected}")
                value = dict(zip(columns, parts))
            else:
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line}: invalid JSON: {error}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line}: expected a JSON object")
            yield line, value


def prompt_ids(tokenizer, prompt, source, args):
    messages = [{"role": "user", "content": [{"type": "text", "text": source,
                 "source_lang_code": args.source_language, "target_lang_code": args.target_language}]}]
    return tokenizer.apply_chat_template(messages, chat_template=prompt, tokenize=True,
                                         add_generation_prompt=True)


def completion_ids(tokenizer, text):
    # Gemma ends assistant turns with end_of_turn, not the document EOS token.
    end = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if end is None or end == tokenizer.unk_token_id:
        raise ValueError("tokenizer has no Gemma <end_of_turn> token")
    return tokenizer.encode(text.rstrip(), add_special_tokens=False) + [end]


def prepare_training(path, tokenizer, prompt, args):
    rows, excluded = [], []
    fields = ["src", "tgt"] if args.method == "sft" else ["src", "tgt", "reject"]
    for line, value in read_examples(path, fields):
        if any(not isinstance(value.get(k), str) or not value[k].strip() for k in fields):
            excluded.append({"line": line, "reason": "missing_or_empty_field"})
            continue
        prefix = prompt_ids(tokenizer, prompt, value["src"], args)
        endings = [completion_ids(tokenizer, value[k]) for k in fields[1:]]
        if any(len(prefix) + len(end) > args.max_length for end in endings):
            excluded.append({"line": line, "reason": "max_length"})
            continue
        if args.method == "sft":
            ids = prefix + endings[0]
            rows.append({"input_ids": ids, "attention_mask": [1] * len(ids),
                         "labels": [-100] * len(prefix) + endings[0]})
        elif endings[0] == endings[1]:
            excluded.append({"line": line, "reason": "identical_preferences"})
        else:
            rows.append({"prompt_input_ids": prefix, "chosen_input_ids": endings[0],
                         "rejected_input_ids": endings[1]})
    return rows, excluded


def prepare_eval(path, tokenizer, prompt, args):
    rows, excluded = [], []
    for line, value in read_examples(path, ["src", "tgt"], optional_reject=args.method == "dpo"):
        if any(not isinstance(value.get(k), str) or not value[k].strip() for k in ("src", "tgt")):
            raise ValueError(f"{path}:{line}: expected nonempty src/tgt fields")
        ids = prompt_ids(tokenizer, prompt, value["src"], args)
        if len(ids) + args.max_new_tokens > args.max_length:
            excluded.append({"line": line, "reason": "generation_context_budget"})
        else:
            rows.append({"line": line, "src": value["src"], "tgt": value["tgt"], "input_ids": ids})
    if not rows:
        raise ValueError(f"{path}: no evaluation rows fit the generation budget")
    return rows, excluded


class TargetCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        length = max(len(row["input_ids"]) for row in rows)
        return {key: torch.tensor([row[key] + [pad] * (length - len(row[key])) for row in rows])
                for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100))}


def load_model(args):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Gemma3ForCausalLM, Gemma3ForConditionalGeneration

    config = gemma3_rope_compatibility(AutoConfig.from_pretrained(args.model))
    if config.model_type != "gemma3":
        raise ValueError("this loader expects the multimodal TranslateGemma/Gemma3 checkpoint")
    # Load on CPU, then discard vision weights before moving the language model
    # to CUDA. Reuse tensors rather than allocating a second 12B model.
    full = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation=args.attention_implementation)
    with init_empty_weights():
        model = Gemma3ForCausalLM(config.text_config)
    model.model = full.model.language_model
    model.lm_head = full.lm_head
    model.generation_config = full.generation_config
    model.config._name_or_path = args.model
    del full
    model.config.use_cache = False
    return model.to("cuda")


def evaluate(model, tokenizer, rows, directory, args, baseline=False):
    import torch
    from sacrebleu.metrics import BLEU, CHRF

    directory.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    was_checkpointing = model.is_gradient_checkpointing
    model.eval()
    if was_checkpointing:
        model.gradient_checkpointing_disable()
    hypotheses, details = [], []
    context = model.disable_adapter() if baseline else nullcontext()
    try:
        with context, torch.inference_mode():
            for start in range(0, len(rows), args.eval_batch_size):
                batch = rows[start:start + args.eval_batch_size]
                length = max(len(row["input_ids"]) for row in batch)
                ids = torch.tensor([[tokenizer.pad_token_id] * (length - len(row["input_ids"]))
                                    + row["input_ids"] for row in batch], device="cuda")
                mask = torch.tensor([[0] * (length - len(row["input_ids"]))
                                     + [1] * len(row["input_ids"]) for row in batch], device="cuda")
                generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                    num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=args.stop_ids)
                for row, output in zip(batch, generated[:, length:].tolist()):
                    hyp = " ".join(tokenizer.decode(output, skip_special_tokens=True).splitlines()).strip()
                    hypotheses.append(hyp)
                    details.append({"line": row["line"], "hyp": hyp,
                                    "hit_token_limit": not any(t in args.stop_ids for t in output)})
    finally:
        if was_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train(was_training)
    (directory / "hyps.txt").write_text("\n".join(hypotheses) + "\n", encoding="utf-8")
    with (directory / "hyps.jsonl").open("w", encoding="utf-8") as stream:
        for detail in details:
            stream.write(json.dumps(detail, ensure_ascii=False) + "\n")
    references = [[row["tgt"] for row in rows]]
    bleu, chrf = BLEU(tokenize=args.bleu_tokenizer), CHRF()
    scores = {"bleu": bleu.corpus_score(hypotheses, references).score,
              "chrf": chrf.corpus_score(hypotheses, references).score,
              "sentences": len(rows), "token_limit_hits": sum(d["hit_token_limit"] for d in details),
              "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}
    write_json(directory / "metrics.json", scores)
    print(f"{directory}: {json.dumps(scores)}", flush=True)
    return scores


def run(args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("expose exactly one CUDA GPU for this script")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory is not empty; select a new experiment directory")
    args.output.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt.read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    args.stop_ids = list({tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")})
    training, excluded = prepare_training(args.train, tokenizer, prompt, args)
    write_json(args.output / "excluded.json", {"train": excluded})
    reasons = dict(Counter(row["reason"] for row in excluded))
    print(f"Retained {len(training)} training samples; excluded {len(excluded)}: {reasons}", flush=True)
    if not training:
        fields = "src/tgt" if args.method == "sft" else "src/tgt/reject"
        raise ValueError(f"no training samples survived filtering ({reasons}); expected {fields} fields. "
                         "Convert seg/tgts postediting records with scripts/prepare_dataset.py. "
                         f"See {args.output / 'excluded.json'}")
    dev, dev_excluded = prepare_eval(args.dev, tokenizer, prompt, args)
    test, test_excluded = prepare_eval(args.test, tokenizer, prompt, args) if args.test else ([], [])
    write_json(args.output / "excluded.json", {"train": excluded, "dev": dev_excluded, "test": test_excluded})
    model = get_peft_model(load_model(args), LoraConfig(
        task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, target_modules=args.lora_targets.split(",")))
    model.print_trainable_parameters()
    evaluate(model, tokenizer, dev, args.output / "baseline-dev", args, baseline=True)
    if test:
        evaluate(model, tokenizer, test, args.output / "baseline-test", args, baseline=True)

    class DevCallback(TrainerCallback):
        def __init__(self):
            self.best = float("-inf")
            self.last_step = None

        def score(self, step):
            if step == self.last_step:
                return
            scores = evaluate(model, tokenizer, dev, args.output / f"dev-step-{step}", args)
            self.last_step = step
            if scores["chrf"] > self.best:
                self.best = scores["chrf"]
                model.save_pretrained(args.output / "best-adapter")
                write_json(args.output / "best.json", {"step": step, **scores})

        def on_step_end(self, state, control, **kwargs):
            if state.global_step % args.eval_steps == 0:
                self.score(state.global_step)
            return control

    callback = DevCallback()
    common = dict(output_dir=str(args.output / "checkpoints"),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=2, report_to="none", optim="adamw_torch", seed=42)
    if args.method == "sft":
        trainer = Trainer(model=model, args=TrainingArguments(**common),
            train_dataset=Dataset.from_list(training), data_collator=TargetCollator(tokenizer.pad_token_id),
            processing_class=tokenizer, callbacks=[callback])
    else:
        class TokenizedDPOTrainer(DPOTrainer):
            def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
                # Already tokenized and length-checked; preserve Gemma turn EOS
                # and do not allow TRL's default prompt truncation or EOS append.
                return dataset

        trainer = TokenizedDPOTrainer(model=model, ref_model=None,
            args=DPOConfig(**common, beta=args.dpo_beta, loss_type=args.dpo_loss,
                max_length=args.max_length, max_prompt_length=None, max_completion_length=None,
                precompute_ref_log_probs=True, precompute_ref_batch_size=1,
                disable_dropout=False), train_dataset=Dataset.from_list(training),
            processing_class=tokenizer, callbacks=[callback])
    trainer.train()
    callback.score(trainer.state.global_step)
    trainer.save_model(str(args.output / "final-adapter"))
    tokenizer.save_pretrained(args.output / "tokenizer")
    model.load_adapter(str(args.output / "best-adapter"), adapter_name="best")
    model.set_adapter("best")
    if test:
        evaluate(model, tokenizer, test, args.output / "best-test", args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter) #argparse.RawDescriptionHelpFormatter)
    common = parser.add_argument_group("Common")
    common.add_argument("--method", choices=["sft", "dpo"], required=True, help="Training method")
    common.add_argument("--train", type=Path, required=True, help="Training JSON/JSONL or TSV, optionally .gz; SFT: src/tgt; DPO: src/tgt/reject")
    common.add_argument("--dev", type=Path, required=True, help="Validation JSON/JSONL or TSV with src/tgt, optionally .gz; DPO reject is accepted but unused")
    common.add_argument("--test", type=Path, help="Optional test file in the same format as dev; omit for dev-only evaluation")
    common.add_argument("--prompt", type=Path, required=True, help="Jinja chat-template file for source-only translation prompts")
    common.add_argument("--output", type=Path, required=True, help="New or empty output directory for adapters, checkpoints, hypotheses, and metrics")
    common.add_argument("--model", default="/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it", help="TranslateGemma checkpoint directory or Hugging Face model ID")
    common.add_argument("--source-language", default="en", help="Source language code supplied to the prompt")
    common.add_argument("--target-language", default="fr", help="Target language code supplied to the prompt")
    common.add_argument("--learning-rate", type=float, help="Optimizer learning rate; when omitted, SFT uses 1e-4 and DPO uses 5e-6", default=argparse.SUPPRESS)
    common.add_argument("--micro-batch-size", type=int, default=1, help="Training examples per GPU forward/backward pass (DPO processes both responses per example)")
    common.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Microbatches per optimizer step; effective batch = micro-batch-size times this value")
    common.add_argument("--epochs", type=float, default=1, help="Training epochs; fractional values are allowed (e.g. 1.5)")
    common.add_argument("--max-length", type=int, default=2048, help="Prompt plus completion token limit, including special tokens; exclude oversized samples without truncation")
    common.add_argument("--max-new-tokens", type=int, default=512, help="Maximum response tokens for greedy evaluation; prompts must fit max-length with this reservation")
    common.add_argument("--eval-steps", type=int, default=100, help="Greedy dev evaluation every N optimizer steps")
    common.add_argument("--eval-batch-size", type=int, default=1, help="Source sentences decoded together during evaluation")
    common.add_argument("--logging-steps", type=int, default=10, help="Optimizer steps between training log updates")
    common.add_argument("--warmup-ratio", type=float, default=0.03, help="Fraction of optimizer steps used for learning-rate warmup")
    common.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    common.add_argument("--lora-alpha", type=int, default=32, help="LoRA scaling numerator; adapter scale is alpha/rank")
    common.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability during training")
    common.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated module names to adapt with LoRA")
    common.add_argument("--attention-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa", help="Attention backend; flash_attention_2 requires the flash-attn package")
    common.add_argument("--bleu-tokenizer", default="13a", help="SacreBLEU tokenization method; does not affect model tokenization")
    parser.add_argument_group("SFT", "No SFT-only options; use the common options.")
    dpo = parser.add_argument_group("DPO")
    dpo.add_argument("--dpo-beta", type=float, default=0.1, help="DPO regularization coefficient relative to the frozen base model")
    dpo.add_argument("--dpo-loss", choices=["sigmoid", "hinge", "ipo"], default="sigmoid", help="Preference loss")
    args = parser.parse_args(argv)
    if not hasattr(args, "learning_rate"):
        args.learning_rate = 1e-4 if args.method == "sft" else 5e-6
    positive = (args.micro_batch_size, args.gradient_accumulation_steps, args.epochs,
                args.max_length, args.max_new_tokens, args.eval_steps, args.eval_batch_size,
                args.logging_steps, args.lora_rank, args.lora_alpha, args.learning_rate, args.dpo_beta)
    if any(value <= 0 for value in positive) or args.max_new_tokens >= args.max_length:
        parser.error("positive budgets required; max-new-tokens must be less than max-length")
    if not 0 <= args.lora_dropout < 1 or not 0 <= args.warmup_ratio <= 1:
        parser.error("invalid dropout or warmup ratio")
    return args


if __name__ == "__main__":
    run(parse_args())
