import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.post_edit_translations import load_pairs
from scripts.translate_tsv import (
    check_token_budget_resume, gemma3_rope_compatibility, generate_batch, load_tsv, run, translation_messages,
)


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [1, 2, 3]


class TranslationTests(unittest.TestCase):
    def test_token_budget_resume_is_restricted_to_increases(self):
        previous = {"model": "same", "input_sha256": "same", "prompt": "same",
                    "max_new_tokens": 512, "max_model_len": 2048, "script_sha256": "old"}
        current = {**previous, "max_new_tokens": 1024, "script_sha256": "new"}
        check_token_budget_resume(previous, current)
        for change in ({"model": "different"}, {"input_sha256": "different"},
                       {"prompt": "different"}, {"max_new_tokens": 256},
                       {"max_model_len": 1024}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    check_token_budget_resume(previous, {**current, **change})
        with self.assertRaises(ValueError):
            check_token_budget_resume(previous, {**previous, "script_sha256": "new"})

    def test_gemma3_rope_preserves_full_scaling_and_local_frequency(self):
        nested = {"full_attention": {"factor": 8.0, "rope_type": "linear"},
                  "sliding_attention": {"rope_type": "default"}}
        text = SimpleNamespace(model_type="gemma3_text", rope_parameters=nested,
                               rope_theta=1000000, rope_local_base_freq=10000)
        config = SimpleNamespace(get_text_config=lambda: text)
        self.assertIs(gemma3_rope_compatibility(config), config)
        self.assertEqual(text.rope_parameters,
                         {"factor": 8.0, "rope_type": "linear", "rope_theta": 1000000})
        self.assertEqual(text.rope_local_base_freq, 10000)
        self.assertEqual(text.rope_scaling, text.rope_parameters)
        self.assertEqual(nested["sliding_attention"], {"rope_type": "default"})
        self.assertNotIn("rope_theta", nested["full_attention"])

    def test_gemma3_rope_respects_per_layer_frequencies_and_rejects_scaled_local_rope(self):
        text = SimpleNamespace(model_type="gemma3_text", rope_parameters={
            "full_attention": {"rope_type": "linear", "factor": 8, "rope_theta": 500000},
            "sliding_attention": {"rope_type": "default", "rope_theta": 20000}},
            rope_theta=1000000, rope_local_base_freq=10000)
        config = SimpleNamespace(get_text_config=lambda: text)
        gemma3_rope_compatibility(config)
        self.assertEqual(text.rope_theta, 500000)
        self.assertEqual(text.rope_local_base_freq, 20000)
        text.rope_parameters = {"full_attention": {"rope_type": "linear", "factor": 8},
                                "sliding_attention": {"rope_type": "linear", "factor": 2}}
        with self.assertRaisesRegex(ValueError, "default sliding"):
            gemma3_rope_compatibility(config)

    def test_other_model_configs_are_unchanged(self):
        text = SimpleNamespace(model_type="qwen3_5", rope_parameters={"rope_type": "default"})
        config = SimpleNamespace(get_text_config=lambda: text)
        self.assertIs(gemma3_rope_compatibility(config), config)
        self.assertEqual(text.rope_parameters, {"rope_type": "default"})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.tsv"
        self.input.write_text('"Hello."\tBonjour.\nGoodbye.\tAu revoir.\n', encoding="utf-8")
        self.args = SimpleNamespace(input=self.input, output=self.root / "student.jsonl.gz",
                                    prompt=Path(__file__).resolve().parents[1] / "prompts/base_translation.txt",
                                    references_output=self.root / "reference.jsonl.gz",
                                    source_language="en", target_language="fr", model="fake",
                                    revision=None, skip_header=False, limit=None, seed=42,
                                    max_model_len=2048, max_new_tokens=512, tensor_parallel_size=1,
                                    batch_size=32, gpu_memory_utilization=0.9, resume=False)

    def test_tsv_preserves_quotes_and_rejects_bad_rows_beyond_limit(self):
        rows = load_tsv(self.input, "en", "fr")
        self.assertEqual(rows[0]["seg"], '"Hello."')
        self.input.write_text("Hello.\tBonjour.\nInvalid\n")
        with self.assertRaisesRegex(ValueError, "expected exactly two"):
            load_tsv(self.input, "en", "fr", limit=1)

    def test_header_and_compressed_input(self):
        path = self.root / "input.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write("source\thuman_reference\n  Hello.  \tBonjour.\n")
        rows = load_tsv(path, "en", "fr", skip_header=True)
        self.assertEqual(rows[0]["seg"], "  Hello.  ")

    def test_reference_never_enters_model_prompt(self):
        rows = load_tsv(self.input, "en", "fr")
        tokenizer = Tokenizer()
        responses = [SimpleNamespace(outputs=[SimpleNamespace(text="Salut.", finish_reason="stop")])]
        llm = SimpleNamespace(generate=lambda *args, **kwargs: responses)
        results = generate_batch(llm, tokenizer, None, rows[:1], self.args)
        self.assertEqual(tokenizer.messages, translation_messages('"Hello."', "en", "fr"))
        self.assertNotIn("Bonjour", json.dumps(tokenizer.messages))
        self.assertEqual(results[0]["record"]["tgts"][0]["base"], "Salut.")
        self.assertNotIn("seg", results[0]["record"]["tgts"][0])
        self.assertEqual(results[0]["record"]["tgts"][0]["human"], "Bonjour.")
        self.assertEqual(rows[0]["tgts"][0]["human"], "Bonjour.")

    def test_truncation_empty_and_context_overflow_are_rejected(self):
        rows = load_tsv(self.input, "en", "fr")[:1]
        for text, reason in (("partial", "length"), (" ", "stop")):
            llm = SimpleNamespace(generate=lambda *args, **kwargs: [SimpleNamespace(
                outputs=[SimpleNamespace(text=text, finish_reason=reason)])])
            with self.assertRaises(ValueError):
                generate_batch(llm, Tokenizer(), None, rows, self.args)
        self.args.max_model_len = 514
        with self.assertRaisesRegex(ValueError, "context budget"):
            generate_batch(None, Tokenizer(), None, rows, self.args)

    def test_base_output_trims_only_trailing_whitespace(self):
        rows = load_tsv(self.input, "en", "fr")[:1]
        llm = SimpleNamespace(generate=lambda *args, **kwargs: [SimpleNamespace(
            outputs=[SimpleNamespace(text="  Bonjour\nmonde. \t\r\n", finish_reason="stop")])])
        result = generate_batch(llm, Tokenizer(), None, rows, self.args)[0]
        self.assertEqual(result["record"]["tgts"][0]["base"], "  Bonjour\nmonde.")
        self.assertEqual(result["record"]["tgts"][0]["human"], "Bonjour.")

    def test_export_resume_and_posteditor_compatibility(self):
        rows = load_tsv(self.input, "en", "fr")
        class LLM:
            def __init__(self, **kwargs):
                pass

            def get_tokenizer(self):
                return Tokenizer()

            def generate(self, prompts, *args, **kwargs):
                return [SimpleNamespace(outputs=[SimpleNamespace(text="Traduction.", finish_reason="stop")])
                        for _ in prompts]

        fake = SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: None)
        with patch.dict(sys.modules, {"vllm": fake}):
            run(self.args, rows)
        pairs = load_pairs(self.args.output, self.args.references_output)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1]["tgts"][0]["human"], "Bonjour.")
        embedded_pairs = load_pairs(self.args.output)
        self.assertEqual(embedded_pairs[0][1]["tgts"][0]["human"], "Bonjour.")
        no_reference_pairs = load_pairs(self.args.output, use_embedded_reference=False)
        self.assertIsNone(no_reference_pairs[0][1])
        with self.assertRaisesRegex(ValueError, "exists"):
            run(self.args, rows)
        self.args.resume = True
        # Complete checkpoints can re-export without importing/loading vLLM.
        run(self.args, rows)
        self.args.max_new_tokens = 1024
        self.args.resume_with_larger_token_budget = True
        run(self.args, rows)
        history = json.loads(Path(str(self.args.output) + ".resume-history.jsonl").read_text())
        self.assertEqual(history["completed_rows"], 2)
        self.assertEqual(history["previous"]["max_new_tokens"], 512)
        self.assertEqual(history["next"]["max_new_tokens"], 1024)
        self.args.resume_with_larger_token_budget = False
        self.args.resume = False
        self.args.output = self.root / "self-contained.jsonl.gz"
        self.args.references_output = None
        with patch.dict(sys.modules, {"vllm": fake}):
            run(self.args, rows)
        self.assertEqual(len(load_pairs(self.args.output)), 2)
        self.assertFalse(Path(str(self.args.output) + ".references.jsonl.gz").exists())
        self.args.resume = True
        self.input.write_text("Changed.\tBonjour.\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            run(self.args, rows)

    def test_incomplete_embedded_reference_is_rejected(self):
        path = self.root / "bad.jsonl.gz"
        record = {"language": "en", "seg": "Hello.", "tgts": [
            {"language": "fr", "seg": "Bonjour.", "human_reference": ""}]}
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "empty embedded"):
            load_pairs(path)

    def test_larger_budget_resume_preserves_partial_checkpoint(self):
        rows = load_tsv(self.input, "en", "fr")
        self.args.batch_size = 1
        class LLM:
            def __init__(self, **kwargs):
                self.calls = 0

            def get_tokenizer(self):
                return Tokenizer()

            def generate(self, prompts, sampling, **kwargs):
                self.calls += 1
                reason = "length" if sampling.max_tokens == 512 and self.calls == 2 else "stop"
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text=f"Translation with budget {sampling.max_tokens}.", finish_reason=reason)])]
        fake = SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        with patch.dict(sys.modules, {"vllm": fake}):
            with self.assertRaisesRegex(ValueError, "length"):
                run(self.args, rows)
            self.args.resume = True
            self.args.resume_with_larger_token_budget = True
            self.args.max_new_tokens = 1024
            run(self.args, rows)
        pairs = load_pairs(self.args.output)
        self.assertEqual(pairs[0][0]["tgts"][0]["base"], "Translation with budget 512.")
        self.assertEqual(pairs[1][0]["tgts"][0]["base"], "Translation with budget 1024.")


if __name__ == "__main__":
    unittest.main()
