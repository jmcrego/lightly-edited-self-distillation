import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.post_edit_translations import load_pairs
from scripts.translate_tsv import generate_batch, load_tsv, run, translation_messages


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [1, 2, 3]


class TranslationTests(unittest.TestCase):
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
        self.assertEqual(results[0]["record"]["tgts"][0]["seg"], "Salut.")
        self.assertEqual(rows[0]["tgts"][0]["seg"], "Bonjour.")

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
        self.assertEqual(pairs[0][1]["tgts"][0]["seg"], "Bonjour.")
        with self.assertRaisesRegex(ValueError, "exists"):
            run(self.args, rows)
        self.args.resume = True
        # Complete checkpoints can re-export without importing/loading vLLM.
        run(self.args, rows)
        self.input.write_text("Changed.\tBonjour.\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            run(self.args, rows)


if __name__ == "__main__":
    unittest.main()
