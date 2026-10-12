import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.test_translation import parse_args, prepare_input, score_comet


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return list(messages[0]["content"][0]["text"].encode())


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input = self.directory / "input.tsv"
        self.args = parse_args(["--input", str(self.input), "--output", str(self.directory / "out"),
                                "--max-length", "20", "--max-new-tokens", "5"])

    def test_reference_alignment_after_excluding_long_input(self):
        self.input.write_text("x" * 30 + "\tlong ref\nHi\tSalut\n")
        rows, excluded, has_refs = prepare_input(self.args, Tokenizer(), "template")
        self.assertTrue(has_refs)
        self.assertEqual(rows[0]["line"], 2)
        self.assertEqual(rows[0]["tgt"], "Salut")
        self.assertEqual(excluded[0]["line"], 1)

    def test_reference_free_and_external_references(self):
        self.input.write_text("Hi\nBye\n")
        rows, _, has_refs = prepare_input(self.args, Tokenizer(), "template")
        self.assertFalse(has_refs)
        self.assertIsNone(rows[0]["tgt"])
        ref = self.directory / "refs.txt"
        ref.write_text("Salut\nAu revoir\n")
        self.args.references = ref
        rows, _, has_refs = prepare_input(self.args, Tokenizer(), "template")
        self.assertTrue(has_refs)
        self.assertEqual(rows[1]["tgt"], "Au revoir")
        ref.write_text("one\n")
        with self.assertRaisesRegex(ValueError, "reference count"):
            prepare_input(self.args, Tokenizer(), "template")

    def test_gzipped_jsonl_and_partial_reference_rejection(self):
        self.args.input = self.directory / "input.jsonl.gz"
        with gzip.open(self.args.input, "wt") as stream:
            stream.write(json.dumps({"src": "Hi", "tgt": "Salut"}) + "\n")
            stream.write(json.dumps({"src": "Bye"}) + "\n")
        with self.assertRaisesRegex(ValueError, "every input record"):
            prepare_input(self.args, Tokenizer(), "template")

    def test_comet_uses_aligned_human_reference(self):
        self.args.output.mkdir()
        rows = [{"line": 2, "src": "Hi", "tgt": "Salut"}]
        calls = []

        def predict(data, **kwargs):
            calls.append(data)
            return SimpleNamespace(scores=[0.8], system_score=0.8)

        module = SimpleNamespace(load_from_checkpoint=lambda path: SimpleNamespace(predict=predict))
        with patch.dict("sys.modules", {"comet": module}):
            result = score_comet(self.args, rows, ["Bonjour"])
        self.assertEqual(calls[0], [{"src": "Hi", "mt": "Bonjour", "ref": "Salut"}])
        self.assertEqual(result["comet"], 0.8)
        self.assertEqual(json.loads((self.args.output / "comet-scores.json").read_text())[0]["line"], 2)
