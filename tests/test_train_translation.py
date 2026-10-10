import ast
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.train_translation import parse_args, prepare_eval, prepare_training
from scripts import train_translation


class Tokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 9

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return [1, 2] + self.encode(messages[0]["content"][0]["text"])


class TrainingTests(unittest.TestCase):
    def test_callback_accepts_transformers_positional_arguments(self):
        # Exercise the callback without importing the CUDA training stack.
        tree = ast.parse(Path(train_translation.__file__).read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "on_step_end")
        namespace = {"args": SimpleNamespace(eval_steps=100)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "callback", "exec"), namespace)
        calls = []
        callback = SimpleNamespace(score=calls.append)
        control = object()
        for step in (1, 100):
            result = namespace["on_step_end"](callback, SimpleNamespace(),
                SimpleNamespace(global_step=step), control, model=object())
            self.assertIs(result, control)
        self.assertEqual(calls, [100])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "train.jsonl.gz"
        self.args = SimpleNamespace(method="sft", source_language="en", target_language="fr",
                                    max_length=20, max_new_tokens=5)
        self.tokenizer = Tokenizer()

    def write_rows(self, rows):
        with gzip.open(self.path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def test_sft_masks_prompt_and_adds_turn_end(self):
        self.write_rows([{"src": "Hi", "tgt": "Salut\n"}])
        rows, excluded = prepare_training(self.path, self.tokenizer, "template", self.args)
        self.assertEqual(excluded, [])
        self.assertEqual(rows[0]["labels"], [-100] * 4 + [ord(c) for c in "Salut"] + [9])
        self.assertNotIn("Salut", str(self.tokenizer.messages))

    def test_excludes_long_and_null_samples(self):
        self.write_rows([{"src": "x" * 30, "tgt": "y"}, {"src": "x", "tgt": None},
                         {"src": "x", "tgt": "y"}])
        rows, excluded = prepare_training(self.path, self.tokenizer, "template", self.args)
        self.assertEqual(len(rows), 1)
        self.assertEqual([r["reason"] for r in excluded], ["max_length", "missing_or_empty_field"])

    def test_dpo_checks_both_targets_and_identical_pairs(self):
        self.args.method = "dpo"
        self.write_rows([{"src": "x", "tgt": "y", "reject": "z" * 30},
                         {"src": "x", "tgt": "y\n", "reject": "y"},
                         {"src": "x", "tgt": "good", "reject": "bad"}])
        rows, excluded = prepare_training(self.path, self.tokenizer, "template", self.args)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["chosen_input_ids"], [ord(c) for c in "good"] + [9])
        self.assertEqual([r["reason"] for r in excluded], ["max_length", "identical_preferences"])

    def test_eval_preserves_original_line_numbers(self):
        path = self.path.with_suffix(".tsv")
        path.write_text("long sentence exceeds limit\tref\nHi\tSalut\n")
        rows, excluded = prepare_eval(path, self.tokenizer, "template", self.args)
        self.assertEqual(rows[0]["line"], 2)
        self.assertEqual(excluded[0]["line"], 1)
        path.write_text("bad\ttoo\tmany\n")
        with self.assertRaisesRegex(ValueError, "src<TAB>"):
            prepare_eval(path, self.tokenizer, "template", self.args)

    def test_method_specific_defaults(self):
        common = ["--train", "train", "--dev", "dev", "--prompt", "prompt", "--output", "out"]
        self.assertEqual(parse_args(["--method", "sft"] + common).learning_rate, 1e-4)
        self.assertEqual(parse_args(["--method", "dpo"] + common).learning_rate, 5e-6)
        args = parse_args(["--method", "sft", "--epochs", "1.5"] + common)
        self.assertEqual(args.epochs, 1.5)
        self.assertIsNone(args.test)
        self.assertFalse(hasattr(args, "max_steps"))

    def test_plain_and_gzip_formats_for_both_methods(self):
        for method in ("sft", "dpo"):
            self.args.method = method
            row = {"src": "Hi", "tgt": "Salut"}
            if method == "dpo":
                row["reject"] = "Bad"
            expected = None
            for suffix in (".json", ".jsonl", ".tsv", ".json.gz", ".jsonl.gz", ".tsv.gz"):
                with self.subTest(method=method, suffix=suffix):
                    path = Path(self.temp.name) / ("data" + suffix)
                    if ".tsv" in suffix:
                        text = "\t".join(row.values()) + "\n"
                    elif ".jsonl" in suffix:
                        text = json.dumps(row) + "\n"
                    else:
                        text = json.dumps([row], indent=2)
                    if suffix.endswith(".gz"):
                        with gzip.open(path, "wt", encoding="utf-8") as stream:
                            stream.write(text)
                    else:
                        path.write_text(text, encoding="utf-8")
                    training, excluded = prepare_training(path, self.tokenizer, "template", self.args)
                    self.assertEqual(excluded, [])
                    if expected is None:
                        expected = training
                    self.assertEqual(training, expected)
                    valid, excluded = prepare_eval(path, self.tokenizer, "template", self.args)
                    self.assertEqual(excluded, [])
                    self.assertEqual(valid[0]["tgt"], "Salut")

    def test_json_object_and_json_lines_with_json_suffix(self):
        path = Path(self.temp.name) / "data.json"
        row = {"src": "Hi", "tgt": "Salut"}
        for text, count in ((json.dumps(row, indent=2), 1),
                            (json.dumps(row) + "\n" + json.dumps(row) + "\n", 2)):
            path.write_text(text)
            training, excluded = prepare_training(path, self.tokenizer, "template", self.args)
            self.assertEqual(len(training), count)
            self.assertEqual(excluded, [])

    def test_dpo_validation_accepts_pair_only(self):
        self.args.method = "dpo"
        path = Path(self.temp.name) / "dev.tsv"
        path.write_text("Hi\tSalut\n")
        rows, excluded = prepare_eval(path, self.tokenizer, "template", self.args)
        self.assertEqual(len(rows), 1)
        self.assertEqual(excluded, [])
