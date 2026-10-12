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
        self.path = Path(self.temp.name) / "train.tsv.gz"
        self.args = SimpleNamespace(method="sft", source_language="en", target_language="fr",
                                    max_length=20, max_new_tokens=5, src_col=1,
                                    tgt_col=4, chosen_col=4, rejected_col=3)
        self.tokenizer = Tokenizer()

    def write_rows(self, rows):
        with gzip.open(self.path, "wt", encoding="utf-8") as stream:
            for row in rows:
                values = [row["src"], row.get("human", "Human"), row.get("reject", "Base"), row.get("tgt") or ""]
                stream.write("\t".join(values) + "\n")

    def test_sft_masks_prompt_and_adds_turn_end(self):
        self.write_rows([{"src": "Hi", "tgt": "Salut "}])
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
                         {"src": "x", "tgt": "y ", "reject": "y"},
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
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            prepare_eval(path, self.tokenizer, "template", self.args)

    def test_method_specific_defaults(self):
        common = ["--train", "train", "--dev", "dev", "--prompt", "prompt", "--output", "out"]
        self.assertEqual(parse_args(["--method", "sft", "--tgt_col", "4"] + common).learning_rate, 1e-4)
        self.assertEqual(parse_args(["--method", "dpo", "--chosen_col", "4", "--rejected_col", "3"] + common).learning_rate, 5e-6)
        args = parse_args(["--method", "sft", "--tgt_col", "2", "--epochs", "1.5"] + common)
        self.assertEqual(args.epochs, 1.5)
        self.assertFalse(hasattr(args, "test"))
        self.assertFalse(hasattr(args, "max_steps"))

    def test_plain_and_gzip_tsv_for_both_methods(self):
        for method in ("sft", "dpo"):
            self.args.method = method
            expected = None
            for suffix in (".tsv", ".tsv.gz"):
                with self.subTest(method=method, suffix=suffix):
                    path = Path(self.temp.name) / ("data" + suffix)
                    text = "Hi\tHuman\tBad\tSalut\n"
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

    def test_column_selection_and_dev_independence(self):
        self.write_rows([{"src": "Hi", "human": "Human", "tgt": "Corrected"}])
        self.args.tgt_col = 2
        self.args.src_col = 3
        training, excluded = prepare_training(self.path, self.tokenizer, "template", self.args)
        self.assertEqual(excluded, [])
        self.assertEqual(self.tokenizer.messages[0]["content"][0]["text"], "Base")
        self.assertEqual(training[0]["labels"][-6:], [ord(c) for c in "Human"] + [9])
        dev = Path(self.temp.name) / "dev.tsv"
        dev.write_text("Hi\tHuman\n")
        rows, _ = prepare_eval(dev, self.tokenizer, "template", self.args)
        self.assertEqual(rows[0]["src"], "Hi")
        self.assertEqual(rows[0]["tgt"], "Human")

    def test_training_requires_four_fields(self):
        with gzip.open(self.path, "wt") as stream:
            stream.write("Hi\tHuman\tBase\n")
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            prepare_training(self.path, self.tokenizer, "template", self.args)

    def test_cli_requires_method_specific_columns(self):
        from contextlib import redirect_stderr
        import io
        common = ["--train", "train", "--dev", "dev", "--prompt", "prompt", "--output", "out"]
        for extra in (["--method", "sft"], ["--method", "dpo"],
                      ["--method", "dpo", "--chosen_col", "3", "--rejected_col", "3"],
                      ["--method", "sft", "--tgt_col", "5"]):
            with self.subTest(options=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(extra + common)

    def test_dpo_validation_accepts_pair_only(self):
        self.args.method = "dpo"
        path = Path(self.temp.name) / "dev.tsv"
        path.write_text("Hi\tSalut\n")
        rows, excluded = prepare_eval(path, self.tokenizer, "template", self.args)
        self.assertEqual(len(rows), 1)
        self.assertEqual(excluded, [])
