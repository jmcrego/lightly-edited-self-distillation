import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.post_edit_translations import (
    check_context_skip_migration, edit_fraction, export_dataset, generate_batch, load_pairs, messages,
    read_checkpoint, records, run, targets, validate_result,
)


def row(source="Hello.", target="Bonjour."):
    return {"language": "en", "seg": source,
            "tgts": [{"language": "fr", "seg": target}]}


def answer(corrected="Bonjour.", edits=None):
    return {"corrected_translation": corrected,
            "edits": [] if edits is None else edits}


class FakeTokenizer:
    def apply_chat_template(self, value, **kwargs):
        assert kwargs["enable_thinking"] is False
        return json.dumps(value)

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))


class FakeLLM:
    def __init__(self, responses, finish_reason="stop"):
        self.responses = responses
        self.finish_reason = finish_reason

    def generate(self, prompts, sampling, **kwargs):
        assert len(prompts) == len(self.responses)
        return [SimpleNamespace(outputs=[SimpleNamespace(
            text=json.dumps(response), finish_reason=self.finish_reason)])
                for response in self.responses]


class PostEditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.args = SimpleNamespace(max_new_tokens=1024, max_model_len=4096,
                                    review_edit_fraction=0.30)

    def write_rows(self, name, values):
        path = self.directory / name
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value) + "\n")
        return path

    def test_alignment_checked_beyond_limit(self):
        left = self.write_rows("left.gz", [row(), row("Second")])
        right = self.write_rows("right.gz", [row(), row("Wrong")])
        with self.assertRaisesRegex(ValueError, "line 2.*mismatch"):
            load_pairs(left, right, limit=1)

    def test_record_count_mismatch(self):
        left = self.write_rows("left.gz", [row(), row()])
        right = self.write_rows("right.gz", [row()])
        with self.assertRaisesRegex(ValueError, "counts differ"):
            load_pairs(left, right)

    def test_optional_reference_and_prompt_payload(self):
        student = row(target="Salut.")
        reference = row()
        payload = json.loads(messages(student, reference, "fr", "prompt")[1]["content"])
        self.assertEqual(payload["student_translation"], "Salut.")
        self.assertEqual(payload["human_reference"], "Bonjour.")
        self.assertNotIn("human_reference", json.loads(messages(student, None, "fr", "p")[1]["content"]))
        path = self.write_rows("input.gz", [student])
        self.assertEqual(load_pairs(path), [(student, None)])

    def test_unchanged_preserves_bytes_and_extra_fields(self):
        student = row(target=" Bonjour. ")
        student["id"] = "example"
        result = generate_batch(FakeLLM([answer(" Bonjour. ")]), FakeTokenizer(), None,
                                [(student, row())], "prompt", self.args)[0]
        self.assertEqual(result["record"]["id"], student["id"])
        self.assertEqual(result["record"]["seg"], student["seg"])
        self.assertEqual(result["record"]["tgts"][0]["base"], " Bonjour.")
        self.assertEqual(result["record"]["tgts"][0]["corrected"], " Bonjour. ")
        self.assertNotIn("seg", result["record"]["tgts"][0])
        self.assertEqual(result["targets"][0]["edit_fraction"], 0)

    def test_correction_and_large_edit_flag(self):
        student = row(target="Au revoir.")
        response = answer(edits=[{"before": "Au revoir", "after": "Bonjour",
                                 "category": "meaning", "reason": "Greeting, not farewell."}])
        result = generate_batch(FakeLLM([response]), FakeTokenizer(), None,
                                [(student, row())], "prompt", self.args)[0]
        self.assertEqual(result["record"]["tgts"][0]["corrected"], "Bonjour.")
        self.assertEqual(result["record"]["tgts"][0]["base"], "Au revoir.")
        self.assertEqual(student["tgts"][0]["seg"], "Au revoir.")
        self.assertTrue(result["targets"][0]["large_edit"])
        self.assertNotIn("needs_review", result["targets"][0])

    def test_named_fields_use_student_not_existing_correction(self):
        student = {"language": "en", "seg": "Hello.", "tgts": [{
            "language": "fr", "student": "Salut.", "human_reference": "Bonjour.",
            "corrected": "Existing correction."}]}
        path = self.write_rows("embedded.gz", [student])
        pairs = load_pairs(path)
        payload = json.loads(messages(*pairs[0], "fr", "prompt")[1]["content"])
        self.assertEqual(payload["student_translation"], "Salut.")
        self.assertEqual(payload["human_reference"], "Bonjour.")
        result = generate_batch(FakeLLM([answer("Salut.")]), FakeTokenizer(), None,
                                pairs, "prompt", self.args)[0]
        self.assertEqual(result["record"]["tgts"][0], {
            "language": "fr", "base": "Salut.", "human": "Bonjour.",
            "corrected": "Salut."})

    def test_base_reader_trims_new_and_legacy_fields_only(self):
        for field in ("base", "student", "seg"):
            student = {"language": "en", "seg": "Hello.", "tgts": [{
                "language": "fr", field: "  Bonjour\nmonde. \t\r\n",
                "human": "Référence.\n"}]}
            with self.subTest(field=field):
                self.assertEqual(targets(student)["fr"], "  Bonjour\nmonde.")
                self.assertEqual(targets(student, "human")["fr"], "Référence.\n")
                payload = json.loads(messages(student, None, "fr", "prompt")[1]["content"])
                self.assertEqual(payload["student_translation"], "  Bonjour\nmonde.")

    def test_null_base_is_preserved_without_calling_teacher(self):
        student = {"language": "en", "seg": "Too long.", "tgts": [{
            "language": "fr", "base": None, "human": "Trop long."}]}
        path = self.write_rows("skipped.gz", [student])
        pairs = load_pairs(path)
        result = generate_batch(None, FakeTokenizer(), None, pairs, "prompt", self.args)[0]
        self.assertIsNone(result["record"]["tgts"][0]["corrected"])
        self.assertEqual(result["targets"][0]["finish_reason"], "skipped")

    def test_inconsistent_or_invented_edits_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_result(json.dumps(answer("Salut.")), "Bonjour.")
        result = answer("Salut.", [{"before": "Not present", "after": "Salut",
                                    "category": "meaning", "reason": "Example"}])
        with self.assertRaisesRegex(ValueError, "spans"):
            validate_result(json.dumps(result), "Bonjour.")

    def test_no_edit_boundary_whitespace_restores_original(self):
        for original, returned in (("Bonjour.\n", "Bonjour."),
                                   ("  Bonjour.\t", "Bonjour."),
                                   ("Bonjour.", "\nBonjour.\n")):
            with self.subTest(original=original):
                result = validate_result(json.dumps(answer(returned)), original)
                self.assertEqual(result["corrected_translation"], original)
                self.assertEqual(result["edits"], [])
        result = generate_batch(FakeLLM([answer("Bonjour.")]), FakeTokenizer(), None,
                                [(row(target="Bonjour.\n"), row())], "prompt", self.args)[0]
        self.assertEqual(result["record"]["tgts"][0]["corrected"], "Bonjour.")
        self.assertEqual(result["targets"][0]["edit_fraction"], 0)

    def test_no_edit_internal_whitespace_or_casing_changes_rejected(self):
        for original, returned in (("Bonjour  monde.", "Bonjour monde."),
                                   ("Bonjour\nmonde.", "Bonjour monde."),
                                   ("Bonjour.", "bonjour.")):
            with self.subTest(original=original):
                with self.assertRaisesRegex(ValueError, "disagree"):
                    validate_result(json.dumps(answer(returned)), original)

    def test_inconsistent_edits_saved_without_validation_or_retry(self):
        student = row(target="La carte pour le patient contient les messages clés.")
        corrected = "La carte patient contient les messages clés."
        actual = {"before": "carte pour le patient", "after": "carte patient",
                  "category": "terminology", "reason": "Domain term."}
        unapplied = {"before": "messages clés", "after": "messages clefs",
                     "category": "terminology", "reason": "Reference spelling."}
        sequence = [answer(corrected, [actual, unapplied]), answer(corrected, [actual])]
        calls = []
        def generate(prompts, sampling, **kwargs):
            calls.append(prompts)
            return FakeLLM([sequence.pop(0)]).generate(prompts, sampling)
        result = generate_batch(SimpleNamespace(generate=generate), FakeTokenizer(), None,
                                [(student, row())], "prompt", self.args)[0]
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["record"]["tgts"][0]["corrected"], corrected)
        self.assertEqual(result["targets"][0]["edits"], [actual, unapplied])
        self.assertNotIn("validation_retries", result["targets"][0])

    def test_unexplained_change_saved_without_retry(self):
        calls = []
        def generate(prompts, sampling, **kwargs):
            calls.append(prompts)
            return FakeLLM([answer("Unexplained change.")]).generate(prompts, sampling)
        result = generate_batch(SimpleNamespace(generate=generate), FakeTokenizer(), None,
                                [(row(), row())], "prompt", self.args)[0]
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["record"]["tgts"][0]["corrected"], "Unexplained change.")

    def test_truncation_recorded_without_stopping(self):
        result = generate_batch(FakeLLM([answer()], "length"), FakeTokenizer(), None,
                                [(row(), row())], "prompt", self.args)[0]
        self.assertEqual(result["targets"][0]["finish_reason"], "length")

    def test_malformed_response_is_saved_and_later_rows_continue(self):
        outputs = [SimpleNamespace(outputs=[SimpleNamespace(text="{broken", finish_reason="length")]),
                   SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(answer()), finish_reason="stop")])]
        llm = SimpleNamespace(generate=lambda *args, **kwargs: outputs)
        results = generate_batch(llm, FakeTokenizer(), None,
                                 [(row(), row()), (row(), row())], "prompt", self.args)
        self.assertIsNone(results[0]["record"]["tgts"][0]["corrected"])
        self.assertEqual(results[0]["targets"][0]["raw_response"], "{broken")
        self.assertIsNotNone(results[0]["targets"][0]["extraction_error"])
        self.assertEqual(results[1]["record"]["tgts"][0]["corrected"], "Bonjour.")

    def test_context_not_silently_truncated(self):
        self.args.max_model_len = 1025
        result = generate_batch(None, FakeTokenizer(), None,
                                [(row(), row())], "prompt", self.args)[0]
        self.assertIsNone(result["record"]["tgts"][0]["corrected"])
        self.assertEqual(result["record"]["tgts"][0]["base"], "Bonjour.")
        self.assertEqual(result["targets"][0]["extraction_error"], "context budget exceeded")

    def test_long_prompt_does_not_shift_later_outputs(self):
        pairs = [(row("x" * 5000), row("x" * 5000)), (row(), row())]
        results = generate_batch(FakeLLM([answer("Salut.")]), FakeTokenizer(), None,
                                 pairs, "prompt", self.args)
        self.assertIsNone(results[0]["record"]["tgts"][0]["corrected"])
        self.assertEqual(results[1]["record"]["tgts"][0]["corrected"], "Salut.")

    def test_context_skip_migration_preserves_other_settings(self):
        previous = {"script_sha256": "old", "max_model_len": 4096, "prompt": "same"}
        current = dict(previous, script_sha256="new", context_overflow_policy="skip")
        self.assertTrue(check_context_skip_migration(previous, current))
        self.assertFalse(check_context_skip_migration(previous, dict(current, prompt="changed")))
        self.assertFalse(check_context_skip_migration(previous, dict(current, max_model_len=8192)))
        self.assertFalse(check_context_skip_migration(current, dict(current, script_sha256="other")))

    def test_partial_checkpoint_recovery_and_export(self):
        path = self.directory / "audit.jsonl"
        entry = {"line": 1, "record": row(), "targets": []}
        with open(path, "w+b") as stream:
            stream.write((json.dumps(entry) + "\n").encode() + b'{"line": 2')
            completed = read_checkpoint(stream)
            self.assertEqual(completed, [entry])
            stream.seek(0)
            self.assertTrue(stream.read().endswith(b"\n"))
        output = self.directory / "result.jsonl.gz"
        export_dataset(output, completed)
        self.assertEqual(list(records(output)), [row()])

    def test_multi_target_matching_by_language(self):
        student = row()
        student["tgts"].append({"language": "de", "seg": "Hallo."})
        reference = row()
        reference["tgts"].insert(0, {"language": "de", "seg": "Guten Tag."})
        left = self.write_rows("left.gz", [student])
        right = self.write_rows("right.gz", [reference])
        pairs = load_pairs(left, right)
        result = generate_batch(FakeLLM([answer(), answer("Hallo.")]), FakeTokenizer(),
                                None, pairs, "prompt", self.args)[0]
        self.assertEqual(result["record"]["tgts"], [
            {"language": "fr", "base": "Bonjour.", "human": "Bonjour.",
             "corrected": "Bonjour."},
            {"language": "de", "base": "Hallo.", "human": "Guten Tag.",
             "corrected": "Hallo."}])

    def test_edit_fraction_punctuation_sensitive(self):
        self.assertEqual(edit_fraction("hello", "hello"), 0)
        self.assertGreater(edit_fraction("hello", "Hello!"), 0)

    def test_run_export_resume_and_manifest_guard(self):
        self.args.synthetic = self.write_rows("synthetic.gz", [row()])
        self.args.references = self.write_rows("references.gz", [row()])
        self.args.output = self.directory / "output.jsonl.gz"
        self.args.model = "fake-teacher"
        self.args.revision = "fixed"
        self.args.limit = None
        self.args.seed = 42
        self.args.tensor_parallel_size = 4
        self.args.gpu_memory_utilization = 0.9
        self.args.batch_size = 1
        self.args.resume = False
        llm = FakeLLM([answer()])
        llm.get_tokenizer = FakeTokenizer
        engine_calls = []

        def create_engine(**kwargs):
            engine_calls.append(kwargs)
            return llm

        modules = {"vllm": SimpleNamespace(LLM=create_engine, SamplingParams=lambda **kw: kw),
                   "vllm.sampling_params": SimpleNamespace(StructuredOutputsParams=lambda **kw: kw)}
        pairs = load_pairs(self.args.synthetic, self.args.references)
        with patch.dict("sys.modules", modules):
            run(self.args, pairs, "prompt")
        self.assertTrue(engine_calls[0]["language_model_only"])
        self.assertEqual(engine_calls[0]["gdn_prefill_backend"], "triton")
        self.assertNotIn("quantization", engine_calls[0])
        exported = list(records(self.args.output))
        self.assertEqual(exported[0]["tgts"][0], {
            "language": "fr", "base": "Bonjour.", "human": "Bonjour.",
            "corrected": "Bonjour."})
        with self.assertRaisesRegex(ValueError, "exists"):
            run(self.args, pairs, "prompt")
        self.args.resume = True
        # A complete checkpoint must not need vLLM or another inference pass.
        with patch.dict("sys.modules", {"vllm": None}):
            run(self.args, pairs, "prompt")
            manifest = Path(str(self.args.output) + ".manifest.json")
            previous = json.loads(manifest.read_text())
            previous.pop("context_overflow_policy")
            previous["script_sha256"] = "old"
            manifest.write_text(json.dumps(previous))
            run(self.args, pairs, "prompt")
            self.assertEqual(json.loads(manifest.read_text())["context_overflow_policy"], "skip")
            self.assertTrue(Path(str(self.args.output) + ".context-skip-migration.json").exists())
        with self.assertRaisesRegex(ValueError, "differs"):
            run(self.args, pairs, "changed prompt")

    def test_inconsistent_response_is_exported_without_stopping(self):
        self.args.synthetic = self.write_rows("synthetic.gz", [row()])
        self.args.references = None
        self.args.output = self.directory / "failed.jsonl.gz"
        self.args.model = "fake-teacher"
        self.args.revision = None
        self.args.limit = None
        self.args.seed = 42
        self.args.tensor_parallel_size = 4
        self.args.gpu_memory_utilization = 0.9
        self.args.batch_size = 1
        self.args.resume = False
        llm = FakeLLM([answer("Unexplained change.")])
        llm.get_tokenizer = FakeTokenizer
        modules = {"vllm": SimpleNamespace(LLM=lambda **kw: llm, SamplingParams=lambda **kw: kw),
                   "vllm.sampling_params": SimpleNamespace(StructuredOutputsParams=lambda **kw: kw)}
        with patch.dict("sys.modules", modules):
            run(self.args, load_pairs(self.args.synthetic), "prompt")
        self.assertEqual(list(records(self.args.output))[0]["tgts"][0]["corrected"],
                         "Unexplained change.")
        audit = json.loads(Path(str(self.args.output) + ".audit.jsonl").read_text())
        self.assertIn("Unexplained change", audit["targets"][0]["raw_response"])


if __name__ == "__main__":
    unittest.main()
