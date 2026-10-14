import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.translate_tsv import generate_batch, parse_args, run


PATH = Path(__file__).resolve().parents[1] / "scripts/translate_json_qwen3.5-122B.py"
SPEC = importlib.util.spec_from_file_location("scripts.translate_qwen", PATH)
QWEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QWEN)


class QwenTests(unittest.TestCase):
    def test_native_template_and_source_only_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "prompt.txt"
            prompt.write_text("Translate {source_language} into {target_language}.")
            args = SimpleNamespace(prompt=prompt, source_language="en", target_language="fr",
                                   max_model_len=100, max_new_tokens=10)
            calls = []

            def template(messages, **kwargs):
                calls.append((messages, kwargs))
                return [1, 2, 3]

            tokenizer = SimpleNamespace(apply_chat_template=template)
            response = SimpleNamespace(outputs=[SimpleNamespace(text="Bonjour.\n", finish_reason="stop")])
            llm = SimpleNamespace(generate=lambda *a, **kw: [response])
            reference = {"language": "en", "seg": "Hello.",
                         "tgts": [{"language": "fr", "human": "Reference not sent"}]}
            entries = generate_batch(llm, tokenizer, None, [reference], args,
                                     prompt_builder=QWEN.prompt_builder, output_field=QWEN.OUTPUT_FIELD)
            messages, kwargs = calls[0]
            self.assertEqual(messages[0]["content"], "Translate en into fr.")
            self.assertEqual(messages[1]["content"], "Hello.")
            self.assertFalse(kwargs["enable_thinking"])
            self.assertNotIn("chat_template", kwargs)
            self.assertEqual(entries[0]["record"]["tgts"][0][QWEN.OUTPUT_FIELD], "Bonjour.")
            self.assertNotIn("base", entries[0]["record"]["tgts"][0])
            self.assertEqual(entries[0]["record"]["tgts"][0]["human"], "Reference not sent")

    def test_json_preservation_named_output_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.jsonl"
            original = {"language": "en", "seg": "Hello.", "id": 7, "tgts": [
                {"language": "de", "seg": "Hallo."}, {"language": "fr", "seg": "Reference"}]}
            source.write_text(json.dumps(original) + "\n")
            prompt = directory / "prompt.txt"
            prompt.write_text("Translate {source_language} to {target_language}.")
            args = parse_args(["--input", str(source), "--output", str(directory / "output.jsonl"),
                "--prompt", str(prompt), "--source-language", "en", "--target-language", "fr"], require_gzip=False)
            records = QWEN.load_json(source, args)
            tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **kw: [1, 2, 3])
            response = SimpleNamespace(outputs=[SimpleNamespace(text="Bonjour.", finish_reason="stop")])
            llm = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=lambda *a, **kw: [response])
            modules = {"vllm": SimpleNamespace(LLM=lambda **kw: llm, SamplingParams=lambda **kw: kw)}
            with patch.dict("sys.modules", modules):
                run(args, records, prompt_builder=QWEN.prompt_builder, driver_path=PATH, output_field=QWEN.OUTPUT_FIELD)
            result = json.loads(args.output.read_text())
            expected = json.loads(json.dumps(original))
            expected["tgts"][1][QWEN.OUTPUT_FIELD] = "Bonjour."
            self.assertEqual(result, expected)
            args.resume = True
            with patch.dict("sys.modules", {"vllm": None}):
                run(args, records, prompt_builder=QWEN.prompt_builder, driver_path=PATH, output_field=QWEN.OUTPUT_FIELD)
            self.assertEqual(json.loads(args.output.read_text()), expected)

    def test_qwen_defaults_and_cli_overrides(self):
        common = ["--input", "in.tsv", "--output", "out.jsonl.gz", "--source-language", "en", "--target-language", "fr"]
        args = parse_args(common, defaults={"model": "qwen", "tensor_parallel_size": 4})
        self.assertEqual(args.model, "qwen")
        self.assertEqual(args.tensor_parallel_size, 4)
        args = parse_args(common + ["--tensor-parallel-size", "2"], defaults={"tensor_parallel_size": 4})
        self.assertEqual(args.tensor_parallel_size, 2)
