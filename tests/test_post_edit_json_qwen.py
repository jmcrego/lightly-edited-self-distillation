import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[1] / 'scripts/post_edit_json_qwen3.5-122B.py'
SPEC = importlib.util.spec_from_file_location('scripts.post_edit_json_qwen', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PostEditJsonTests(unittest.TestCase):
    def row(self):
        return {'language': 'en', 'seg': 'Hello.', 'id': 3, 'tgts': [
            {'language': 'fr', 'seg': 'Bonjour.'},
            {'language': 'fr', 'translategemma-12b': 'Salut.'},
            {'language': 'fr', 'qwen3.5-122b': 'Original Qwen'}]}

    def test_prompt_and_preservation(self):
        calls = []

        def template(messages, **kwargs):
            calls.append((messages, kwargs))
            return 'prompt'

        tokenizer = SimpleNamespace(apply_chat_template=template, encode=lambda *a, **kw: [1])
        response = json.dumps({'corrected_translation': 'Salut.', 'edits': []})
        llm = SimpleNamespace(generate=lambda *a, **kw: [SimpleNamespace(outputs=[
            SimpleNamespace(text=response, finish_reason='stop')])])
        row = self.row()
        entry = MODULE.generate_batch(llm, tokenizer, None, [(row, None)], 'minimal prompt',
            SimpleNamespace(max_model_len=100, max_new_tokens=10, review_edit_fraction=.3))[0]
        expected = self.row()
        expected['tgts'][1][MODULE.OUTPUT_FIELD] = 'Salut.'
        self.assertEqual(entry['record'], expected)
        self.assertNotIn(MODULE.OUTPUT_FIELD, row['tgts'][1])
        payload = json.loads(calls[0][0][1]['content'])
        self.assertEqual(payload['student_translation'], 'Salut.')
        self.assertEqual(payload['human_reference'], 'Bonjour.')
        self.assertNotIn('Original Qwen', json.dumps(calls))
        self.assertFalse(calls[0][1]['enable_thinking'])

    def test_missing_or_ambiguous_reference(self):
        row = self.row()
        row['tgts'].pop(0)
        with self.assertRaisesRegex(ValueError, 'missing human reference'):
            MODULE.canonical_pair(row)
        row = self.row()
        row['tgts'].append({'language': 'fr', 'seg': 'Duplicate'})
        with self.assertRaisesRegex(ValueError, 'duplicate seg'):
            MODULE.canonical_pair(row)
