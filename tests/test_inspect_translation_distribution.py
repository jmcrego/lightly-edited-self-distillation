import importlib.util
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.inspect_translation_distribution import (
    distances, examples, main, score_candidate, summarize,
)


class InspectionTests(unittest.TestCase):
    def test_edit_distance_uses_levenshtein_and_max_length(self):
        edits = distances('a b c', 'a x c d')
        self.assertEqual(edits['word']['distance'], 2)
        self.assertEqual(edits['word']['normalized'], .5)
        self.assertFalse(edits['identical'])
        self.assertTrue(distances('identical', 'identical')['identical'])
        self.assertEqual(distances('Hello.', 'hello!')['word']['distance'], 2)

    def test_complete_triplets_exclude_unpaired_scores(self):
        def score(nll, tokens):
            return {'status': 'ok', 'tokens': tokens, 'nll': nll,
                    'total_logprob': -nll * tokens}
        rows = [
            {'scores': {'base': score(1, 2), 'human': score(3, 4), 'corrected': score(2, 3)},
             'distances': {}},
            {'scores': {'base': score(5, 6), 'human': score(6, 8),
                        'corrected': {'status': 'context_budget_exceeded'}}, 'distances': {}},
        ]
        result = summarize(rows)
        self.assertEqual(result['complete_triplets'], 1)
        self.assertEqual(result['likelihood']['complete_triplets']['base']['token_weighted_nll'], 1)
        self.assertEqual(result['likelihood']['all_available']['base']['token_weighted_nll'], 4)
        self.assertEqual(result['paired_nll_deltas_complete_triplets']['corrected_minus_base']['mean'], 1)
        self.assertEqual(result['status_counts']['corrected']['context_budget_exceeded'], 1)

    def test_dry_run_reports_text_edits_and_missing_scores_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.jsonl'
            row = {'language': 'en', 'seg': 'Hello.', 'tgts': [
                {'language': 'fr', 'base': 'Salut.', 'human': 'Bonjour.', 'corrected': 'Salut.'}]}
            source.write_text(json.dumps(row) + '\n')
            prefix = root / 'scores'
            with redirect_stdout(io.StringIO()):
                main(['--input', str(source), '--output-prefix', str(prefix), '--dry-run'])
            report = json.loads(Path(str(prefix) + '.sentences.jsonl').read_text())
            self.assertEqual(report['translations']['human'], 'Bonjour.')
            self.assertTrue(report['distances']['base_vs_corrected']['identical'])
            self.assertEqual(report['scores']['base']['status'], 'not_scored_dry_run')
            summary = json.loads(Path(str(prefix) + '.summary.json').read_text())
            self.assertEqual(summary['complete_triplets'], 0)
            self.assertEqual(summary['sentences'], 1)
            with self.assertRaisesRegex(ValueError, 'new paths'):
                main(['--input', str(source), '--output-prefix', str(prefix), '--dry-run'])
            row['tgts'].append({'language': 'de', 'base': 'Hallo.'})
            source.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'select exactly one target'):
                list(examples(source))
            self.assertEqual(list(examples(source, 'fr'))[0][2]['base'], 'Salut.')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch needed for numerical likelihood check')
    def test_teacher_forcing_alignment_and_target_only_nll(self):
        import torch
        class Model:
            device = 'cpu'
            def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep):
                # Prefix [0, 1], target [2, 3]: positions 1 and 2 predict 2 and 3.
                self_input = input_ids.tolist()
                self_test.assertEqual(self_input, [[0, 1, 2]])
                self_test.assertEqual(logits_to_keep.tolist(), [1, 2])
                self_test.assertFalse(use_cache)
                return SimpleNamespace(logits=torch.tensor([[[0., 0., 2., 0.],
                                                              [0., 0., 0., 1.]]]))
        self_test = self
        model = Model()
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [2, 3])
        score = score_candidate(model, tokenizer, [0, 1], 'candidate', 4)
        expected_total = math.log(math.exp(2) + 3) - 2 + math.log(math.e + 3) - 1
        self.assertAlmostEqual(score['total_logprob'], -expected_total, places=6)
        self.assertAlmostEqual(score['nll'], expected_total / 2, places=6)
        self.assertEqual(score['tokens'], 2)
        self.assertEqual(score_candidate(None, tokenizer, [0, 1], 'long', 3)['status'],
                         'context_budget_exceeded')
        self.assertEqual(score_candidate(None, tokenizer, [0], None, 4)['status'], 'missing_or_empty')


if __name__ == '__main__':
    unittest.main()
