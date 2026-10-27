#!/usr/bin/env python3
"""Inspect base/human/corrected edits and frozen TranslateGemma likelihoods.

Uses the CUDA training environment, not vLLM. No adapters or training are used.
Outputs PREFIX.sentences.jsonl, PREFIX.sentences.tsv, and PREFIX.summary.json.
"""

import argparse
import csv
from itertools import combinations
import json
import math
from pathlib import Path

if __package__:
    from .compare_translations import tokenize
    from .post_edit_translations import digest, open_text
    from .train_translation import load_model, prompt_ids
else:
    from compare_translations import tokenize
    from post_edit_translations import digest, open_text
    from train_translation import load_model, prompt_ids

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('base', 'human', 'corrected')
PAIRS = tuple(combinations(VARIANTS, 2))


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def distances(left, right):
    result = {'identical': left == right}
    for name, a, b in [('character', left, right),
                       ('word', tokenize(left), tokenize(right))]:
        distance = levenshtein(a, b)
        denominator = max(len(a), len(b))
        result[name] = {'distance': distance, 'normalizer': denominator,
                        'normalized': distance / denominator if denominator else 0.0}
    return result


def examples(path, target_language=None, limit=None):
    with open_text(path) as stream:
        for line, text in enumerate(stream, 1):
            if limit is not None and line > limit:
                break
            row = json.loads(text)
            if (not isinstance(row, dict) or not isinstance(row.get('seg'), str)
                    or not row['seg'].strip() or not isinstance(row.get('language'), str)
                    or not isinstance(row.get('tgts'), list)):
                raise ValueError(f'line {line}: expected source language/seg and tgts list')
            targets = row['tgts']
            if any(not isinstance(t, dict) for t in targets):
                raise ValueError(f'line {line}: targets must be objects')
            if target_language:
                targets = [t for t in targets if t.get('language') == target_language]
            if len(targets) != 1 or not isinstance(targets[0].get('language'), str):
                raise ValueError(f'line {line}: select exactly one target using --target-language')
            target = targets[0]
            for field in VARIANTS:
                if target.get(field) is not None and not isinstance(target[field], str):
                    raise ValueError(f'line {line}: {field} must be a string or null')
            yield line, row, target


def score_candidate(model, tokenizer, prefix, text, max_length):
    import torch
    import torch.nn.functional as functional

    if text is None or not text.strip():
        return {'status': 'missing_or_empty'}
    # Preserve candidate whitespace exactly. Score translation text only, excluding
    # end-of-turn, so all three variants use the same explicit scoring convention.
    completion = tokenizer.encode(text, add_special_tokens=False)
    if not completion:
        return {'status': 'empty_tokenization'}
    if not prefix:
        raise ValueError('translation prompt has no tokens')
    if len(prefix) + len(completion) > max_length:
        return {'status': 'context_budget_exceeded', 'tokens': len(completion)}
    ids = torch.tensor([prefix + completion], device=model.device)
    # Position prefix_len-1 predicts the first candidate token. Keep only the
    # logits needed for candidate tokens; the final input token need not be fed.
    positions = torch.arange(len(prefix) - 1, ids.shape[1] - 1, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=ids[:, :-1], attention_mask=torch.ones_like(ids[:, :-1]),
                       use_cache=False, logits_to_keep=positions).logits[0]
        losses = []
        labels = ids[0, len(prefix):]
        for start in range(0, len(completion), 32):
            losses.extend(functional.cross_entropy(
                logits[start:start + 32].float(), labels[start:start + 32],
                reduction='none').double().cpu().tolist())
    total = math.fsum(losses)
    nll = total / len(completion)
    return {'status': 'ok', 'tokens': len(completion), 'total_logprob': -total,
            'nll': nll, 'perplexity': safe_exp(nll)}


def safe_exp(value):
    return math.exp(value) if value < 709 else None


def describe(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    n = len(ordered)
    def quantile(p):
        index = (n - 1) * p
        lo = int(index)
        hi = min(lo + 1, n - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {'count': n, 'mean': math.fsum(values) / n,
            'median': quantile(.5), 'p10': quantile(.1), 'p90': quantile(.9)}


def summarize(rows):
    summary = {'sentences': len(rows), 'likelihood': {}, 'edit_distance': {}}
    complete = [r for r in rows if all(r['scores'][v]['status'] == 'ok' for v in VARIANTS)]
    for label, subset in [('all_available', rows), ('complete_triplets', complete)]:
        group = {}
        for variant in VARIANTS:
            scores = [r['scores'][variant] for r in subset if r['scores'][variant]['status'] == 'ok']
            tokens = sum(s['tokens'] for s in scores)
            total = -math.fsum(s['total_logprob'] for s in scores)
            micro = total / tokens if tokens else None
            group[variant] = {'sentence_nll': describe([s['nll'] for s in scores]),
                              'tokens': tokens, 'token_weighted_nll': micro,
                              'corpus_perplexity': safe_exp(micro) if micro is not None else None}
        summary['likelihood'][label] = group
    summary['complete_triplets'] = len(complete)
    summary['interpretation'] = ('Lower NLL means more predictable under the original model, '
        'not necessarily a better translation. NLL differences are not distributional KL.')
    summary['status_counts'] = {v: {status: sum(r['scores'][v]['status'] == status for r in rows)
        for status in sorted({r['scores'][v]['status'] for r in rows})} for v in VARIANTS}
    summary['paired_nll_deltas_complete_triplets'] = {
        f'{b}_minus_{a}': describe([r['scores'][b]['nll'] - r['scores'][a]['nll'] for r in complete])
        for a, b in PAIRS}
    for a, b in PAIRS:
        key = f'{a}_vs_{b}'
        edits = [r['distances'][key] for r in rows if key in r['distances']]
        identical = sum(e['identical'] for e in edits)
        item = {'pairs': len(edits), 'identical': identical,
                'changed_fraction': (len(edits) - identical) / len(edits) if edits else None}
        for unit in ('character', 'word'):
            denominator = sum(e[unit]['normalizer'] for e in edits)
            item[unit] = {'sentence_normalized': describe([e[unit]['normalized'] for e in edits]),
                          'pooled_normalized': sum(e[unit]['distance'] for e in edits) / denominator
                          if denominator else None}
        summary['edit_distance'][key] = item
    return summary


def flatten(row):
    result = {k: row[k] for k in ('line', 'source_language', 'target_language', 'source')}
    for variant in VARIANTS:
        result[variant] = row['translations'][variant]
        for key in ('status', 'tokens', 'total_logprob', 'nll', 'perplexity'):
            result[f'{variant}_{key}'] = row['scores'][variant].get(key)
    for a, b in PAIRS:
        key = f'{a}_vs_{b}'
        edit = row['distances'].get(key, {})
        result[f'{key}_identical'] = edit.get('identical')
        for unit in ('character', 'word'):
            for metric in ('distance', 'normalized'):
                result[f'{key}_{unit}_{metric}'] = edit.get(unit, {}).get(metric)
        result[f'nll_{b}_minus_{a}'] = row['nll_deltas'].get(f'{b}_minus_{a}')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Post-edited JSONL, optionally .gz')
    parser.add_argument('--output-prefix', type=Path, help='Default: INPUT without .gz/.jsonl + .distribution')
    parser.add_argument('--model', default='/lustre/fsmisc/dataset/HuggingFace_Models/google/translategemma-12b-it')
    parser.add_argument('--prompt', type=Path, default=ROOT / 'prompts/base_translation.txt')
    parser.add_argument('--target-language', help='Required when a record has multiple targets')
    parser.add_argument('--max-length', type=int, default=4096, help='Prompt + candidate token budget; never truncate')
    parser.add_argument('--limit', type=int, help='Inspect first N records')
    parser.add_argument('--attention-implementation', choices=('sdpa', 'eager', 'flash_attention_2'), default='sdpa')
    parser.add_argument('--dry-run', action='store_true', help='Validate input and compute edits without loading model/tokenizer')
    args = parser.parse_args(argv)
    if args.max_length <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('token budget and limit must be positive')
    if args.output_prefix is None:
        prefix = str(args.input)
        for suffix in ('.gz', '.jsonl', '.json'):
            if prefix.endswith(suffix):
                prefix = prefix[:-len(suffix)]
        args.output_prefix = Path(prefix + '.distribution')
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = [Path(str(args.output_prefix) + suffix) for suffix in
             ('.sentences.jsonl', '.sentences.tsv', '.summary.json')]
    if any(p.exists() or p.resolve() in (args.input.resolve(), args.prompt.resolve()) for p in paths):
        raise ValueError('outputs must be new paths, distinct from input and prompt')
    prompt = args.prompt.read_text(encoding='utf-8')
    if not prompt.strip():
        raise ValueError('empty translation prompt')
    # Validate selected records before loading the 12B checkpoint.
    selected = list(examples(args.input, args.target_language, args.limit))
    if not selected:
        raise ValueError('no input records')
    model = tokenizer = None
    if not args.dry_run:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = load_model(args)
        model.eval()
        model.requires_grad_(False)
    rows = []
    for line, original, target in selected:
        texts = {v: target.get(v) for v in VARIANTS}
        row = {'line': line, 'source_language': original['language'],
               'target_language': target['language'], 'source': original['seg'],
               'translations': texts, 'distances': {}, 'scores': {}, 'nll_deltas': {}}
        for a, b in PAIRS:
            if all(isinstance(texts[v], str) and texts[v].strip() for v in (a, b)):
                row['distances'][f'{a}_vs_{b}'] = distances(texts[a], texts[b])
        if not args.dry_run:
            args.source_language = original['language']
            language = args.target_language
            args.target_language = target['language']
            prefix = prompt_ids(tokenizer, prompt, original['seg'], args)
            args.target_language = language
        cache = {}
        for variant, text in texts.items():
            if args.dry_run:
                score = {'status': 'not_scored_dry_run' if text and text.strip() else 'missing_or_empty'}
            else:
                if text not in cache:
                    cache[text] = score_candidate(model, tokenizer, prefix, text, args.max_length)
                score = cache[text]
            row['scores'][variant] = score
        for a, b in PAIRS:
            if all(row['scores'][v]['status'] == 'ok' for v in (a, b)):
                row['nll_deltas'][f'{b}_minus_{a}'] = row['scores'][b]['nll'] - row['scores'][a]['nll']
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f'Inspected {len(rows)}/{len(selected)} sentences', flush=True)
    summary = summarize(rows)
    summary['configuration'] = {'model': args.model, 'input': str(args.input),
        'input_sha256': digest(args.input), 'prompt': prompt, 'script_sha256': digest(Path(__file__)),
        'max_length': args.max_length, 'limit': args.limit, 'dry_run': args.dry_run,
        'model_commit': getattr(model.config, '_commit_hash', None) if model is not None else None,
        'attention_implementation': args.attention_implementation, 'dtype': 'bfloat16',
        'scoring': 'teacher-forced candidate text only; no end-of-turn; no prompt loss; natural log',
        'normalization': 'candidate model tokens for NLL; max pair length for edit distance',
        'whitespace': 'preserved; word edits use punctuation-aware regex tokens'}
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    with paths[1].open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flatten(rows[0])), delimiter='\t')
        writer.writeheader()
        writer.writerows(flatten(row) for row in rows)
    paths[2].write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print('Reports:', ', '.join(str(p) for p in paths))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError) as error:
        raise SystemExit(f'Error: {error}') from error
