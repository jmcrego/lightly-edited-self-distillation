
#!/usr/bin/env python3
"""Compare two aligned comet-score --to_json detail files with paired bootstrap.

Example: python scripts/paired_boostrap_comet.py a.comet.details b.comet.details
Requires NumPy; no COMET model or GPU is needed.
"""

import argparse
import json
import numpy as np

def load_scores(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # comet-score stores one list per hypothesis file. Never silently select
    # a system from a file containing multiple systems.
    if isinstance(data, dict):
        if len(data) != 1:
            raise ValueError(f"{path}: expected exactly one system")
        records = next(iter(data.values()))
    else:
        records = data
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty list of sentence records")
    for i, record in enumerate(records, 1):
        if not isinstance(record, dict) or not isinstance(record.get("src"), str):
            raise ValueError(f"{path}: sentence {i} must contain a string 'src'")
        score = record.get("COMET")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not np.isfinite(score)):
            raise ValueError(f"{path}: sentence {i} must contain a finite numeric 'COMET' score")
    return records

def compare_systems(path_a, path_b, n_boot=20000, seed=42, confidence=0.95):
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    a = load_scores(path_a)
    b = load_scores(path_b)

    if len(a) != len(b):
        raise ValueError(f"Different sentence counts: A={len(a)}, B={len(b)}")

    # Ensure that the paired sentences are aligned.
    for i, (x, y) in enumerate(zip(a, b), 1):
        for field in ("src", "ref"):
            if (field in x) != (field in y) or x.get(field) != y.get(field):
                raise ValueError(f"Sentence {i}: {field!r} differs between systems")

    scores_a = np.array([x["COMET"] for x in a])
    scores_b = np.array([x["COMET"] for x in b])

    diff = scores_b - scores_a
    rng = np.random.default_rng(seed)

    bootstrap = np.empty(n_boot)
    for i in range(n_boot):
        indices = rng.integers(0, len(diff), len(diff))
        bootstrap[i] = diff[indices].mean()

    tail = 100 * (1 - confidence) / 2
    ci = np.percentile(bootstrap, [tail, 100 - tail])

    print(f"System A file: {path_a}")
    print(f"System B file: {path_b}")
    print(f"Sentences: {len(diff)}; bootstrap samples: {n_boot}; seed: {seed}")
    print(f"System A: {scores_a.mean():.6f}")
    print(f"System B: {scores_b.mean():.6f}")
    print(f"Difference (B - A): {diff.mean():+.6f}")
    print(f"{100 * confidence:g}% bootstrap CI: [{ci[0]:+.6f}, {ci[1]:+.6f}]")
    print(f"Significant at {100 * (1 - confidence):g}%: {bool(ci[0] > 0 or ci[1] < 0)}")

    print(f"B better: {(diff > 0).sum()} sentences")
    print(f"A better: {(diff < 0).sum()} sentences")
    print(f"Ties: {(diff == 0).sum()} sentences")

    return diff, a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path_a", help="System A COMET detail JSON")
    parser.add_argument("path_b", help="System B COMET detail JSON")
    parser.add_argument("--n-boot", type=int, default=20000, help="Bootstrap samples (default: 20000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level (default: 0.95)")
    args = parser.parse_args()
    try:
        compare_systems(args.path_a, args.path_b, args.n_boot, args.seed, args.confidence)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
