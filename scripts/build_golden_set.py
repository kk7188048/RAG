"""Samples real Q&A pairs from the ingested MedQuAD corpus to seed
eval/golden_set.json, spread evenly across all source categories.

These are genuine NIH-sourced ground truth answers, not invented ones.
Review and edit them before treating this as your final golden set --
swap in your own questions/phrasing where useful, this just avoids
starting from a blank file.

Usage: python scripts/build_golden_set.py [--n-per-source 6]
"""

import argparse
import json
import random
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent / "ingest" / "corpus"
OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden_set.json"
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-source", type=int, default=6)
    args = parser.parse_args()

    rng = random.Random(SEED)
    golden = []

    for json_file in sorted(CORPUS_DIR.glob("*.json")):
        records = json.loads(json_file.read_text())
        # prefer reasonably concise ground-truth answers for cleaner eval
        candidates = [r for r in records if 40 < len(r["answer"]) < 1200]
        if not candidates:
            candidates = records
        sample = rng.sample(candidates, min(args.n_per_source, len(candidates)))
        for r in sample:
            golden.append(
                {
                    "question": r["question"],
                    "ground_truth": r["answer"],
                    "focus_area": r["focus_area"],
                    "source": json_file.stem,
                }
            )

    OUT_PATH.write_text(json.dumps(golden, indent=2))
    print(f"Wrote {len(golden)} golden Q&A pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
