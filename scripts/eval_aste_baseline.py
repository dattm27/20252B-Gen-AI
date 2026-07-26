"""CLI: train and evaluate the frequency-lookup ASTE baseline (src/baseline/aste_lookup_baseline.py)
on the SemEval Restaurant Triplet data (14res+15res+16res), for a fair non-neural reference point
against the T5 models (notebooks/train-{t5-small,t5-base,flan-t5-base}-for-aste-*.ipynb) — same
task, same triplet-F1 metric.

--data-dir isn't committed (data/raw/ is gitignored) — clone it first:
    git clone --depth 1 https://github.com/xuuuluuu/SemEval-Triplet-data.git \
        data/raw/semeval-triplet-data

Usage:
    python scripts/eval_aste_baseline.py \
        --data-dir data/raw/semeval-triplet-data/ASTE-Data-V1-AAAI2020
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline.aste_lookup_baseline import AsteLookupBaseline
from src.data.aste_loader import corpus_triplet_prf, load_aste_file

DOMAINS = ["14res", "15res", "16res"]


def find_split_file(domain_dir: Path, split: str) -> Path:
    """File naming is inconsistent across domains in the upstream repo (e.g. `train.txt` for
    14res, `15rest_train.txt` for 15res) — same fallback-by-substring logic used by the ASTE
    training notebooks' `find_split_file`."""
    files = sorted(domain_dir.glob("*.txt"))
    for f in files:
        if f.stem.lower() == split or f.name.lower() == f"{domain_dir.name}rest_{split}.txt":
            return f
    for f in files:
        if split in f.name.lower():
            return f
    raise FileNotFoundError(f"No {split} file found in {domain_dir} (files: {[f.name for f in files]})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing 14res/15res/16res subfolders, each with train.txt/dev.txt/test.txt",
    )
    parser.add_argument("--metrics-out", default="results/aste_baseline_metrics.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    train_sentences = []
    test_sentences = []
    for domain in DOMAINS:
        domain_dir = data_dir / domain
        train_sentences.extend(load_aste_file(find_split_file(domain_dir, "train")))
        test_sentences.extend(load_aste_file(find_split_file(domain_dir, "test")))

    print(f"Train: {len(train_sentences)} sentences | Test: {len(test_sentences)} sentences")

    model = AsteLookupBaseline().fit(train_sentences)
    print(f"Aspect vocab: {len(model.aspect_vocab)} phrases | Opinion lexicon: {len(model.opinion_sentiment)} phrases")

    pred_triplets_per_sentence = [model.predict(sent.text) for sent in test_sentences]
    gold_triplets_per_sentence = [sent.triplets for sent in test_sentences]

    precision, recall, f1 = corpus_triplet_prf(pred_triplets_per_sentence, gold_triplets_per_sentence)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Triplet-F1: {f1:.4f}")

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": "AsteLookupBaseline",
        "train_sentences": len(train_sentences),
        "test_sentences": len(test_sentences),
        "aspect_vocab_size": len(model.aspect_vocab),
        "opinion_lexicon_size": len(model.opinion_sentiment),
        "triplet_precision": precision,
        "triplet_recall": recall,
        "triplet_f1": f1,
    }, indent=2))
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
