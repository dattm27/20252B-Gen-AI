"""CLI: train and evaluate the TF-IDF + Logistic Regression ABSA baseline.

Usage (with a real gold test file):
    python scripts/train_baseline.py \
        --train data/raw/train/Laptop_Train_v2.xml \
        --test data/raw/test/Laptops_Test_Gold.xml

Usage (no gold test yet — hold out a dev split from the train file instead):
    python scripts/train_baseline.py --train data/raw/train/Laptop_Train_v2.xml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from src.baseline.tfidf_logreg import TfidfLogRegBaseline
from src.data.preprocess import build_examples
from src.data.semeval_loader import load_semeval_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Path to Laptop_Train_v2.xml")
    parser.add_argument(
        "--test",
        default=None,
        help="Path to the gold test XML. If omitted, a dev split is held out from --train instead.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Fraction of sentences held out as a dev/test split when --test is not given.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-out", default="results/baseline_metrics.json")
    args = parser.parse_args()

    all_train_sentences = load_semeval_xml(args.train)

    if args.test:
        train_sentences = all_train_sentences
        test_sentences = load_semeval_xml(args.test)
        split_note = "test=gold file"
    else:
        train_sentences, test_sentences = train_test_split(
            all_train_sentences, test_size=args.dev_ratio, random_state=args.seed
        )
        split_note = (
            f"test=dev split held out from --train "
            f"(no gold test file given, dev_ratio={args.dev_ratio}, seed={args.seed})"
        )

    train_ctx, train_asp, train_labels = build_examples(train_sentences)
    test_ctx, test_asp, test_labels = build_examples(test_sentences)

    print(f"Split: {split_note}")
    print(f"Train examples: {len(train_labels)} | Test examples: {len(test_labels)}")

    model = TfidfLogRegBaseline().fit(train_ctx, train_asp, train_labels)
    result = model.evaluate(test_ctx, test_asp, test_labels)

    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro-F1: {result.macro_f1:.4f}")
    print(result.report)

    out_path = Path(args.metrics_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"accuracy": result.accuracy, "macro_f1": result.macro_f1}, ensure_ascii=False, indent=2)
    )
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
