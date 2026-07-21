"""CLI: split a SemEval-2014 Task 4 train XML into train/valid/test XML files.

The official test files (data/raw/test) are blind — no gold polarity — so
evaluation has to come from a held-out split of the labeled train file instead. This script performs that split once and writes
the three resulting XML files (same schema as the input) so every downstream
script/notebook can load a stable, reproducible train/valid/test.

Usage:
    python scripts/split_dataset.py \
        --input data/raw/train/Laptop_Train_v2.xml \
        --output-dir data/processed/laptop
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_sentences(
    sentences: list[ET.Element], valid_ratio: float, test_ratio: float, seed: int
) -> tuple[list[ET.Element], list[ET.Element], list[ET.Element]]:
    train_val, test = train_test_split(sentences, test_size=test_ratio, random_state=seed)
    valid_size = valid_ratio / (1.0 - test_ratio)
    train, valid = train_test_split(train_val, test_size=valid_size, random_state=seed)
    return train, valid, test


def write_split(sentences: list[ET.Element], path: Path) -> None:
    root = ET.Element("sentences")
    root.extend(sentences)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def count_aspect_terms(sentences: list[ET.Element]) -> int:
    return sum(len(sent.findall("./aspectTerms/aspectTerm")) for sent in sentences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the labeled train XML (e.g. Laptop_Train_v2.xml)")
    parser.add_argument("--output-dir", required=True, help="Directory to write train.xml/valid.xml/test.xml into")
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = ET.parse(args.input).getroot()
    sentences = root.findall("sentence")

    train, valid, test = split_sentences(sentences, args.valid_ratio, args.test_ratio, args.seed)

    out_dir = Path(args.output_dir)
    splits = {"train": train, "valid": valid, "test": test}
    for name, split_sentences_ in splits.items():
        write_split(split_sentences_, out_dir / f"{name}.xml")

    print(f"Input: {args.input} ({len(sentences)} sentences)")
    for name, split_sentences_ in splits.items():
        print(
            f"{name:>5}: {len(split_sentences_):4d} sentences, "
            f"{count_aspect_terms(split_sentences_):4d} aspect terms -> {out_dir / f'{name}.xml'}"
        )


if __name__ == "__main__":
    main()
