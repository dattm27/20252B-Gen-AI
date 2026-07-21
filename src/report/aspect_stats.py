"""Aggregate per-example (aspect, sentiment) pairs into per-aspect sentiment counts.

Pure Python/stdlib on purpose: the sentiment labels themselves come from either the
SemEval gold XML or a fine-tuned model's predictions (both produced elsewhere, e.g. on
Kaggle), but turning a flat list of examples into a summary table is cheap and doesn't
need GPU/transformers, so it runs and is tested locally like the rest of `src/`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

SENTIMENT_LABELS = ("positive", "negative", "neutral")


@dataclass
class AspectSummary:
    aspect: str
    positive: int
    negative: int
    neutral: int
    total: int
    majority_sentiment: str


def aggregate_aspect_sentiment(
    records: list[tuple[str, str]], min_mentions: int = 1
) -> list[AspectSummary]:
    """Group (aspect, sentiment) pairs by aspect (case/whitespace-insensitive) and count
    each sentiment. Returns one `AspectSummary` per distinct aspect, sorted by total
    mentions descending (ties broken alphabetically), skipping aspects with fewer than
    `min_mentions` total mentions.
    """
    counts: dict[str, Counter] = {}
    for aspect, sentiment in records:
        key = aspect.strip().lower()
        if not key or sentiment not in SENTIMENT_LABELS:
            continue
        counts.setdefault(key, Counter())[sentiment] += 1

    summaries = []
    for aspect, counter in counts.items():
        total = sum(counter.values())
        if total < min_mentions:
            continue
        majority_sentiment = max(SENTIMENT_LABELS, key=lambda label: (counter[label], -SENTIMENT_LABELS.index(label)))
        summaries.append(
            AspectSummary(
                aspect=aspect,
                positive=counter["positive"],
                negative=counter["negative"],
                neutral=counter["neutral"],
                total=total,
                majority_sentiment=majority_sentiment,
            )
        )

    summaries.sort(key=lambda s: (-s.total, s.aspect))
    return summaries
