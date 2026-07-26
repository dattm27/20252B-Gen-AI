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


@dataclass
class AspectReasonSummary:
    aspect: str
    positive: int
    positive_reasons: list[tuple[str, int]]
    negative: int
    negative_reasons: list[tuple[str, int]]
    neutral: int
    neutral_reasons: list[tuple[str, int]]
    total: int
    majority_sentiment: str


def _top_reasons(counter: Counter, top_n: int) -> list[tuple[str, int]]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_n]


def aggregate_aspect_reasons(
    records: list[tuple[str, str, str]], top_n: int = 10, min_mentions: int = 1
) -> list[AspectReasonSummary]:
    """Group (aspect, opinion, sentiment) triples by aspect (case/whitespace-insensitive).
    For each aspect, count each sentiment (like `aggregate_aspect_sentiment`) and, within each
    sentiment, rank distinct opinion phrases (also case/whitespace-insensitive) by frequency,
    keeping the top `top_n` as example "reasons". Returns one `AspectReasonSummary` per aspect,
    sorted by total mentions descending (ties broken alphabetically), skipping aspects with
    fewer than `min_mentions` total mentions.
    """
    counts: dict[str, Counter] = {}
    reason_counts: dict[str, dict[str, Counter]] = {}
    for aspect, opinion, sentiment in records:
        key = aspect.strip().lower()
        if not key or sentiment not in SENTIMENT_LABELS:
            continue
        counts.setdefault(key, Counter())[sentiment] += 1
        reason = opinion.strip().lower()
        if reason:
            reason_counts.setdefault(key, {}).setdefault(sentiment, Counter())[reason] += 1

    summaries = []
    for aspect, counter in counts.items():
        total = sum(counter.values())
        if total < min_mentions:
            continue
        majority_sentiment = max(SENTIMENT_LABELS, key=lambda label: (counter[label], -SENTIMENT_LABELS.index(label)))
        aspect_reasons = reason_counts.get(aspect, {})
        summaries.append(
            AspectReasonSummary(
                aspect=aspect,
                positive=counter["positive"],
                positive_reasons=_top_reasons(aspect_reasons.get("positive", Counter()), top_n),
                negative=counter["negative"],
                negative_reasons=_top_reasons(aspect_reasons.get("negative", Counter()), top_n),
                neutral=counter["neutral"],
                neutral_reasons=_top_reasons(aspect_reasons.get("neutral", Counter()), top_n),
                total=total,
                majority_sentiment=majority_sentiment,
            )
        )

    summaries.sort(key=lambda s: (-s.total, s.aspect))
    return summaries
