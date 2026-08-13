"""Parser for the ASTE triplet tag format used by the SemEval Triplet data
(github.com/xuuuluuu/SemEval-Triplet-data), e.g.:

    The food is good . #### food=T-POS is=O good=O #### food=O is=O good=S

Line format: `sentence #### aspect tags #### opinion tags`. Aspect tokens carry a
`<group>-<POS|NEG|NEU>` tag; opinion tokens carry a matching `S`-repeated group id (aspect
group `1` pairs with opinion group `S`, group `12` pairs with `SS`, etc).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SENTIMENT_MAP = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

_TRIPLET_RE = re.compile(
    r"aspect:\s*(.*?)\s*\|\s*opinion:\s*(.*?)\s*\|\s*sentiment:\s*(positive|negative|neutral)",
    re.IGNORECASE,
)


@dataclass
class AsteTriplet:
    aspect: str
    opinion: str
    sentiment: str


@dataclass
class AsteSentence:
    text: str
    triplets: list[AsteTriplet] = field(default_factory=list)


def _split_token_tag(item: str) -> tuple[str, str]:
    token, tag = item.rsplit("=", 1)
    return token, tag


def _parse_tag_sequence(tag_text: str) -> list[tuple[str, str]]:
    return [_split_token_tag(item) for item in tag_text.strip().split()]


def _phrase_from_tokens(tokens: list[str]) -> str:
    return " ".join(tokens).replace(" n't", "n't").replace(" 's", "'s").strip()


def parse_aste_line(line: str) -> AsteSentence | None:
    """Parse one `sentence #### aspect tags #### opinion tags` line. Returns None if the
    line doesn't have exactly 3 `####`-separated parts (e.g. blank/malformed lines)."""
    parts = line.strip().split("####")
    if len(parts) != 3:
        return None

    sentence, target_tag_text, opinion_tag_text = parts
    target_pairs = _parse_tag_sequence(target_tag_text)
    opinion_pairs = _parse_tag_sequence(opinion_tag_text)

    target_groups: dict[str, dict] = {}
    for token, tag in target_pairs:
        if tag == "O" or "-" not in tag:
            continue
        group_id, sentiment_code = tag.split("-", 1)
        target_groups.setdefault(group_id, {"tokens": [], "sentiment": sentiment_code})
        target_groups[group_id]["tokens"].append(token)

    opinion_groups: dict[str, list[str]] = {}
    for token, tag in opinion_pairs:
        if tag == "O":
            continue
        opinion_groups.setdefault(tag, []).append(token)

    triplets = []
    for group_id, target_info in sorted(target_groups.items(), key=lambda x: (len(x[0]), x[0])):
        opinion_group_id = "S" * len(group_id)
        aspect = _phrase_from_tokens(target_info["tokens"])
        opinion = _phrase_from_tokens(opinion_groups.get(opinion_group_id, []))
        sentiment = SENTIMENT_MAP.get(target_info["sentiment"], target_info["sentiment"].lower())
        if aspect and opinion:
            triplets.append(AsteTriplet(aspect=aspect, opinion=opinion, sentiment=sentiment))

    return AsteSentence(text=sentence.strip(), triplets=triplets)


def triplets_to_text(triplets: list[AsteTriplet]) -> str:
    """Render triplets as `aspect: X | opinion: Y | sentiment: Z ; ...` — the T5 target/prompt
    format used to fine-tune the ASTE models (see notebooks/train-*-for-aste-*.ipynb)."""
    if not triplets:
        return "no triplet"
    chunks = [f"aspect: {t.aspect} | opinion: {t.opinion} | sentiment: {t.sentiment}" for t in triplets]
    return " ; ".join(chunks)


def text_to_triplets(text: str) -> list[AsteTriplet]:
    """Inverse of `triplets_to_text`: parse a T5-generated `aspect: X | opinion: Y |
    sentiment: Z ; ...` string back into triplets (same regex the ASTE training notebooks use
    at inference time, e.g. `notebooks/train-t5-base-for-aste-on-14res-15res-16res.ipynb`).
    Malformed/empty generations simply yield no triplets."""
    return [
        AsteTriplet(aspect=aspect.strip(), opinion=opinion.strip(), sentiment=sentiment.lower())
        for aspect, opinion, sentiment in _TRIPLET_RE.findall(text)
    ]


def load_aste_file(path: str | Path) -> list[AsteSentence]:
    """Load a `train.txt`/`dev.txt`/`test.txt` ASTE triplet file. Blank/malformed lines are skipped."""
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = parse_aste_line(line)
            if item is not None:
                sentences.append(item)
    return sentences


def _normalize_triplet_set(triplets: list[AsteTriplet]) -> set[tuple[str, str, str]]:
    return {(t.aspect.strip().lower(), t.opinion.strip().lower(), t.sentiment) for t in triplets}


def corpus_triplet_prf(
    pred_triplets_per_sentence: list[list[AsteTriplet]],
    gold_triplets_per_sentence: list[list[AsteTriplet]],
) -> tuple[float, float, float]:
    """Micro-averaged precision/recall/F1 via set overlap of normalized (aspect, opinion,
    sentiment) triples, aggregated across the whole corpus (not averaged per-sentence) — the
    same triplet-level metric used to evaluate the T5 ASTE models (see the training notebooks'
    `triplet_prf`), so scores are directly comparable across baselines and models.
    """
    tp = pred_total = gold_total = 0
    for preds, golds in zip(pred_triplets_per_sentence, gold_triplets_per_sentence):
        pred_set = _normalize_triplet_set(preds)
        gold_set = _normalize_triplet_set(golds)
        tp += len(pred_set & gold_set)
        pred_total += len(pred_set)
        gold_total += len(gold_set)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
