"""Grid Tagging Scheme (GTS) representation for ASTE, following Wu et al. (Findings of EMNLP
2020), "Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction"
(https://aclanthology.org/2020.findings-emnlp.234/).

For a sentence of n tokens, GTS tags every word-pair (w_i, w_j) with i <= j (an upper-triangular
grid, since the relation is unordered/symmetric) with one of six tags:

    N   - no relation between w_i and w_j
    A   - w_i and w_j both belong to the same aspect span (includes i == j, for single-word spans)
    O   - w_i and w_j both belong to the same opinion span
    POS/NEU/NEG - w_i belongs to an aspect span, w_j belongs to an opinion span (or vice versa),
                  and the two spans form an opinion pair with that sentiment

This module builds a gold grid from `AsteSentence`-style span data (`build_grid_tags`), and
recovers (aspect, opinion, sentiment) triplets from a predicted grid (`decode_grid`), following
the paper's decoding strategy: aspect/opinion spans are read off maximal same-tag runs on the
diagonal, then every (aspect span, opinion span) pair is checked for a majority Pos/Neu/Neg tag
among the cross cells connecting them.

This is an independent reimplementation grounded in the paper's description (Section 2.2-2.3),
not the authors' original code -- the encoder representation r_ij (Section 3.1) in particular is
a plain concat+MLP here (see `src/baseline/gts_bert_model.py`), since the paper does not fully
specify its attention-layer formula.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

TAGS = ["N", "A", "O", "POS", "NEU", "NEG"]
TAG2ID = {tag: idx for idx, tag in enumerate(TAGS)}
ID2TAG = {idx: tag for tag, idx in TAG2ID.items()}
IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss default ignore_index, for the unused lower triangle

_SENTIMENT_TO_TAG = {"positive": "POS", "neutral": "NEU", "negative": "NEG"}
_TAG_TO_SENTIMENT = {tag: sentiment for sentiment, tag in _SENTIMENT_TO_TAG.items()}


@dataclass
class SpanTriplet:
    """A gold triplet with token-index spans (end-exclusive), used to build/check grids."""

    aspect_span: tuple[int, int]
    opinion_span: tuple[int, int]
    sentiment: str  # "positive" | "neutral" | "negative"


@dataclass
class SpanSentence:
    tokens: list[str]
    triplets: list[SpanTriplet] = field(default_factory=list)


def _split_token_tag(item: str) -> tuple[str, str]:
    token, tag = item.rsplit("=", 1)
    return token, tag


def parse_aste_line_with_spans(line: str) -> SpanSentence | None:
    """Same `sentence #### aspect tags #### opinion tags` format as
    `src.data.aste_loader.parse_aste_line`, but keeps token-index spans instead of collapsing
    each group to a joined string -- needed to build a GTS grid. Returns None for malformed
    lines (same rule as `parse_aste_line`: exactly 3 `####`-separated parts)."""
    parts = line.strip().split("####")
    if len(parts) != 3:
        return None

    sentence, target_tag_text, opinion_tag_text = parts
    tokens = sentence.strip().split()
    target_pairs = [_split_token_tag(item) for item in target_tag_text.strip().split()]
    opinion_pairs = [_split_token_tag(item) for item in opinion_tag_text.strip().split()]

    target_groups: dict[str, dict] = {}
    for idx, (_, tag) in enumerate(target_pairs):
        if tag == "O" or "-" not in tag:
            continue
        group_id, sentiment_code = tag.split("-", 1)
        info = target_groups.setdefault(group_id, {"indices": [], "sentiment": sentiment_code})
        info["indices"].append(idx)

    opinion_groups: dict[str, list[int]] = {}
    for idx, (_, tag) in enumerate(opinion_pairs):
        if tag == "O":
            continue
        opinion_groups.setdefault(tag, []).append(idx)

    sentiment_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
    triplets = []
    for group_id, target_info in sorted(target_groups.items(), key=lambda x: (len(x[0]), x[0])):
        opinion_group_id = "S" * len(group_id)
        aspect_idx = target_info["indices"]
        opinion_idx = opinion_groups.get(opinion_group_id, [])
        if not aspect_idx or not opinion_idx:
            continue
        sentiment = sentiment_map.get(target_info["sentiment"], target_info["sentiment"].lower())
        triplets.append(
            SpanTriplet(
                aspect_span=(min(aspect_idx), max(aspect_idx) + 1),
                opinion_span=(min(opinion_idx), max(opinion_idx) + 1),
                sentiment=sentiment,
            )
        )
    return SpanSentence(tokens=tokens, triplets=triplets)


def build_grid_tags(sentence: SpanSentence) -> np.ndarray:
    """Build the gold n x n grid (upper triangle valid, lower triangle = IGNORE_INDEX)."""
    n = len(sentence.tokens)
    grid = np.full((n, n), IGNORE_INDEX, dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            grid[i, j] = TAG2ID["N"]

    def mark_span(span: tuple[int, int], tag: str) -> None:
        start, end = span
        for i in range(start, end):
            for j in range(start, end):
                if i <= j:
                    grid[i, j] = TAG2ID[tag]

    for triplet in sentence.triplets:
        mark_span(triplet.aspect_span, "A")
        mark_span(triplet.opinion_span, "O")

    for triplet in sentence.triplets:
        tag = _SENTIMENT_TO_TAG[triplet.sentiment]
        a_start, a_end = triplet.aspect_span
        o_start, o_end = triplet.opinion_span
        for ai in range(a_start, a_end):
            for oi in range(o_start, o_end):
                i, j = (ai, oi) if ai <= oi else (oi, ai)
                grid[i, j] = TAG2ID[tag]

    return grid


def _diagonal_spans(grid: np.ndarray, tag: str) -> list[tuple[int, int]]:
    """Maximal runs of consecutive diagonal cells tagged `tag` -> list of (start, end) spans."""
    n = grid.shape[0]
    tag_id = TAG2ID[tag]
    spans = []
    i = 0
    while i < n:
        if grid[i, i] == tag_id:
            j = i
            while j + 1 < n and grid[j + 1, j + 1] == tag_id:
                j += 1
            spans.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return spans


def decode_grid(tokens: list[str], grid: np.ndarray) -> list[SpanTriplet]:
    """Recover triplets from a predicted grid: read aspect/opinion spans off the diagonal, then
    for every (aspect span, opinion span) pair, majority-vote the Pos/Neu/Neg tags among the
    cells connecting them (paper Section 2.3). A pair with no Pos/Neu/Neg cell is dropped."""
    aspect_spans = _diagonal_spans(grid, "A")
    opinion_spans = _diagonal_spans(grid, "O")

    sentiment_tag_ids = {TAG2ID[t] for t in ("POS", "NEU", "NEG")}
    triplets = []
    for a_start, a_end in aspect_spans:
        for o_start, o_end in opinion_spans:
            votes: Counter = Counter()
            for ai in range(a_start, a_end):
                for oi in range(o_start, o_end):
                    i, j = (ai, oi) if ai <= oi else (oi, ai)
                    cell = int(grid[i, j])
                    if cell in sentiment_tag_ids:
                        votes[cell] += 1
            if not votes:
                continue
            best_tag_id, _ = max(votes.items(), key=lambda kv: (kv[1], -kv[0]))
            triplets.append(
                SpanTriplet(
                    aspect_span=(a_start, a_end),
                    opinion_span=(o_start, o_end),
                    sentiment=_TAG_TO_SENTIMENT[ID2TAG[best_tag_id]],
                )
            )
    return triplets


def expand_grid_to_subwords(word_grid: np.ndarray, word_ids: list[int | None]) -> np.ndarray:
    """Map a word-level grid to subword-level positions, for BERT tokenizers that split words
    into multiple subwords. `word_ids` is what HuggingFace's `BatchEncoding.word_ids()` returns:
    one entry per subword token, holding the source word index (or `None` for special tokens
    like [CLS]/[SEP]/padding). Every subword of word i gets the tags of word i; cells touching a
    `None` position are IGNORE_INDEX (not used in the loss)."""
    m = len(word_ids)
    grid = np.full((m, m), IGNORE_INDEX, dtype=np.int64)
    for p in range(m):
        wi = word_ids[p]
        if wi is None:
            continue
        for q in range(p, m):
            wj = word_ids[q]
            if wj is None:
                continue
            i, j = (wi, wj) if wi <= wj else (wj, wi)
            grid[p, q] = word_grid[i, j]
    return grid


def collapse_grid_from_subwords(subword_grid: np.ndarray, word_ids: list[int | None]) -> np.ndarray:
    """Inverse of `expand_grid_to_subwords`: recover a word-level grid from subword-level
    predictions by reading, for each word, its *first* subword's predictions (the standard
    convention for BERT token classification -- only the first subword of a word is used)."""
    first_subword: dict[int, int] = {}
    for p, wi in enumerate(word_ids):
        if wi is not None and wi not in first_subword:
            first_subword[wi] = p
    num_words = len(first_subword)
    grid = np.full((num_words, num_words), TAG2ID["N"], dtype=np.int64)
    for i in range(num_words):
        for j in range(i, num_words):
            grid[i, j] = subword_grid[first_subword[i], first_subword[j]]
    return grid


def span_triplets_to_aste_triplets(tokens: list[str], span_triplets: list[SpanTriplet]):
    """Convert `SpanTriplet` (token-index spans) into `src.data.aste_loader.AsteTriplet`
    (joined phrase strings), so grid-decoded predictions can be scored with the same
    `corpus_triplet_prf` used for the T5 models and `AsteLookupBaseline`."""
    from src.data.aste_loader import AsteTriplet

    def phrase(span: tuple[int, int]) -> str:
        return " ".join(tokens[span[0] : span[1]])

    return [
        AsteTriplet(aspect=phrase(t.aspect_span), opinion=phrase(t.opinion_span), sentiment=t.sentiment)
        for t in span_triplets
    ]
