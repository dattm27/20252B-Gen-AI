"""A simple frequency-lookup baseline for Aspect Sentiment Triplet Extraction (ASTE) — the
"dumb" non-neural reference point for the Restaurant/T5 track, playing the same role
`TfidfLogRegBaseline` plays for the Laptop/BERT track's aspect-sentiment classification (that
baseline can't do this task at all: it requires a known aspect term up front, but ASTE has to
find the aspects too).

No training/gradient descent involved: it just memorizes which aspect and opinion phrases
appeared in the train split and which sentiment each opinion phrase most often carried, then at
inference time looks for those exact phrases in a new sentence and pairs each aspect found with
its nearest opinion match. It's expected to have low recall (fails on any aspect/opinion phrasing
not seen verbatim in training) — that's the point: a floor to show how much a real model improves
on, not a competitive system.
"""
from __future__ import annotations

from collections import Counter

from ..data.aste_loader import AsteSentence, AsteTriplet

DEFAULT_MAX_NGRAM = 3


class AsteLookupBaseline:
    def __init__(self, max_ngram: int = DEFAULT_MAX_NGRAM):
        self.max_ngram = max_ngram
        self.aspect_vocab: set[str] = set()
        self.opinion_sentiment: dict[str, str] = {}

    def fit(self, sentences: list[AsteSentence]) -> "AsteLookupBaseline":
        opinion_counts: dict[str, Counter] = {}
        for sent in sentences:
            for t in sent.triplets:
                self.aspect_vocab.add(t.aspect.strip().lower())
                key = t.opinion.strip().lower()
                opinion_counts.setdefault(key, Counter())[t.sentiment] += 1
        self.opinion_sentiment = {
            phrase: counter.most_common(1)[0][0] for phrase, counter in opinion_counts.items()
        }
        return self

    def _find_spans(self, tokens: list[str], vocab) -> list[tuple[int, int, str]]:
        """Greedy longest-match, left-to-right, non-overlapping n-gram spans matching `vocab`
        (membership-checked via `in`, so works for both a set and a dict's keys)."""
        spans = []
        i = 0
        while i < len(tokens):
            matched = None
            for n in range(min(self.max_ngram, len(tokens) - i), 0, -1):
                phrase = " ".join(tokens[i : i + n]).lower()
                if phrase in vocab:
                    matched = (i, i + n, phrase)
                    break
            if matched:
                spans.append(matched)
                i = matched[1]
            else:
                i += 1
        return spans

    def predict(self, text: str) -> list[AsteTriplet]:
        tokens = text.split()
        aspect_spans = self._find_spans(tokens, self.aspect_vocab)
        opinion_spans = self._find_spans(tokens, self.opinion_sentiment)
        if not opinion_spans:
            return []

        triplets = []
        for a_start, a_end, aspect_phrase in aspect_spans:
            a_center = (a_start + a_end) / 2
            _, _, opinion_phrase = min(
                opinion_spans, key=lambda span: abs(((span[0] + span[1]) / 2) - a_center)
            )
            sentiment = self.opinion_sentiment[opinion_phrase]
            triplets.append(AsteTriplet(aspect=aspect_phrase, opinion=opinion_phrase, sentiment=sentiment))
        return triplets
