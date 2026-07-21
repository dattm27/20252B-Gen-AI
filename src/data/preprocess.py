"""Turn parsed SemEval sentences into flat (context, aspect, polarity) examples."""
from __future__ import annotations

from .semeval_loader import Sentence

# "conflict" is excluded: <1% of examples, not enough signal to train/evaluate on.
VALID_POLARITIES = {"positive", "negative", "neutral"}


def mark_aspect(text: str, start: int, end: int) -> str:
    """Wrap the aspect span in $T$ markers so the vectorizer can pick up on it,
    mirroring the target-marking convention used by common ABSA baselines.
    Falls back to the untouched sentence if the offsets look invalid.
    """
    if start < 0 or end < 0 or end > len(text) or start >= end:
        return text
    return f"{text[:start]}$T$ {text[start:end]} $T${text[end:]}"


def build_examples(sentences: list[Sentence]) -> tuple[list[str], list[str], list[str]]:
    """Flatten sentences into parallel (marked_context, aspect_term, polarity) lists,
    one entry per labeled aspect term.
    """
    contexts, aspects, labels = [], [], []
    for sent in sentences:
        for term in sent.aspect_terms:
            if term.polarity not in VALID_POLARITIES:
                continue
            contexts.append(mark_aspect(sent.text, term.start, term.end))
            aspects.append(term.term)
            labels.append(term.polarity)
    return contexts, aspects, labels
