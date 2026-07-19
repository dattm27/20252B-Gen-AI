"""TF-IDF + Logistic Regression baseline for aspect-term polarity classification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score


@dataclass
class BaselineResult:
    accuracy: float
    macro_f1: float
    report: str


class TfidfLogRegBaseline:
    """TF-IDF(sentence with $T$-marked aspect) + TF-IDF(aspect term) -> Logistic Regression."""

    def __init__(self, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)):
        self.context_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.aspect_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def _vectorize(self, contexts: list[str], aspects: list[str], fit: bool):
        if fit:
            x_ctx = self.context_vectorizer.fit_transform(contexts)
            x_asp = self.aspect_vectorizer.fit_transform(aspects)
        else:
            x_ctx = self.context_vectorizer.transform(contexts)
            x_asp = self.aspect_vectorizer.transform(aspects)
        return hstack([x_ctx, x_asp])

    def fit(self, contexts: list[str], aspects: list[str], labels: list[str]) -> "TfidfLogRegBaseline":
        x = self._vectorize(contexts, aspects, fit=True)
        self.clf.fit(x, labels)
        return self

    def predict(self, contexts: list[str], aspects: list[str]) -> np.ndarray:
        x = self._vectorize(contexts, aspects, fit=False)
        return self.clf.predict(x)

    def evaluate(self, contexts: list[str], aspects: list[str], labels: list[str]) -> BaselineResult:
        preds = self.predict(contexts, aspects)
        acc = float(np.mean(preds == np.array(labels)))
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        report = classification_report(labels, preds, zero_division=0)
        return BaselineResult(accuracy=acc, macro_f1=macro_f1, report=report)
