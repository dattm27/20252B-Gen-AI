from pathlib import Path

from src.baseline.tfidf_logreg import TfidfLogRegBaseline
from src.data.preprocess import build_examples, mark_aspect
from src.data.semeval_loader import load_semeval_xml

FIXTURE = Path(__file__).parent / "fixtures" / "sample_laptop.xml"


def test_load_semeval_xml_parses_sentences_and_aspect_terms():
    sentences = load_semeval_xml(FIXTURE)

    assert len(sentences) == 6
    first = sentences[0]
    assert first.sentence_id == "1"
    assert first.text.startswith("The battery life")
    assert [t.term for t in first.aspect_terms] == ["battery life", "screen"]
    assert [t.polarity for t in first.aspect_terms] == ["positive", "negative"]


def test_mark_aspect_wraps_the_correct_span():
    text = "The battery life is amazing."
    marked = mark_aspect(text, 4, 16)
    assert marked == "The $T$ battery life $T$ is amazing."


def test_mark_aspect_falls_back_on_invalid_offsets():
    text = "short text"
    assert mark_aspect(text, -1, 5) == text
    assert mark_aspect(text, 0, 999) == text


def test_build_examples_flattens_one_row_per_aspect_term():
    sentences = load_semeval_xml(FIXTURE)
    contexts, aspects, labels = build_examples(sentences)

    expected_count = sum(len(s.aspect_terms) for s in sentences)
    assert len(contexts) == len(aspects) == len(labels) == expected_count
    assert "$T$" in contexts[0]
    assert set(labels) <= {"positive", "negative", "neutral", "conflict"}


def test_baseline_trains_and_predicts_end_to_end():
    sentences = load_semeval_xml(FIXTURE)
    contexts, aspects, labels = build_examples(sentences)

    model = TfidfLogRegBaseline().fit(contexts, aspects, labels)
    preds = model.predict(contexts, aspects)

    assert len(preds) == len(labels)

    result = model.evaluate(contexts, aspects, labels)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
