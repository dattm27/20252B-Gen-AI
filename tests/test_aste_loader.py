from src.data.aste_loader import (
    AsteTriplet,
    corpus_triplet_prf,
    parse_aste_line,
    text_to_triplets,
    triplets_to_text,
)

ONE_TRIPLET_LINE = (
    "The food is good . #### "
    "The=O food=1-POS is=O good=O .=O #### "
    "The=O food=O is=O good=S .=O"
)

TWO_TRIPLET_LINE = (
    "Screen is nice but battery is bad . #### "
    "Screen=1-POS is=O nice=O but=O battery=11-NEG is=O bad=O .=O #### "
    "Screen=O is=O nice=S but=O battery=O is=O bad=SS .=O"
)


def test_parse_aste_line_extracts_a_single_triplet():
    result = parse_aste_line(ONE_TRIPLET_LINE)

    assert result.text == "The food is good ."
    assert result.triplets == [AsteTriplet(aspect="food", opinion="good", sentiment="positive")]


def test_parse_aste_line_extracts_multiple_triplets_in_group_order():
    result = parse_aste_line(TWO_TRIPLET_LINE)

    assert result.triplets == [
        AsteTriplet(aspect="Screen", opinion="nice", sentiment="positive"),
        AsteTriplet(aspect="battery", opinion="bad", sentiment="negative"),
    ]


def test_parse_aste_line_returns_none_for_malformed_lines():
    assert parse_aste_line("not enough #### separators") is None
    assert parse_aste_line("too #### many #### #### separators") is None


def test_parse_aste_line_handles_no_triplets():
    line = "It works . #### It=O works=O .=O #### It=O works=O .=O"
    result = parse_aste_line(line)

    assert result.text == "It works ."
    assert result.triplets == []


def test_triplets_to_text_formats_and_handles_empty():
    triplets = [
        AsteTriplet(aspect="food", opinion="good", sentiment="positive"),
        AsteTriplet(aspect="service", opinion="slow", sentiment="negative"),
    ]
    assert triplets_to_text(triplets) == (
        "aspect: food | opinion: good | sentiment: positive ; "
        "aspect: service | opinion: slow | sentiment: negative"
    )
    assert triplets_to_text([]) == "no triplet"


def test_text_to_triplets_parses_generated_string_and_is_inverse_of_triplets_to_text():
    triplets = [
        AsteTriplet(aspect="food", opinion="good", sentiment="positive"),
        AsteTriplet(aspect="service", opinion="slow", sentiment="negative"),
    ]
    assert text_to_triplets(triplets_to_text(triplets)) == triplets


def test_text_to_triplets_handles_no_triplet_and_malformed_text():
    assert text_to_triplets("no triplet") == []
    assert text_to_triplets("garbage model output") == []


def test_corpus_triplet_prf_perfect_match():
    gold = [[AsteTriplet(aspect="food", opinion="good", sentiment="positive")]]
    assert corpus_triplet_prf(gold, gold) == (1.0, 1.0, 1.0)


def test_corpus_triplet_prf_partial_overlap_is_micro_averaged():
    # sentence 1: 1 correct out of 1 predicted, 1 gold; sentence 2: 0 correct out of 1 predicted, 2 gold
    preds = [
        [AsteTriplet(aspect="food", opinion="good", sentiment="positive")],
        [AsteTriplet(aspect="staff", opinion="rude", sentiment="negative")],
    ]
    golds = [
        [AsteTriplet(aspect="food", opinion="good", sentiment="positive")],
        [
            AsteTriplet(aspect="service", opinion="slow", sentiment="negative"),
            AsteTriplet(aspect="price", opinion="high", sentiment="negative"),
        ],
    ]
    # tp=1, pred_total=2, gold_total=3
    precision, recall, f1 = corpus_triplet_prf(preds, golds)
    assert precision == 0.5
    assert recall == 1 / 3
    assert f1 == 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3)


def test_corpus_triplet_prf_normalizes_case_and_whitespace():
    preds = [[AsteTriplet(aspect=" Food ", opinion="GOOD", sentiment="positive")]]
    golds = [[AsteTriplet(aspect="food", opinion="good", sentiment="positive")]]
    assert corpus_triplet_prf(preds, golds) == (1.0, 1.0, 1.0)


def test_corpus_triplet_prf_handles_no_predictions_or_no_gold():
    empty = [[]]
    non_empty = [[AsteTriplet(aspect="food", opinion="good", sentiment="positive")]]
    assert corpus_triplet_prf(empty, non_empty) == (0.0, 0.0, 0.0)
    assert corpus_triplet_prf(non_empty, empty) == (0.0, 0.0, 0.0)
    assert corpus_triplet_prf(empty, empty) == (0.0, 0.0, 0.0)
