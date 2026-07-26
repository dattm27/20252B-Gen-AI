import numpy as np

from src.data.aste_loader import AsteTriplet
from src.data.gts_grid import (
    ID2TAG,
    IGNORE_INDEX,
    TAG2ID,
    SpanSentence,
    SpanTriplet,
    build_grid_tags,
    collapse_grid_from_subwords,
    decode_grid,
    expand_grid_to_subwords,
    parse_aste_line_with_spans,
    span_triplets_to_aste_triplets,
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

# The running example from Wu et al. (Findings of EMNLP 2020), Figures 1-3:
# aspects "hot dogs" (positive, opinion "top notch") and "coffee" (neutral, opinion "average" --
# note the opinion sits *before* the aspect here, a good ordering edge case).
PAPER_EXAMPLE_TOKENS = ["The", "hot", "dogs", "are", "top", "notch", "but", "average", "coffee"]
PAPER_EXAMPLE_TRIPLETS = [
    SpanTriplet(aspect_span=(1, 3), opinion_span=(4, 6), sentiment="positive"),
    SpanTriplet(aspect_span=(8, 9), opinion_span=(7, 8), sentiment="neutral"),
]


def test_parse_aste_line_with_spans_single_triplet():
    result = parse_aste_line_with_spans(ONE_TRIPLET_LINE)

    assert result.tokens == ["The", "food", "is", "good", "."]
    assert result.triplets == [SpanTriplet(aspect_span=(1, 2), opinion_span=(3, 4), sentiment="positive")]


def test_parse_aste_line_with_spans_multiword_and_multiple_triplets():
    result = parse_aste_line_with_spans(TWO_TRIPLET_LINE)

    assert result.tokens == ["Screen", "is", "nice", "but", "battery", "is", "bad", "."]
    assert result.triplets == [
        SpanTriplet(aspect_span=(0, 1), opinion_span=(2, 3), sentiment="positive"),
        SpanTriplet(aspect_span=(4, 5), opinion_span=(6, 7), sentiment="negative"),
    ]


def test_parse_aste_line_with_spans_returns_none_for_malformed_lines():
    assert parse_aste_line_with_spans("not enough #### separators") is None


def test_build_grid_tags_matches_paper_figure_example():
    sentence = SpanSentence(tokens=PAPER_EXAMPLE_TOKENS, triplets=PAPER_EXAMPLE_TRIPLETS)
    grid = build_grid_tags(sentence)
    n = len(PAPER_EXAMPLE_TOKENS)

    assert grid.shape == (n, n)
    # lower triangle is unused
    assert grid[1, 0] == IGNORE_INDEX
    # "hot dogs" (indices 1-2) is an aspect span -> A on all cells within it
    assert grid[1, 1] == TAG2ID["A"]
    assert grid[1, 2] == TAG2ID["A"]
    assert grid[2, 2] == TAG2ID["A"]
    # "top notch" (indices 4-5) is an opinion span -> O
    assert grid[4, 4] == TAG2ID["O"]
    assert grid[4, 5] == TAG2ID["O"]
    assert grid[5, 5] == TAG2ID["O"]
    # cross cells between "hot dogs" and "top notch" -> POS
    assert grid[1, 4] == TAG2ID["POS"]
    assert grid[2, 5] == TAG2ID["POS"]
    # "average" (index 7, opinion) before "coffee" (index 8, aspect) -> cross cell (7, 8) = NEU
    assert grid[7, 7] == TAG2ID["O"]
    assert grid[8, 8] == TAG2ID["A"]
    assert grid[7, 8] == TAG2ID["NEU"]
    # an unrelated cell stays N
    assert grid[0, 0] == TAG2ID["N"]


def test_decode_grid_round_trips_paper_example():
    sentence = SpanSentence(tokens=PAPER_EXAMPLE_TOKENS, triplets=PAPER_EXAMPLE_TRIPLETS)
    grid = build_grid_tags(sentence)

    decoded = decode_grid(PAPER_EXAMPLE_TOKENS, grid)

    assert sorted(decoded, key=lambda t: t.aspect_span) == sorted(
        PAPER_EXAMPLE_TRIPLETS, key=lambda t: t.aspect_span
    )


def test_decode_grid_ignores_span_pair_with_no_sentiment_cell():
    n = 4
    grid = np.full((n, n), IGNORE_INDEX, dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            grid[i, j] = TAG2ID["N"]
    grid[0, 0] = TAG2ID["A"]  # aspect span (0,1)
    grid[2, 2] = TAG2ID["O"]  # opinion span (2,3), never paired (cross cell stays N)

    decoded = decode_grid(["a", "x", "b", "y"], grid)

    assert decoded == []


def test_decode_grid_majority_votes_sentiment_across_multiword_spans():
    # 2-word aspect (0,2), 2-word opinion (2,4): 4 cross cells, 3 vote POS, 1 votes NEG -> POS wins.
    n = 4
    grid = np.full((n, n), IGNORE_INDEX, dtype=np.int64)
    for i in range(n):
        for j in range(i, n):
            grid[i, j] = TAG2ID["N"]
    grid[0, 0] = grid[0, 1] = grid[1, 1] = TAG2ID["A"]
    grid[2, 2] = grid[2, 3] = grid[3, 3] = TAG2ID["O"]
    grid[0, 2] = TAG2ID["POS"]
    grid[0, 3] = TAG2ID["POS"]
    grid[1, 2] = TAG2ID["POS"]
    grid[1, 3] = TAG2ID["NEG"]

    decoded = decode_grid(["a", "b", "c", "d"], grid)

    assert decoded == [SpanTriplet(aspect_span=(0, 2), opinion_span=(2, 4), sentiment="positive")]


def test_span_triplets_to_aste_triplets_joins_phrases():
    result = span_triplets_to_aste_triplets(PAPER_EXAMPLE_TOKENS, PAPER_EXAMPLE_TRIPLETS)

    assert result == [
        AsteTriplet(aspect="hot dogs", opinion="top notch", sentiment="positive"),
        AsteTriplet(aspect="coffee", opinion="average", sentiment="neutral"),
    ]


def test_tag_id_round_trip_covers_all_six_tags():
    assert {ID2TAG[TAG2ID[tag]] for tag in TAG2ID} == set(TAG2ID)
    assert len(TAG2ID) == 6


def test_expand_and_collapse_grid_round_trips_through_subword_split():
    # 2 words: "food" (word 0, one subword) and "unbelievable" (word 1, split into 2 subwords by
    # BERT's WordPiece) -> tokens [CLS] food un ##believable [SEP], word_ids = [None,0,1,1,None]
    word_grid = np.array([[TAG2ID["A"], TAG2ID["POS"]], [IGNORE_INDEX, TAG2ID["O"]]], dtype=np.int64)
    word_ids = [None, 0, 1, 1, None]

    subword_grid = expand_grid_to_subwords(word_grid, word_ids)
    assert subword_grid.shape == (5, 5)
    assert subword_grid[0, 0] == IGNORE_INDEX  # [CLS] row, untouched
    assert subword_grid[1, 1] == TAG2ID["A"]  # "food" x "food"
    assert subword_grid[1, 2] == TAG2ID["POS"]  # "food" x "un"
    assert subword_grid[1, 3] == TAG2ID["POS"]  # "food" x "##believable"
    assert subword_grid[2, 3] == TAG2ID["O"]  # "un" x "##believable"

    collapsed = collapse_grid_from_subwords(subword_grid, word_ids)
    assert collapsed.shape == (2, 2)
    np.testing.assert_array_equal(collapsed, np.array([[TAG2ID["A"], TAG2ID["POS"]], [TAG2ID["N"], TAG2ID["O"]]]))
