from src.data.aste_loader import AsteTriplet, parse_aste_line, triplets_to_text

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
