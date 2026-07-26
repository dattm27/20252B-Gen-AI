import json

import pytest

from src.report.factual_checker import check_reasoned_report, check_report
from src.report.flan_t5_report import build_prompt, build_reasoned_prompt, select_report_rows
from src.report.stats_io import load_aspect_stats


@pytest.fixture
def rows():
    return load_aspect_stats("output/aspect_stats.txt", table="predicted")


def test_real_predicted_stats_are_valid_and_prompt_is_bounded(rows):
    prompt = build_prompt(rows, max_aspects=2)
    assert "aspect_start screen aspect_end | total=60" in prompt
    assert "aspect_start price aspect_end | total=56" in prompt
    assert "Required aspect order: screen | price" in prompt
    assert "use |" not in prompt
    assert "numbers are not prices, hours, ratings" in prompt


def test_selection_covers_popular_strength_weakness_and_divided_aspects(rows):
    selected = select_report_rows(rows, max_aspects=4)
    assert [row["aspect"] for row in selected] == ["screen", "price", "battery", "keyboard"]


def test_checker_accepts_source_grounded_claims(rows):
    report = (
        "screen: 60 mentions (32 positive, 24 negative, 4 neutral). "
        "battery: 47 mentions (9 positive, 34 negative, 4 neutral)."
    )
    result = check_report(report, rows)
    assert result["passed"] is True
    assert result["claims_checked"] == 2


def test_checker_accepts_natural_paragraph(rows):
    report = (
        "Screen received 60 mentions, including 32 positive, 24 negative, and 4 neutral. "
        "Finally, battery received 47 mentions, with 9 positive, 34 negative, and 4 neutral."
    )
    result = check_report(report, rows, required_aspects=["screen", "battery"])
    assert result["passed"] is True
    assert result["claims_checked"] == 2
    assert result["unexpected_text"] == ""


def test_checker_accepts_analytical_paragraph_and_roles(rows):
    report = (
        "Screen attracted the most attention, with 60 mentions: 32 positive, 24 negative, and 4 neutral. "
        "Price emerged as the clearest strength, receiving 49 positive mentions out of 56, alongside 4 negative and 3 neutral. "
        "In contrast, battery was the main concern, with 34 negative mentions out of 47, compared with 9 positive and 4 neutral. "
        "Keyboard feedback was the most divided, with 25 positive, 20 negative, and 5 neutral mentions among 50 total."
    )
    aspects = ["screen", "price", "battery", "keyboard"]
    roles = ["most_discussed", "strongest_positive", "strongest_negative", "most_divided"]
    result = check_report(report, rows, required_aspects=aspects, required_roles=roles)
    assert result["passed"] is True
    assert result["claims_checked"] == 4
    assert result["missing_roles"] == []
    assert result["role_errors"] == []


def test_checker_reports_hallucinated_number(rows):
    report = "screen: 60 mentions (40 positive, 24 negative, 4 neutral)."
    result = check_report(report, rows)
    assert result["passed"] is False
    assert "positive: claimed 40, expected 32" in result["checks"][0]["errors"]


def test_checker_requires_every_selected_aspect(rows):
    report = "screen: 60 mentions (32 positive, 24 negative, 4 neutral)."
    result = check_report(report, rows, required_aspects=["screen", "battery"])
    assert result["passed"] is False
    assert result["missing_aspects"] == ["battery"]


def test_checker_rejects_extra_narrative_even_without_numbers(rows):
    report = (
        "screen: 60 mentions (32 positive, 24 negative, 4 neutral). "
        "Apple makes the best laptop."
    )
    result = check_report(report, rows)
    assert result["passed"] is False
    assert result["unexpected_text"] == "Apple makes the best laptop"


def test_checker_rejects_unknown_aspect_and_unstructured_numbers(rows):
    result = check_report(
        "camera: 10 mentions (8 positive, 1 negative, 1 neutral). Overall score is 95%.",
        rows,
    )
    assert result["passed"] is False
    assert result["unparsed_numeric_tokens"] == 1
    assert result["checks"][0]["expected"] is None


def test_number_inside_aspect_name_is_not_treated_as_unparsed():
    rows = [{"aspect": "windows 7", "positive": 2, "negative": 1, "neutral": 0, "total": 3}]
    result = check_report("windows 7: 3 mentions (2 positive, 1 negative, 0 neutral).", rows)
    assert result["passed"] is True
    assert result["unparsed_numeric_tokens"] == 0


def test_loader_rejects_inconsistent_total(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"aspect": "x", "positive": 1, "negative": 1, "neutral": 0, "total": 3}]))
    with pytest.raises(ValueError, match="do not add up"):
        load_aspect_stats(path)


def test_reason_table_is_loaded_and_prompt_contains_evidence():
    rows = load_aspect_stats("output/aspect_reasons_restaurant.json", table="predicted")
    food = rows[0]
    assert food["aspect"] == "food"
    assert food["positive_reasons"][0] == ["great", 109]
    prompt = build_reasoned_prompt(rows)
    assert "positive_reasons=great (109), good (100)" in prompt
    assert "role=" not in prompt
    assert "Avoid a row-by-row list" in prompt


def test_reason_aware_selection_uses_distinct_comparative_aspects():
    rows = load_aspect_stats("output/aspect_reasons_restaurant.json", table="predicted")
    selected = select_report_rows(rows)
    assert [row["aspect"] for row in selected] == ["food", "indian food", "dessert", "waiter"]


def test_reasoned_checker_accepts_flexible_grounded_paragraph():
    rows = [
        {
            "aspect": "food",
            "total": 100,
            "positive": 75,
            "negative": 20,
            "neutral": 5,
            "positive_reasons": [["delicious", 30]],
            "negative_reasons": [["bland", 8]],
            "neutral_reasons": [],
        },
        {
            "aspect": "service",
            "total": 80,
            "positive": 20,
            "negative": 55,
            "neutral": 5,
            "positive_reasons": [["friendly", 10]],
            "negative_reasons": [["slow", 25]],
            "neutral_reasons": [],
        },
    ]
    report = (
        "Food led the discussion with 100 mentions, including 75 positive responses, "
        "as diners repeatedly called it delicious. "
        "Service needs attention: 55 of its 80 mentions were negative, most often because it was slow."
    )
    result = check_reasoned_report(report, rows)
    assert result["passed"] is True
    assert result["valid_claims"] == 2


def test_reasoned_checker_rejects_invented_number_and_missing_reason():
    rows = [
        {
            "aspect": "service",
            "total": 80,
            "positive": 20,
            "negative": 55,
            "neutral": 5,
            "positive_reasons": [["friendly", 10]],
            "negative_reasons": [["slow", 25]],
            "neutral_reasons": [],
        }
    ]
    result = check_reasoned_report(
        "Service received 80 mentions, including 99 negative responses.",
        rows,
    )
    assert result["passed"] is False
    assert "unsupported numbers: [99]" in result["checks"][0]["errors"]
    assert "no grounded reason mentioned" in result["checks"][0]["errors"]


def test_reasoned_checker_separates_factuality_from_coverage():
    rows = [
        {
            "aspect": "food",
            "total": 100,
            "positive": 75,
            "negative": 20,
            "neutral": 5,
            "positive_reasons": [["delicious", 30]],
            "negative_reasons": [["bland", 8]],
            "neutral_reasons": [],
        },
        {
            "aspect": "service",
            "total": 80,
            "positive": 20,
            "negative": 55,
            "neutral": 5,
            "positive_reasons": [["friendly", 10]],
            "negative_reasons": [["slow", 25]],
            "neutral_reasons": [],
        },
    ]
    report = "Food received 100 mentions, including 75 positive responses because it was delicious."
    relaxed = check_reasoned_report(report, rows)
    strict = check_reasoned_report(report, rows, mode="strict")
    assert relaxed["factual_passed"] is True
    assert relaxed["coverage_passed"] is False
    assert relaxed["passed"] is True
    assert relaxed["missing_aspects"] == ["service"]
    assert relaxed["claims_checked"] == 1
    assert strict["factual_passed"] is True
    assert strict["coverage_passed"] is False
    assert strict["passed"] is False


def test_reasoned_checker_handles_two_aspects_in_contrast_sentence():
    rows = [
        {
            "aspect": "price",
            "total": 22,
            "positive": 21,
            "negative": 1,
            "neutral": 0,
            "positive_reasons": [["reasonable", 8]],
            "negative_reasons": [],
            "neutral_reasons": [],
        },
        {
            "aspect": "service",
            "total": 26,
            "positive": 9,
            "negative": 16,
            "neutral": 1,
            "positive_reasons": [],
            "negative_reasons": [["slow", 3]],
            "neutral_reasons": [],
        },
    ]
    report = (
        "Price was praised in 21 of 22 mentions for being reasonable, "
        "whereas service received 16 negative comments out of 26, largely citing slow."
    )
    assert check_reasoned_report(report, rows)["passed"] is True
