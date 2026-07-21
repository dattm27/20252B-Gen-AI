from src.report.aspect_stats import aggregate_aspect_sentiment


def test_groups_case_and_whitespace_insensitively():
    records = [
        ("battery life", "positive"),
        ("Battery Life", "positive"),
        (" battery life ", "negative"),
        ("screen", "neutral"),
    ]

    summaries = aggregate_aspect_sentiment(records)

    battery = next(s for s in summaries if s.aspect == "battery life")
    assert battery.positive == 2
    assert battery.negative == 1
    assert battery.neutral == 0
    assert battery.total == 3
    assert battery.majority_sentiment == "positive"


def test_sorted_by_total_descending_then_alphabetically():
    records = [
        ("screen", "positive"),
        ("keyboard", "positive"),
        ("keyboard", "negative"),
        ("battery", "positive"),
        ("battery", "negative"),
    ]

    summaries = aggregate_aspect_sentiment(records)

    assert [s.aspect for s in summaries] == ["battery", "keyboard", "screen"]


def test_ignores_unknown_sentiment_labels_and_blank_aspects():
    records = [
        ("screen", "positive"),
        ("screen", "conflict"),  # dropped: not one of the 3 supported labels
        ("  ", "positive"),  # dropped: blank aspect
    ]

    summaries = aggregate_aspect_sentiment(records)

    assert len(summaries) == 1
    assert summaries[0].aspect == "screen"
    assert summaries[0].total == 1


def test_min_mentions_filters_rare_aspects():
    records = [("screen", "positive"), ("battery", "positive"), ("battery", "negative")]

    summaries = aggregate_aspect_sentiment(records, min_mentions=2)

    assert [s.aspect for s in summaries] == ["battery"]


def test_majority_sentiment_tie_break_prefers_positive_then_negative():
    tie_pos_neg = aggregate_aspect_sentiment([("x", "positive"), ("x", "negative")])
    assert tie_pos_neg[0].majority_sentiment == "positive"

    tie_neg_neutral = aggregate_aspect_sentiment([("x", "negative"), ("x", "neutral")])
    assert tie_neg_neutral[0].majority_sentiment == "negative"
