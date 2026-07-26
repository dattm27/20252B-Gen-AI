from src.baseline.aste_lookup_baseline import AsteLookupBaseline
from src.data.aste_loader import AsteSentence, AsteTriplet


def make_sentence(text, triplets):
    return AsteSentence(text=text, triplets=[AsteTriplet(*t) for t in triplets])


def test_fit_builds_aspect_vocab_and_majority_opinion_sentiment():
    train = [
        make_sentence("The food is good .", [("food", "good", "positive")]),
        make_sentence("The food was bad .", [("food", "bad", "negative")]),
        # "slow" seen twice as negative, once as positive -> majority vote should keep negative
        make_sentence("Service was slow .", [("service", "slow", "negative")]),
        make_sentence("Service was slow but ok .", [("service", "slow", "negative")]),
        make_sentence("Weirdly slow was fine .", [("pace", "slow", "positive")]),
    ]
    model = AsteLookupBaseline().fit(train)

    assert model.aspect_vocab == {"food", "service", "pace"}
    assert model.opinion_sentiment["good"] == "positive"
    assert model.opinion_sentiment["bad"] == "negative"
    assert model.opinion_sentiment["slow"] == "negative"


def test_predict_exact_verbatim_match():
    train = [make_sentence("The food is good .", [("food", "good", "positive")])]
    model = AsteLookupBaseline().fit(train)

    preds = model.predict("The food is good .")
    assert preds == [AsteTriplet(aspect="food", opinion="good", sentiment="positive")]


def test_predict_returns_empty_when_nothing_matches():
    train = [make_sentence("The food is good .", [("food", "good", "positive")])]
    model = AsteLookupBaseline().fit(train)

    assert model.predict("Completely unrelated sentence here .") == []


def test_predict_pairs_each_aspect_with_nearest_opinion():
    train = [
        make_sentence("x", [("food", "great", "positive")]),
        make_sentence("x", [("service", "rude", "negative")]),
    ]
    model = AsteLookupBaseline().fit(train)

    # "great" should pair with "food" (closer) and "rude" with "service" (closer)
    preds = model.predict("The food was great but the service was rude .")
    assert AsteTriplet(aspect="food", opinion="great", sentiment="positive") in preds
    assert AsteTriplet(aspect="service", opinion="rude", sentiment="negative") in preds
    assert len(preds) == 2


def test_predict_matches_multi_word_ngram_phrases():
    train = [make_sentence("x", [("battery life", "above average", "positive")])]
    model = AsteLookupBaseline().fit(train)

    preds = model.predict("The battery life was above average today .")
    assert preds == [AsteTriplet(aspect="battery life", opinion="above average", sentiment="positive")]


def test_predict_is_case_insensitive():
    train = [make_sentence("x", [("food", "good", "positive")])]
    model = AsteLookupBaseline().fit(train)

    preds = model.predict("The FOOD is GOOD .")
    assert preds == [AsteTriplet(aspect="food", opinion="good", sentiment="positive")]
