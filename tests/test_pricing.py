"""Prices are dated references; usage gaps never imply a zero charge."""
import pytest

from biometry_parser.pricing import estimate_reference_cost, price_for


@pytest.mark.parametrize("model,incoming,outgoing", [
    ("gemini-3.5-flash-lite", .30, 2.5),
    ("gemini-3.6-flash", .75, 3.75),
])
def test_known_model_prices_are_references_not_actual_free_tier_charges(model, incoming, outgoing):
    price = price_for("gemini", model, as_of="2026-11-14")
    assert price["input_per_million_tokens"] == incoming
    assert price["output_per_million_tokens"] == outgoing
    assert price["free_tier_available"] is True
    assert price["basis"] == "paid_standard_reference"
    assert price["actual_charge_usd"] is None
    assert price["source"].startswith("https://aistudio.google.com/docs/pricing#")


@pytest.mark.parametrize("model", ["gemini-3.6-flash"])
def test_2027_rate_boundary_is_applied_without_mutating_the_old_reference(model):
    old = price_for("gemini", model, as_of="2026-12-31")
    new = price_for("gemini", model, as_of="2027-01-01")
    assert (old["input_per_million_tokens"], old["output_per_million_tokens"]) == (.75,3.75)
    assert (new["input_per_million_tokens"], new["output_per_million_tokens"]) == (1.5,7.5)
    assert new["effective_from"] == "2027-01-01" and new["valid_through"] is None
    assert old == price_for("gemini", model, as_of="2026-12-31")


def test_thought_tokens_are_added_to_gemini_visible_output_once():
    pricing = {"input_per_million_tokens":1, "output_per_million_tokens":2}
    usage = {"promptTokenCount":100, "candidatesTokenCount":200, "thoughtsTokenCount":300,
             "totalTokenCount":600}
    assert estimate_reference_cost(pricing, usage) == pytest.approx(.0011)
    assert usage["candidatesTokenCount"] == 200
    usage["totalTokenCount"] = 999999
    assert estimate_reference_cost(pricing, usage) == pytest.approx(.0011)


def test_unknown_models_and_paddle_hosted_prices_do_not_become_zero_cost():
    assert price_for("gemini", "invented-model") is None
    assert price_for("other-vlm", "gemini-3.5-flash-lite") is None
    paddle = price_for("paddleocr_api", "PP-OCRv6")
    assert all(paddle[name] is None for name in ("input_per_million_tokens","output_per_million_tokens","per_page","per_request","actual_charge_usd"))
    assert estimate_reference_cost(paddle, {"promptTokenCount":10,"candidatesTokenCount":5}) is None
    assert estimate_reference_cost(None, {"promptTokenCount":10,"candidatesTokenCount":5}) is None


@pytest.mark.parametrize("usage", [None, {}, {"promptTokenCount":10},
    {"promptTokenCount":10,"candidatesTokenCount":True},
    {"promptTokenCount":True,"candidatesTokenCount":10},
    {"promptTokenCount":10,"candidatesTokenCount":float("inf")},
    {"promptTokenCount":float("nan"),"candidatesTokenCount":10},
    {"promptTokenCount":1.5,"candidatesTokenCount":10},
    {"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":-1},
    {"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":True},
])
def test_invalid_or_incomplete_usage_returns_unknown(usage):
    assert estimate_reference_cost({"input_per_million_tokens":1,"output_per_million_tokens":2},usage) is None


@pytest.mark.parametrize("rate", [None, -1, True, float("nan"), float("inf")])
def test_invalid_reference_rates_return_unknown(rate):
    assert estimate_reference_cost({"input_per_million_tokens":rate,"output_per_million_tokens":2},
                                  {"promptTokenCount":10,"candidatesTokenCount":5}) is None
