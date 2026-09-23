"""Offline tests for the real estate example (tools + agent loop with the mock provider)."""

from __future__ import annotations

import json

import pytest

from common.agent import Agent, ToolCall
from common.providers import ScriptedProvider
from examples.real_estate_listings import tools as re_tools
from examples.real_estate_listings.prompts import SYSTEM_PROMPT, mock_script


@pytest.fixture(scope="module")
def listings() -> dict[str, re_tools.Listing]:
    return re_tools.load_listings()


def make(listing_id: str, **overrides) -> re_tools.Listing:
    base = dict(listing_id=listing_id, address="1 Test St", neighborhood="N", city="C",
                lat=40.0, lon=-83.0, property_type="single_family", status="sold",
                list_price=400_000, sold_price=400_000, list_date="2026-01-01", beds=3,
                baths=2, sqft=2000, lot_sqft=6000, year_built=2000)
    return re_tools.Listing(**{**base, **overrides})


def test_sample_data_valid(listings):
    assert len(listings) == 36
    assert {x.status for x in listings.values()} == {"active", "sold"}
    assert all(x.sold_price for x in listings.values() if x.status == "sold")


def test_haversine():
    assert re_tools.haversine_km(40, -83, 40, -83) == 0
    # one degree of latitude is ~111 km
    assert re_tools.haversine_km(40, -83, 41, -83) == pytest.approx(111.2, abs=0.2)


def test_identical_property_scores_100():
    s = make("A")
    score, dist = re_tools.score_pair(s, make("B"), radius_km=2)
    assert score == 100 and dist == 0


def test_score_decreases_with_differences():
    s = make("A")
    close = re_tools.score_pair(s, make("B", sqft=2100), 2)[0]
    far = re_tools.score_pair(s, make("C", sqft=2100, lat=40.01), 2)[0]
    bigger = re_tools.score_pair(s, make("D", sqft=3200, beds=5, lat=40.01), 2)[0]
    assert 100 > close > far > bigger


def test_condo_scoring_skips_lot():
    s = make("A", property_type="condo", lot_sqft=None)
    assert re_tools.score_pair(s, make("B", property_type="condo", lot_sqft=None), 2)[0] == 100


def test_comparables_filters_and_estimate():
    data = {x.listing_id: x for x in [
        make("S", status="active", sold_price=None, list_price=500_000),
        make("C1", sold_price=420_000),                      # $210/sqft
        make("C2", sold_price=440_000, lat=40.005),          # $220/sqft
        make("C3", sold_price=460_000, lon=-83.005),         # $230/sqft
        make("X1", property_type="condo", lot_sqft=None),    # wrong type
        make("X2", status="active", sold_price=None),        # not sold
        make("X3", lat=40.5),                                # too far
    ]}
    res = re_tools.comparables(data, "S", radius_km=2)
    ids = [c["listing_id"] for c in res["comparables"]]
    assert sorted(ids) == ["C1", "C2", "C3"] and ids[0] == "C1"
    assert res["estimate"]["estimated_value"] == 440_000
    assert res["estimate"]["low"] == 420_000 and res["estimate"]["high"] == 460_000
    assert res["estimate"]["subject_vs_estimate_pct"] == pytest.approx(13.6, abs=0.1)
    assert len(re_tools.comparables(data, "S", radius_km=2, status="any",
                                    same_type=False)["comparables"]) == 5


def test_summary(listings):
    s = re_tools.summarize(listings, "L-2019")
    assert s["listing_id"] == "L-2019"
    assert s["price_per_sqft"] == pytest.approx(s["price"] / listings["L-2019"].sqft, abs=0.1)
    assert "median" in s["pricing_position"]


def test_tools_via_registry(listings):
    reg = re_tools.build_registry(listings=listings)
    out, err = reg.execute(ToolCall("1", "search_listings", {"property_type": "condo",
                                                             "status": "any"}))
    rows = json.loads(out)
    assert not err and rows and all(r["property_type"] == "condo" for r in rows)
    assert [r["price"] for r in rows] == sorted(r["price"] for r in rows)
    out, err = reg.execute(ToolCall("2", "get_listing", {"listing_id": "L-0000"}))
    assert err and "unknown listing" in out
    out, err = reg.execute(ToolCall("3", "find_comparables", {"listing_id": "L-2019",
                                                              "radius_km": -1}))
    assert err


def test_agent_with_mock_provider(listings):
    provider = ScriptedProvider(mock_script("L-2019"))
    result = Agent(provider, re_tools.build_registry(listings=listings), SYSTEM_PROMPT).run("go")
    assert result.stop_reason == "final"
    assert result.trace.tool_calls == ["summarize_listing", "find_comparables"]
    assert "Verdict" in result.output and "not an appraisal" in result.output
