"""Prompts for the real estate listings agent, plus the offline mock script."""

from __future__ import annotations

from common.agent import Message
from common.providers import Step, final_step, tool_results, tool_step

SYSTEM_PROMPT = """\
You are a real estate research assistant helping an agent prepare for a client.

Ground every fact in tool output; never invent square footage, prices or
features. Typical workflow:
1. summarize_listing for the subject property.
2. find_comparables (sold comps by default) to benchmark the price.
3. Write: a 2-3 sentence plain-English summary of the property, a short table
   of the top comparables (address, price, $/sqft, distance, similarity), and
   a one-line pricing verdict using the estimate range.

Be neutral and factual. Avoid language that describes or steers by the
demographics of a neighborhood or its residents (fair-housing rules).
Note that the value estimate is a rough comps-based indication, not an appraisal.
"""

DEFAULT_TASK = (
    "Summarize listing L-2019 for a buyer and tell me whether the asking price "
    "looks fair compared with recent sales nearby."
)


def format_answer(messages: list[Message]) -> str:
    summary = tool_results(messages, "summarize_listing")[-1]
    comps = tool_results(messages, "find_comparables")[-1]
    est = comps["estimate"]
    lines = [
        f"{summary['address']}: {summary['headline']}.",
        f"{summary['price_label'].capitalize()} ${summary['price']:,} "
        f"(${summary['price_per_sqft']:,.0f}/sqft), {summary['pricing_position']}.",
        f"Key features: {', '.join(summary['key_features'])}.",
        "",
        "| Comparable | Price | $/sqft | Distance | Similarity |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in comps["comparables"]:
        lines.append(f"| {c['listing_id']} {c['address']} | ${c['price']:,} | "
                     f"${c['price_per_sqft']:,.0f} | {c['distance_km']:.1f} km | "
                     f"{c['similarity']:.0f} |")
    lines.append("")
    if est:
        pct = est["subject_vs_estimate_pct"]
        verdict = ("above" if pct > 5 else "below" if pct < -5 else "close to")
        lines.append(
            f"Verdict: the asking price is {abs(pct):.0f}% {verdict} the comps-based estimate "
            f"of ${est['estimated_value']:,} (range ${est['low']:,} - ${est['high']:,})."
            if verdict != "close to" else
            f"Verdict: the asking price is close to the comps-based estimate of "
            f"${est['estimated_value']:,} (range ${est['low']:,} - ${est['high']:,})."
        )
    else:
        lines.append("Verdict: no comparable sales found within the search radius.")
    lines.append("This is a rough comps-based indication, not an appraisal. "
                 "(Generated offline by the scripted mock provider.)")
    return "\n".join(lines)


def mock_script(listing_id: str = "L-2019") -> list[Step]:
    """Deterministic 'model' used by --provider mock and the tests."""
    return [
        tool_step(("summarize_listing", {"listing_id": listing_id})),
        tool_step(("find_comparables", {"listing_id": listing_id, "radius_km": 3.0,
                                        "status": "sold"})),
        final_step(format_answer),
    ]
