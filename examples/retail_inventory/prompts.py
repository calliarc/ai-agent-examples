"""Prompts for the retail inventory agent, plus the offline mock script."""

from __future__ import annotations

from common.agent import Message
from common.providers import Step, final_step, tool_results, tool_step

SYSTEM_PROMPT = """\
You are an inventory planning assistant for a small retailer.

Use the tools to ground every number you report: never invent stock levels,
forecasts or order quantities. Typical workflow:
1. Call reorder_report to see which SKUs need attention.
2. For critical or borderline SKUs, call forecast_demand or get_sales_history
   to explain the trend (compare the 'sma' and 'ses' methods if useful).
3. Answer with a short purchase plan: a table of SKU, name, status, order qty,
   supplier and cost, followed by 2-4 bullet points of reasoning and risks.

If the user asks for a different service level or review period, pass it to
the tools rather than adjusting numbers yourself. Keep the answer concise.
"""

DEFAULT_TASK = (
    "Which products should we reorder this week at a 95% service level? "
    "Give me the purchase plan and flag anything at risk of stocking out."
)


def format_plan(messages: list[Message]) -> str:
    """Render the mock agent's final answer from the tool results it saw."""
    report = tool_results(messages, "reorder_report")[-1]
    forecasts = {f["sku"]: f for f in tool_results(messages, "forecast_demand")}
    lines = [
        f"Purchase plan (service level {report['service_level']:.0%}, method "
        f"{report['method']}): {report['skus_to_order']} of {report['skus_checked']} SKUs "
        f"to order, total ${report['total_order_cost']:,.2f}.",
        "",
        "| SKU | Product | Status | Order qty | Supplier | Cost |",
        "|---|---|---|---:|---|---:|",
    ]
    for r in report["recommendations"]:
        if r["recommended_order_qty"] > 0:
            lines.append(f"| {r['sku']} | {r['name']} | {r['status']} | "
                         f"{r['recommended_order_qty']} | {r['supplier']} | "
                         f"${r['order_cost']:,.2f} |")
    lines.append("")
    for r in report["recommendations"]:
        if r["status"] == "critical":
            fc = forecasts.get(r["sku"])
            trend = ""
            if fc:
                trend = (f" Last 7 days averaged {fc['last_7_day_avg']:.1f}/day vs "
                         f"forecast {fc['daily_forecast']:.1f}/day.")
            lines.append(f"- {r['sku']} is critical: {r['reason']}.{trend}")
    ok = [r["sku"] for r in report["recommendations"] if r["status"] == "ok"]
    if ok:
        lines.append(f"- No order needed yet for: {', '.join(ok)}.")
    lines.append("(Generated offline by the scripted mock provider.)")
    return "\n".join(lines)


def _forecast_critical(messages: list[Message]) -> Message:
    report = tool_results(messages, "reorder_report")[-1]
    critical = [r["sku"] for r in report["recommendations"] if r["status"] == "critical"]
    calls = [("forecast_demand", {"sku": s, "method": "ses"}) for s in critical]
    if not calls:
        return final_step(format_plan)(messages)  # type: ignore[operator]
    return tool_step(*calls)


def mock_script() -> list[Step]:
    """Deterministic 'model' used by --provider mock and the tests."""
    return [
        tool_step(("reorder_report", {"service_level": 0.95})),
        _forecast_critical,
        final_step(format_plan),
    ]
