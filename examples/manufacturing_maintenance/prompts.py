"""Prompts for the maintenance-log analyzer agent, plus the offline mock script."""

from __future__ import annotations

from common.agent import Message
from common.providers import Step, final_step, tool_results, tool_step

SYSTEM_PROMPT = """\
You are a reliability engineer's assistant for a manufacturing plant.

Use the tools for every number; never guess failure counts or MTBF. Typical workflow:
1. parse_logs to confirm the data window and note any skipped/malformed lines.
2. find_recurring_failures to see which machine/component pairs keep failing.
3. compute_mtbf and flag_at_risk_machines to rank machines by risk.
4. Answer with: a ranked list of at-risk machines (score, MTBF, top component,
   the reasons), recurring failure patterns, and 1-2 concrete next actions per
   at-risk machine (e.g. root-cause analysis, spare parts, shorter PM interval).

Mention data-quality issues (skipped lines) briefly at the end.
"""

DEFAULT_TASK = (
    "Analyze the maintenance log. Which machines are at risk of failing again soon, "
    "what keeps breaking, and what should maintenance do next?"
)

NEXT_ACTIONS = {
    "hydraulic": "run a root-cause analysis on the pump/seal and stock seal kits",
    "spindle_bearing": "schedule vibration analysis and plan a spindle bearing replacement",
    "belt": "check pulley alignment and move to a heavier-duty belt",
    "motor": "check motor load and cooling; thermal-image the drive",
    "coolant": "flush the coolant system and add a filter check to PM",
    "sensor": "replace and re-bracket the sensor; check cabling",
    "controller": "review controller logs with the OEM",
    "welding_consumable": "shorten the consumable replacement interval",
}


def format_report(messages: list[Message]) -> str:
    parsed = tool_results(messages, "parse_logs")[-1]
    groups = tool_results(messages, "find_recurring_failures")[-1]
    risks = tool_results(messages, "flag_at_risk_machines")[-1]
    at_risk = [r for r in risks if r["at_risk"]]
    w = parsed["window"]
    lines = [f"Analyzed {parsed['records']} log records ({w['start'][:10]} to {w['end'][:10]}).",
             "", f"At-risk machines ({len(at_risk)}):"]
    for i, r in enumerate(at_risk, 1):
        action = NEXT_ACTIONS.get(r["top_component"] or "", "inspect and review failure history")
        lines.append(f"{i}. {r['machine_id']} ({r['machine_type']}) - "
                     f"risk {r['risk_score']:.0f}/100, "
                     f"MTBF {r['mtbf_hours']:.0f} h, {r['failures']} failures")
        for reason in r["reasons"]:
            lines.append(f"   - {reason}")
        lines.append(f"   - Next action: {action}.")
    lines += ["", "Recurring failure patterns:"]
    for g in groups[:5]:
        lines.append(f"- {g['machine_id']} / {g['component']}: {g['occurrences']}x, every "
                     f"~{g['mean_days_between']} days, {g['total_downtime_hours']} h downtime")
    if parsed["skipped_lines"]:
        lines += ["", f"Data quality: {len(parsed['skipped_lines'])} malformed line(s) skipped "
                      f"(e.g. line {parsed['skipped_lines'][0]['line_no']})."]
    lines.append("(Generated offline by the scripted mock provider.)")
    return "\n".join(lines)


def mock_script() -> list[Step]:
    """Deterministic 'model' used by --provider mock and the tests."""
    return [
        tool_step(("parse_logs", {})),
        tool_step(("find_recurring_failures", {"min_occurrences": 3}),
                  ("compute_mtbf", {})),
        tool_step(("flag_at_risk_machines", {"threshold": 50})),
        final_step(format_report),
    ]
