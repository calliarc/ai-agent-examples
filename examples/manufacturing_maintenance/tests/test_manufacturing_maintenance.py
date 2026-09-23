"""Offline tests for the maintenance-log example (parser, analytics, agent loop with mock)."""

from __future__ import annotations

import json

import pytest

from common.agent import Agent, ToolCall
from common.providers import ScriptedProvider
from examples.manufacturing_maintenance import tools as mt
from examples.manufacturing_maintenance.prompts import SYSTEM_PROMPT, mock_script

SMALL_LOG = """\
# comment line
2026-01-01 00:00 | M1 | inspection | routine check | downtime=0m
2026-01-02 00:00 | M1 | FAILURE | Hydraulic pump leak | downtime=60m | tech=A
2026-01-04 00:00 | m1 | failure | hyd. pump seal worn | downtime=1.5h
2026-01-06 00:00 | M1 | Failure | pump seal replaced after hydraulic drop | downtime=30
2026-01-08 00:00 | M2 | failure | conveyor belt torn | downtime=2h

garbage line without separators
2026-01-11 00:00 | M2 | exploded | nope | downtime=1m
2026-01-11 00:00 | M2 | preventive | PM done | downtime=45m
"""


@pytest.fixture
def small() -> mt.ParsedLog:
    return mt.parse_text(SMALL_LOG)


@pytest.mark.parametrize("value, minutes", [("95m", 95), ("1.5h", 90), ("30", 30),
                                            ("2 hours", 120), ("10 min", 10)])
def test_parse_downtime(value, minutes):
    assert mt.parse_downtime(value) == minutes


@pytest.mark.parametrize("text, component", [
    ("Hydraulic pressure drop", "hydraulic"), ("hyd. pump cavitation", "hydraulic"),
    ("spindle brg vibration", "spindle_bearing"), ("belt misalignment", "belt"),
    ("PLC comms timeout", "controller"), ("something odd", "other"),
])
def test_classify_component(text, component):
    assert mt.classify_component(text) == component


def test_parser_handles_messy_lines(small):
    assert len(small.records) == 6
    assert [s["line_no"] for s in small.skipped] == [8, 9]
    assert "unknown event type" in small.skipped[1]["error"]
    fails = small.failures
    assert len(fails) == 4
    assert {f.machine_id for f in fails} == {"M1", "M2"}  # case normalized
    assert [f.downtime_min for f in fails] == [60, 90, 30, 120]
    assert fails[0].technician == "A"


def test_recurring_failures(small):
    groups = mt.recurring_failures(small, min_occurrences=3)
    assert len(groups) == 1
    g = groups[0]
    assert (g["machine_id"], g["component"], g["occurrences"]) == ("M1", "hydraulic", 3)
    assert g["mean_days_between"] == 2.0 and g["total_downtime_hours"] == 3.0
    fleet = mt.recurring_failures(small, min_occurrences=1, group_by="component")
    assert {g["component"] for g in fleet} == {"hydraulic", "belt"}


def test_mtbf(small):
    rel = {r.machine_id: r for r in mt.reliability(small, hours_per_day=24)}
    # window 2026-01-01 .. 2026-01-11 = 10 days = 240 h
    m1 = rel["M1"]
    assert m1.failures == 3
    assert m1.operating_hours == pytest.approx(240 - 3.0)
    assert m1.mtbf_hours == pytest.approx((240 - 3.0) / 3, abs=0.1)
    assert m1.mttr_minutes == 60
    assert rel["M2"].mtbf_hours == pytest.approx(238.0)
    assert m1.top_component == "hydraulic"
    with pytest.raises(KeyError):
        mt.reliability(small, machine_id="M9")


def test_sample_log_flags_expected_machines():
    log = mt.load_log()
    assert len(log.skipped) == 1  # the deliberately malformed line
    risks = mt.risk_assessment(log)
    flagged = {r["machine_id"] for r in risks if r["at_risk"]}
    assert {"PRESS-02", "CNC-03"} <= flagged
    assert "PRESS-01" not in flagged
    assert [r["risk_score"] for r in risks] == sorted((r["risk_score"] for r in risks),
                                                      reverse=True)


def test_tools_via_registry(small):
    reg = mt.build_registry(log=small)
    out, err = reg.execute(ToolCall("1", "parse_logs", {}))
    info = json.loads(out)
    assert not err and info["records"] == 6 and len(info["skipped_lines"]) == 2
    out, err = reg.execute(ToolCall("2", "list_events", {"machine_id": "m1", "limit": 2}))
    assert [e["component"] for e in json.loads(out)] == ["hydraulic", "hydraulic"]
    out, err = reg.execute(ToolCall("3", "compute_mtbf", {"hours_per_day": 30}))
    assert err and "hours_per_day" in out


def test_agent_with_mock_provider():
    provider = ScriptedProvider(mock_script())
    result = Agent(provider, mt.build_registry(), SYSTEM_PROMPT).run("analyze")
    assert result.stop_reason == "final" and result.steps == 4
    assert result.trace.tool_calls == ["parse_logs", "find_recurring_failures", "compute_mtbf",
                                       "flag_at_risk_machines"]
    assert "At-risk machines" in result.output and "PRESS-02" in result.output
