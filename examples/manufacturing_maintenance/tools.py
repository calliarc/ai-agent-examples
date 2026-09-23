"""Manufacturing maintenance tools: log parsing, recurring failures, MTBF and risk flags.

Log format (one event per line, ``|``-separated)::

    2026-03-04 08:15 | PRESS-02 | FAILURE | Hydraulic pump leak | downtime=95m | tech=R. Ortiz

Components are inferred from the free-text description with a keyword
taxonomy (``COMPONENT_KEYWORDS``), so "hyd. pump cavitation" and "Hydraulic
pressure drop" both group under ``hydraulic``.

Reliability metrics per machine over the observation window:

* operating hours = window days * ``hours_per_day`` - failure downtime
* MTBF = operating hours / failures
* MTTR = mean failure downtime
* availability = MTBF / (MTBF + MTTR)
"""

from __future__ import annotations

import csv
import datetime as dt
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from common.agent import ToolRegistry

DATA_DIR = Path(__file__).parent / "data"

COMPONENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "hydraulic": ("hydraulic", "hyd.", "hyd ", "pump seal", "hose"),
    "spindle_bearing": ("spindle", "bearing", "brg"),
    "belt": ("belt",),
    "coolant": ("coolant",),
    "motor": ("motor",),
    "sensor": ("sensor", "limit switch"),
    "controller": ("plc", "controller", "fault code", "comms"),
    "welding_consumable": ("wire feeder", "torch", "tip"),
}
EventType = Literal["failure", "preventive", "inspection"]


@dataclass
class LogRecord:
    timestamp: dt.datetime
    machine_id: str
    event_type: str
    description: str
    component: str
    downtime_min: float
    technician: str | None = None
    line_no: int = 0


@dataclass
class ParsedLog:
    records: list[LogRecord]
    skipped: list[dict] = field(default_factory=list)
    machines: dict[str, dict] = field(default_factory=dict)

    @property
    def failures(self) -> list[LogRecord]:
        return [r for r in self.records if r.event_type == "failure"]

    @property
    def window(self) -> tuple[dt.datetime, dt.datetime]:
        ts = [r.timestamp for r in self.records]
        return min(ts), max(ts)


def classify_component(description: str) -> str:
    text = description.lower() + " "
    for component, words in COMPONENT_KEYWORDS.items():
        if any(w in text for w in words):
            return component
    return "other"


_DOWNTIME = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(m|min|mins|minutes|h|hr|hrs|hours)?\s*$", re.I)


def parse_downtime(value: str) -> float:
    """'95m' -> 95.0, '1.5h' -> 90.0, '30' -> 30.0 (minutes)."""
    m = _DOWNTIME.match(value)
    if not m:
        raise ValueError(f"bad downtime {value!r}")
    amount, unit = float(m.group(1)), (m.group(2) or "m").lower()
    return amount * 60 if unit.startswith("h") else amount


def parse_line(line: str, line_no: int = 0) -> LogRecord:
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 4:
        raise ValueError("expected at least 4 '|'-separated fields")
    ts = dt.datetime.strptime(parts[0], "%Y-%m-%d %H:%M")
    machine, etype, desc = parts[1].upper(), parts[2].lower(), parts[3]
    if etype not in ("failure", "preventive", "inspection"):
        raise ValueError(f"unknown event type {parts[2]!r}")
    downtime, tech = 0.0, None
    for extra in parts[4:]:
        key, _, val = extra.partition("=")
        key = key.strip().lower()
        if key == "downtime":
            downtime = parse_downtime(val)
        elif key == "tech":
            tech = val.strip() or None
    component = classify_component(desc) if etype == "failure" else "n/a"
    return LogRecord(ts, machine, etype, desc, component, downtime, tech, line_no)


def parse_text(text: str) -> ParsedLog:
    records, skipped = [], []
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            records.append(parse_line(line, i))
        except ValueError as exc:
            skipped.append({"line_no": i, "line": line[:120], "error": str(exc)})
    records.sort(key=lambda r: r.timestamp)
    return ParsedLog(records=records, skipped=skipped)


def load_log(data_dir: Path | None = None) -> ParsedLog:
    data_dir = Path(data_dir or DATA_DIR)
    parsed = parse_text((data_dir / "maintenance_log.txt").read_text(encoding="utf-8"))
    machines_csv = data_dir / "machines.csv"
    if machines_csv.exists():
        with open(machines_csv, newline="") as fh:
            parsed.machines = {r["machine_id"]: r for r in csv.DictReader(fh)}
    return parsed


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def recurring_failures(log: ParsedLog, min_occurrences: int = 3,
                       group_by: Literal["machine_component", "component"] = "machine_component"
                       ) -> list[dict]:
    groups: dict[tuple, list[LogRecord]] = defaultdict(list)
    for r in log.failures:
        key = (r.machine_id, r.component) if group_by == "machine_component" else (r.component,)
        groups[key].append(r)
    out = []
    for key, recs in groups.items():
        if len(recs) < min_occurrences:
            continue
        gaps = [(b.timestamp - a.timestamp).total_seconds() / 86400
                for a, b in zip(recs, recs[1:], strict=False)]
        out.append({
            "machine_id": key[0] if group_by == "machine_component" else None,
            "machines": sorted({r.machine_id for r in recs}),
            "component": key[-1],
            "occurrences": len(recs),
            "first_seen": recs[0].timestamp.date().isoformat(),
            "last_seen": recs[-1].timestamp.date().isoformat(),
            "mean_days_between": round(statistics.fmean(gaps), 1) if gaps else None,
            "total_downtime_hours": round(sum(r.downtime_min for r in recs) / 60, 1),
            "sample_descriptions": list(dict.fromkeys(r.description for r in recs))[:3],
        })
    return sorted(out, key=lambda g: (-g["occurrences"], -g["total_downtime_hours"]))


class MachineReliability(BaseModel):
    machine_id: str
    machine_type: str | None
    failures: int
    operating_hours: float
    mtbf_hours: float | None
    mttr_minutes: float | None
    availability_pct: float | None
    last_failure: str | None
    hours_since_last_failure: float | None
    top_component: str | None


def reliability(log: ParsedLog, machine_id: str | None = None,
                hours_per_day: float = 16.0) -> list[MachineReliability]:
    start, end = log.window
    window_days = (end - start).total_seconds() / 86400
    machines = sorted({r.machine_id for r in log.records} | set(log.machines))
    if machine_id:
        key = machine_id.strip().upper()
        if key not in machines:
            raise KeyError(f"unknown machine {machine_id!r}; known: {', '.join(machines)}")
        machines = [key]
    out = []
    for m in machines:
        fails = [r for r in log.failures if r.machine_id == m]
        down_h = sum(r.downtime_min for r in fails) / 60
        op_h = window_days * hours_per_day - down_h
        n = len(fails)
        mtbf = op_h / n if n else None
        mttr = statistics.fmean(r.downtime_min for r in fails) if n else None
        avail = mtbf / (mtbf + mttr / 60) * 100 if n else None
        last = fails[-1].timestamp if fails else None
        since = ((end - last).total_seconds() / 86400 * hours_per_day) if last else None
        top = Counter(r.component for r in fails).most_common(1)
        out.append(MachineReliability(
            machine_id=m, machine_type=log.machines.get(m, {}).get("machine_type"),
            failures=n, operating_hours=round(op_h, 1),
            mtbf_hours=round(mtbf, 1) if mtbf else None,
            mttr_minutes=round(mttr, 1) if mttr else None,
            availability_pct=round(avail, 2) if avail else None,
            last_failure=last.isoformat(timespec="minutes") if last else None,
            hours_since_last_failure=round(since, 1) if since is not None else None,
            top_component=top[0][0] if top else None,
        ))
    return out


def risk_assessment(log: ParsedLog, threshold: float = 50.0, hours_per_day: float = 16.0,
                    recent_days: int = 60) -> list[dict]:
    """Score each machine 0-100 and explain why.

    * low MTBF vs. fleet median ........ up to 35
    * a component failing repeatedly ... up to 25  (5 points per repeat beyond the first)
    * failure rate rising recently ..... up to 20  (last ``recent_days`` vs. before)
    * running past its typical MTBF .... up to 20  (hours since last failure / MTBF)
    """
    rel = reliability(log, hours_per_day=hours_per_day)
    mtbfs = [r.mtbf_hours for r in rel if r.mtbf_hours]
    fleet_median = statistics.median(mtbfs) if mtbfs else None
    start, end = log.window
    cutoff = end - dt.timedelta(days=recent_days)
    recent_span = recent_days
    prior_span = max((cutoff - start).total_seconds() / 86400, 1)
    out = []
    for r in rel:
        score, reasons = 0.0, []
        fails = [f for f in log.failures if f.machine_id == r.machine_id]
        if r.mtbf_hours and fleet_median:
            ratio = fleet_median / r.mtbf_hours
            pts = min(35.0, max(0.0, (ratio - 1) * 35))
            if pts > 0:
                score += pts
                reasons.append(f"MTBF {r.mtbf_hours:.0f} h is {ratio:.1f}x worse than fleet "
                               f"median {fleet_median:.0f} h")
        comp_counts = Counter(f.component for f in fails)
        if comp_counts:
            comp, cnt = comp_counts.most_common(1)[0]
            if cnt >= 3:
                score += min(25.0, (cnt - 1) * 5)
                reasons.append(f"recurring {comp} failures ({cnt}x)")
        recent = sum(1 for f in fails if f.timestamp >= cutoff)
        prior = len(fails) - recent
        recent_rate, prior_rate = recent / recent_span, prior / prior_span
        if recent >= 2 and recent_rate > 1.3 * prior_rate:
            score += 20
            reasons.append(f"failure rate rising: {recent} in last {recent_days} days vs "
                           f"{prior_rate * recent_days:.1f} expected from earlier history")
        if r.mtbf_hours and r.hours_since_last_failure is not None:
            due = r.hours_since_last_failure / r.mtbf_hours
            if due >= 0.8:
                score += 20 * min(due, 1.5) / 1.5
                reasons.append(f"{r.hours_since_last_failure:.0f} h since last failure "
                               f"({due:.0%} of MTBF) - next failure may be due")
        out.append({
            "machine_id": r.machine_id,
            "machine_type": r.machine_type,
            "risk_score": round(min(score, 100.0), 1),
            "at_risk": score >= threshold,
            "failures": r.failures,
            "mtbf_hours": r.mtbf_hours,
            "top_component": r.top_component,
            "reasons": reasons or ["no significant risk signals"],
        })
    return sorted(out, key=lambda x: -x["risk_score"])


# --------------------------------------------------------------------------- #
# Agent tools
# --------------------------------------------------------------------------- #
MachineId = Annotated[str | None, Field(description="machine id such as PRESS-02; omit for all")]
HoursPerDay = Annotated[float, Field(gt=0, le=24, description="scheduled operating hours per day")]


def build_registry(data_dir: Path | None = None, log: ParsedLog | None = None) -> ToolRegistry:
    log = log or load_log(data_dir)
    reg = ToolRegistry()

    @reg.tool
    def parse_logs() -> dict:
        """Parse the maintenance log and report what was read: counts, window, skipped lines."""
        start, end = log.window
        return {
            "records": len(log.records),
            "window": {"start": start.isoformat(timespec="minutes"),
                       "end": end.isoformat(timespec="minutes")},
            "event_counts": dict(Counter(r.event_type for r in log.records)),
            "failures_by_machine": dict(Counter(r.machine_id for r in log.failures)),
            "failures_by_component": dict(Counter(r.component for r in log.failures)),
            "machines": log.machines or sorted({r.machine_id for r in log.records}),
            "skipped_lines": log.skipped,
        }

    @reg.tool
    def list_events(
        machine_id: MachineId = None,
        event_type: EventType | None = "failure",
        component: str | None = None,
        limit: Annotated[int, Field(ge=1, le=200)] = 25,
    ) -> list[dict]:
        """List individual log events (most recent first), filtered by machine/type/component."""
        rows = [r for r in reversed(log.records)
                if (not machine_id or r.machine_id == machine_id.upper())
                and (not event_type or r.event_type == event_type)
                and (not component or r.component == component)]
        return [{"timestamp": r.timestamp.isoformat(timespec="minutes"),
                 "machine_id": r.machine_id, "event_type": r.event_type,
                 "component": r.component, "description": r.description,
                 "downtime_min": r.downtime_min, "technician": r.technician}
                for r in rows[:limit]]

    @reg.tool
    def find_recurring_failures(
        min_occurrences: Annotated[int, Field(ge=2, le=50)] = 3,
        group_by: Literal["machine_component", "component"] = "machine_component",
    ) -> list[dict]:
        """Group failures by machine+component (or fleet-wide by component); return repeats."""
        return recurring_failures(log, min_occurrences, group_by)

    @reg.tool
    def compute_mtbf(machine_id: MachineId = None, hours_per_day: HoursPerDay = 16.0
                     ) -> list[MachineReliability]:
        """Mean time between failures, MTTR and availability per machine."""
        return reliability(log, machine_id, hours_per_day)

    @reg.tool
    def flag_at_risk_machines(
        threshold: Annotated[float, Field(ge=0, le=100)] = 50.0,
        hours_per_day: HoursPerDay = 16.0,
        recent_days: Annotated[int, Field(ge=7, le=180)] = 60,
    ) -> list[dict]:
        """Score every machine's failure risk (0-100) with reasons; at_risk = score >= threshold."""
        return risk_assessment(log, threshold, hours_per_day, recent_days)

    return reg
