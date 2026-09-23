"""Regenerate the SYNTHETIC maintenance log (deterministic, stdlib only).

    python examples/manufacturing_maintenance/data/generate_data.py

Machines, technicians and events are made up. The log intentionally contains
a few messy lines (comments, blank lines, a malformed entry, mixed units and
casing) so the parser has something realistic to deal with.
"""

from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

HERE = Path(__file__).parent
START = dt.datetime(2026, 1, 5, 6, 0)
END = dt.datetime(2026, 8, 28, 22, 0)

MACHINES = [
    # id, type, line, install year
    ("PRESS-01", "hydraulic press", "Line A", 2016),
    ("PRESS-02", "hydraulic press", "Line A", 2011),
    ("CNC-01", "cnc mill", "Line B", 2019),
    ("CNC-03", "cnc mill", "Line B", 2014),
    ("CONV-01", "conveyor", "Line A", 2018),
    ("WELD-02", "robotic welder", "Line C", 2020),
]

# machine -> list of (component phrase templates, mean days between failures, trend)
# trend < 1 means failures get more frequent over time.
FAILURE_MODES = {
    "PRESS-01": [(["hydraulic hose weeping", "minor hydraulic leak at fitting"], 45, 1.0)],
    "PRESS-02": [(["hydraulic pump leak, replaced seal", "Hydraulic pressure drop, pump seal",
                   "hyd. pump cavitation noise"], 19, 0.85),
                 (["limit switch sensor fault", "proximity sensor misread"], 60, 1.0)],
    "CNC-01": [(["coolant pump clogged", "coolant flow low"], 90, 1.0)],
    "CNC-03": [(["spindle bearing overheating", "spindle brg vibration high",
                 "Spindle bearing noise, regreased"], 24, 0.7),
               (["servo motor overload trip"], 75, 1.0)],
    "CONV-01": [(["belt slipping, re-tensioned", "conveyor belt torn edge", "belt misalignment"],
                 17, 1.0),
                (["drive motor overheating"], 95, 1.0)],
    "WELD-02": [(["wire feeder jam", "torch tip worn, replaced"], 45, 1.0),
                (["PLC comms timeout", "controller fault code E42"], 110, 1.0)],
}
TECHS = ["R. Ortiz", "M. Chen", "A. Novak", "S. Patel", "J. Okafor"]


def fmt_downtime(minutes: int, rng: random.Random) -> str:
    return f"{minutes / 60:.1f}h" if minutes >= 120 and rng.random() < 0.4 else f"{minutes}m"


def main() -> None:
    rng = random.Random(1234)
    events: list[tuple[dt.datetime, str, str, str, str, str]] = []
    total_days = (END - START).days
    for mid, modes in FAILURE_MODES.items():
        for phrases, mean_days, trend in modes:
            t, gap = START, mean_days
            while True:
                t = t + dt.timedelta(days=rng.expovariate(1 / gap), hours=rng.randint(0, 12))
                if t > END:
                    break
                gap = max(6.0, gap * trend ** 0.3)  # gradual acceleration
                mins = int(rng.lognormvariate(4.3, 0.5))
                etype = rng.choice(["FAILURE", "FAILURE", "Failure", "failure"])
                events.append((t, mid, etype, rng.choice(phrases),
                               fmt_downtime(mins, rng), rng.choice(TECHS)))
        # preventive maintenance every ~30 days and inspections every ~14 days
        for every, etype, text, mins in [(30, "PREVENTIVE", "scheduled PM completed", 60),
                                         (14, "INSPECTION", "routine inspection, no issues", 15)]:
            for k in range(1, total_days // every + 1):
                t = START + dt.timedelta(days=k * every, hours=rng.randint(0, 8))
                events.append((t, mid, etype, text, f"{mins}m", rng.choice(TECHS)))

    events.sort()
    lines = [
        "# SYNTHETIC maintenance log for the ai-agent-examples demo (not real data)",
        "# timestamp | machine | event | description | downtime | technician",
        "",
    ]
    for i, (t, mid, etype, desc, down, tech) in enumerate(events):
        lines.append(
            f"{t:%Y-%m-%d %H:%M} | {mid} | {etype} | {desc} | downtime={down} | tech={tech}"
        )
        if i == 40:
            lines.append("2026-03-?? | CONV-01 | FAILURE belt noise")  # malformed on purpose
        if i == 90:
            lines.append("")
    (HERE / "maintenance_log.txt").write_text("\n".join(lines) + "\n")

    with open(HERE / "machines.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["machine_id", "machine_type", "line", "install_year"])
        w.writerows(MACHINES)


if __name__ == "__main__":
    main()
