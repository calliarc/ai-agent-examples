"""Regenerate the SYNTHETIC retail sample data (deterministic, stdlib only).

    python examples/retail_inventory/data/generate_data.py

All products, suppliers and numbers are made up for demonstration.
"""

from __future__ import annotations

import csv
import datetime as dt
import math
import random
from pathlib import Path

HERE = Path(__file__).parent
START = dt.date(2026, 5, 4)
DAYS = 120  # through 2026-08-31

# sku, name, base daily demand, weekly trend, weekend lift, zero-sale prob, unit cost
PRODUCTS = [
    ("SKU-1001", "Organic Coffee Beans 1kg", 14.0, 0.00, 1.3, 0.00, 11.20),
    ("SKU-1002", "Oat Milk 1L", 32.0, 0.35, 1.4, 0.00, 1.10),
    ("SKU-1003", "Reusable Water Bottle", 6.0, 0.10, 1.6, 0.05, 6.50),
    ("SKU-1004", "Bamboo Toothbrush 4-pack", 9.0, -0.08, 1.1, 0.02, 2.40),
    ("SKU-1005", "Cast Iron Skillet 26cm", 1.2, 0.00, 1.8, 0.45, 18.00),
    ("SKU-1006", "LED Desk Lamp", 3.5, 0.05, 1.0, 0.15, 14.75),
    ("SKU-1007", "Sparkling Water 12-pack", 22.0, 0.50, 1.5, 0.00, 4.30),
    ("SKU-1008", "Yoga Mat", 2.5, -0.05, 1.7, 0.20, 9.90),
]

# sku, on_hand, on_order
STOCK = {
    "SKU-1001": (180, 0),
    "SKU-1002": (260, 200),
    "SKU-1003": (140, 0),
    "SKU-1004": (310, 0),
    "SKU-1005": (9, 0),
    "SKU-1006": (22, 40),
    "SKU-1007": (150, 0),
    "SKU-1008": (75, 0),
}

# sku, supplier, lead_time_days, lead_time_std_days, min_order_qty, pack_size
LEAD_TIMES = [
    ("SKU-1001", "Highland Roasters (fictional)", 10, 2.0, 50, 10),
    ("SKU-1002", "Northfield Dairy Alt (fictional)", 5, 1.0, 120, 12),
    ("SKU-1003", "BlueRiver Goods (fictional)", 21, 4.0, 48, 24),
    ("SKU-1004", "GreenLeaf Supply (fictional)", 14, 3.0, 100, 25),
    ("SKU-1005", "Forge & Co (fictional)", 30, 6.0, 12, 6),
    ("SKU-1006", "Brightline Imports (fictional)", 25, 5.0, 20, 10),
    ("SKU-1007", "Clearspring Beverages (fictional)", 4, 1.0, 60, 12),
    ("SKU-1008", "BlueRiver Goods (fictional)", 21, 4.0, 20, 10),
]


def main() -> None:
    rng = random.Random(42)
    with open(HERE / "sales_history.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "sku", "units_sold"])
        for d in range(DAYS):
            day = START + dt.timedelta(days=d)
            weekend = day.weekday() >= 5
            for sku, _, base, trend, lift, p_zero, _ in PRODUCTS:
                mean = max(0.1, base + trend * d / 7) * (lift if weekend else 1.0)
                if rng.random() < p_zero:
                    units = 0
                else:
                    units = max(0, round(rng.gauss(mean, math.sqrt(mean) * 0.9)))
                w.writerow([day.isoformat(), sku, units])

    with open(HERE / "stock.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sku", "name", "on_hand", "on_order", "unit_cost"])
        for sku, name, *_, cost in PRODUCTS:
            on_hand, on_order = STOCK[sku]
            w.writerow([sku, name, on_hand, on_order, f"{cost:.2f}"])

    with open(HERE / "lead_times.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sku", "supplier", "lead_time_days", "lead_time_std_days",
                    "min_order_qty", "pack_size"])
        w.writerows(LEAD_TIMES)


if __name__ == "__main__":
    main()
