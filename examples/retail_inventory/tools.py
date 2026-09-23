"""Retail inventory tools: demand forecasting and reorder recommendations.

The analytics are plain functions (easy to test and reuse); ``build_registry``
wraps them as agent tools bound to a dataset.

Formulas (continuous review with a periodic order cycle):

* safety stock  ``SS  = z * sqrt(LT * sigma_d^2 + d^2 * sigma_LT^2)``
* reorder point ``ROP = d * LT + SS``
* order-up-to   ``S   = d * (LT + R) + SS``
* if inventory position (on hand + on order) <= ROP, order ``S - IP``,
  at least the supplier MOQ, rounded up to a whole pack.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from common.agent import ToolRegistry

DATA_DIR = Path(__file__).parent / "data"
Method = Literal["sma", "ses"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Product:
    sku: str
    name: str
    on_hand: int
    on_order: int
    unit_cost: float
    supplier: str
    lead_time_days: float
    lead_time_std_days: float
    min_order_qty: int
    pack_size: int


@dataclass
class InventoryData:
    products: dict[str, Product]
    sales: dict[str, list[tuple[str, int]]]  # sku -> [(iso date, units)] sorted by date

    def product(self, sku: str) -> Product:
        key = sku.strip().upper()
        if key not in self.products:
            raise KeyError(f"unknown SKU {sku!r}; known: {', '.join(sorted(self.products))}")
        return self.products[key]

    def demand(self, sku: str) -> list[int]:
        return [u for _, u in self.sales.get(self.product(sku).sku, [])]


def load_data(data_dir: Path | None = None) -> InventoryData:
    data_dir = Path(data_dir or DATA_DIR)
    with open(data_dir / "lead_times.csv", newline="") as fh:
        lead = {r["sku"]: r for r in csv.DictReader(fh)}
    products: dict[str, Product] = {}
    with open(data_dir / "stock.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            lt = lead[r["sku"]]
            products[r["sku"]] = Product(
                sku=r["sku"], name=r["name"], on_hand=int(r["on_hand"]),
                on_order=int(r["on_order"]), unit_cost=float(r["unit_cost"]),
                supplier=lt["supplier"], lead_time_days=float(lt["lead_time_days"]),
                lead_time_std_days=float(lt["lead_time_std_days"]),
                min_order_qty=int(lt["min_order_qty"]), pack_size=int(lt["pack_size"]),
            )
    sales: dict[str, list[tuple[str, int]]] = {}
    with open(data_dir / "sales_history.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            sales.setdefault(r["sku"], []).append((r["date"], int(r["units_sold"])))
    for rows in sales.values():
        rows.sort()
    return InventoryData(products=products, sales=sales)


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
class Forecast(BaseModel):
    sku: str
    method: Method
    daily_forecast: float = Field(description="expected units per day")
    daily_std: float = Field(description="std-dev of daily demand / forecast error")
    horizon_days: int
    horizon_total: float
    history_days: int
    last_7_day_avg: float


def moving_average(series: list[int], window: int) -> tuple[float, float]:
    """Simple moving average of the last ``window`` points and their std-dev."""
    if not series:
        raise ValueError("no sales history")
    tail = series[-window:]
    mean = statistics.fmean(tail)
    std = statistics.stdev(tail) if len(tail) > 1 else 0.0
    return mean, std


def exponential_smoothing(series: list[int], alpha: float, window: int) -> tuple[float, float]:
    """Simple exponential smoothing. Returns (level, RMSE of one-step errors in last window)."""
    if not series:
        raise ValueError("no sales history")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    init = min(7, len(series))
    level = statistics.fmean(series[:init])
    errors: list[float] = []
    for y in series[init:]:
        errors.append(y - level)
        level = alpha * y + (1 - alpha) * level
    recent = errors[-window:]
    rmse = math.sqrt(sum(e * e for e in recent) / len(recent)) if recent else 0.0
    return level, rmse


def forecast(data: InventoryData, sku: str, method: Method = "ses", horizon_days: int = 30,
             window: int = 28, alpha: float = 0.3) -> Forecast:
    p = data.product(sku)
    series = data.demand(p.sku)
    if method == "sma":
        mean, std = moving_average(series, window)
    else:
        mean, std = exponential_smoothing(series, alpha, window)
    return Forecast(
        sku=p.sku, method=method, daily_forecast=round(mean, 3), daily_std=round(std, 3),
        horizon_days=horizon_days, horizon_total=round(mean * horizon_days, 1),
        history_days=len(series), last_7_day_avg=round(statistics.fmean(series[-7:]), 3),
    )


# --------------------------------------------------------------------------- #
# Reorder policy
# --------------------------------------------------------------------------- #
class ReorderRecommendation(BaseModel):
    sku: str
    name: str
    status: Literal["critical", "reorder", "ok"]
    reason: str
    daily_forecast: float
    on_hand: int
    on_order: int
    inventory_position: int
    days_of_cover: float | None
    lead_time_days: float
    safety_stock: float
    reorder_point: float
    order_up_to: float
    recommended_order_qty: int
    order_cost: float
    supplier: str


def reorder_policy(data: InventoryData, sku: str, service_level: float = 0.95,
                   review_period_days: int = 7, method: Method = "ses") -> ReorderRecommendation:
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be in [0.5, 1)")
    p = data.product(sku)
    fc = forecast(data, p.sku, method=method)
    d, sd = fc.daily_forecast, fc.daily_std
    lt, slt = p.lead_time_days, p.lead_time_std_days
    z = statistics.NormalDist().inv_cdf(service_level)
    ss = z * math.sqrt(lt * sd**2 + d**2 * slt**2)
    rop = d * lt + ss
    target = d * (lt + review_period_days) + ss
    ip = p.on_hand + p.on_order
    cover = round(p.on_hand / d, 1) if d > 0 else None

    qty = 0
    if ip <= rop:
        qty = max(math.ceil(target - ip), p.min_order_qty)
        qty = math.ceil(qty / p.pack_size) * p.pack_size

    if cover is not None and cover < lt and qty > 0:
        status = "critical"
        reason = (f"on-hand covers ~{cover} days but lead time is {lt:g} days: "
                  "expect a stockout before a new order arrives")
    elif qty > 0:
        status = "reorder"
        reason = f"inventory position {ip} is at or below reorder point {rop:.0f}"
    else:
        status = "ok"
        reason = f"inventory position {ip} is above reorder point {rop:.0f}"

    return ReorderRecommendation(
        sku=p.sku, name=p.name, status=status, reason=reason, daily_forecast=d,
        on_hand=p.on_hand, on_order=p.on_order, inventory_position=ip, days_of_cover=cover,
        lead_time_days=lt, safety_stock=round(ss, 1), reorder_point=round(rop, 1),
        order_up_to=round(target, 1), recommended_order_qty=qty,
        order_cost=round(qty * p.unit_cost, 2), supplier=p.supplier,
    )


def reorder_report(data: InventoryData, service_level: float = 0.95,
                   review_period_days: int = 7, method: Method = "ses") -> dict:
    recs = [reorder_policy(data, s, service_level, review_period_days, method)
            for s in sorted(data.products)]
    rank = {"critical": 0, "reorder": 1, "ok": 2}
    recs.sort(key=lambda r: (rank[r.status], r.days_of_cover or 0))
    to_order = [r for r in recs if r.recommended_order_qty > 0]
    return {
        "service_level": service_level,
        "method": method,
        "skus_checked": len(recs),
        "skus_to_order": len(to_order),
        "total_order_cost": round(sum(r.order_cost for r in to_order), 2),
        "recommendations": [r.model_dump() for r in recs],
    }


# --------------------------------------------------------------------------- #
# Agent tools
# --------------------------------------------------------------------------- #
SKU = Annotated[str, Field(description="product SKU, e.g. SKU-1001")]
ServiceLevel = Annotated[float, Field(ge=0.5, lt=1, description="target cycle service level")]
MethodArg = Annotated[Method, Field(description="'sma' moving average or 'ses' exp. smoothing")]


def build_registry(data_dir: Path | None = None, data: InventoryData | None = None) -> ToolRegistry:
    data = data or load_data(data_dir)
    reg = ToolRegistry()

    @reg.tool
    def list_products() -> list[dict]:
        """List all products with current stock, supplier and lead time."""
        return [{"sku": p.sku, "name": p.name, "on_hand": p.on_hand, "on_order": p.on_order,
                 "lead_time_days": p.lead_time_days, "supplier": p.supplier}
                for p in data.products.values()]

    @reg.tool
    def get_sales_history(
        sku: SKU,
        last_n_days: Annotated[int, Field(ge=1, le=365)] = 28,
    ) -> dict:
        """Daily units sold for one SKU over the most recent N days."""
        p = data.product(sku)
        rows = data.sales.get(p.sku, [])[-last_n_days:]
        units = [u for _, u in rows]
        return {"sku": p.sku, "days": len(rows), "total_units": sum(units),
                "daily": [{"date": d, "units": u} for d, u in rows]}

    @reg.tool
    def forecast_demand(
        sku: SKU,
        method: MethodArg = "ses",
        horizon_days: Annotated[int, Field(ge=1, le=180)] = 30,
        window: Annotated[int, Field(ge=3, le=120, description="lookback days")] = 28,
        alpha: Annotated[float, Field(gt=0, le=1, description="SES smoothing factor")] = 0.3,
    ) -> Forecast:
        """Forecast daily demand for a SKU with a simple moving average or exponential smoothing."""
        return forecast(data, sku, method, horizon_days, window, alpha)

    @reg.tool
    def compute_reorder(
        sku: SKU,
        service_level: ServiceLevel = 0.95,
        review_period_days: Annotated[int, Field(ge=0, le=60)] = 7,
        method: MethodArg = "ses",
    ) -> ReorderRecommendation:
        """Compute safety stock, reorder point and recommended order quantity for one SKU."""
        return reorder_policy(data, sku, service_level, review_period_days, method)

    @reg.tool(name="reorder_report")
    def reorder_report_tool(
        service_level: ServiceLevel = 0.95,
        review_period_days: Annotated[int, Field(ge=0, le=60)] = 7,
        method: MethodArg = "ses",
    ) -> dict:
        """Run the reorder policy for every SKU, most urgent first, with total order cost."""
        return reorder_report(data, service_level, review_period_days, method)

    return reg
