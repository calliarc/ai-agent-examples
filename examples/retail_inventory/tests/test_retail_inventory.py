"""Offline tests for the retail inventory example (tools + agent loop with the mock provider)."""

from __future__ import annotations

import csv
import json
import math
import statistics

import pytest

from common.agent import Agent, ToolCall
from common.providers import ScriptedProvider
from examples.retail_inventory import tools as inv
from examples.retail_inventory.prompts import SYSTEM_PROMPT, mock_script


@pytest.fixture(scope="module")
def data() -> inv.InventoryData:
    return inv.load_data()


def test_sample_data_loads(data):
    assert len(data.products) == 8
    assert all(len(rows) == 120 for rows in data.sales.values())
    assert data.product("sku-1001").name.startswith("Organic")
    with pytest.raises(KeyError):
        data.product("SKU-9999")


def test_moving_average_and_ses():
    assert inv.moving_average([1, 2, 3, 10, 10, 10], window=3) == (10.0, 0.0)
    level, rmse = inv.exponential_smoothing([5] * 30, alpha=0.5, window=10)
    assert level == 5 and rmse == 0
    # SES tracks a level shift; SMA over a long window lags behind it
    series = [10] * 30 + [20] * 10
    ses, _ = inv.exponential_smoothing(series, alpha=0.5, window=10)
    sma, _ = inv.moving_average(series, window=28)
    assert ses > 19.9 and sma < 14
    with pytest.raises(ValueError):
        inv.exponential_smoothing([1, 2], alpha=0, window=2)


def test_reorder_math_matches_formula(data):
    rec = inv.reorder_policy(data, "SKU-1006", service_level=0.95, review_period_days=7)
    p = data.product("SKU-1006")
    fc = inv.forecast(data, "SKU-1006")
    z = statistics.NormalDist().inv_cdf(0.95)
    ss = z * math.sqrt(p.lead_time_days * fc.daily_std**2
                       + fc.daily_forecast**2 * p.lead_time_std_days**2)
    assert rec.safety_stock == pytest.approx(ss, abs=0.1)
    assert rec.reorder_point == pytest.approx(fc.daily_forecast * p.lead_time_days + ss, abs=0.1)
    assert rec.inventory_position == p.on_hand + p.on_order
    assert rec.status == "critical"
    assert rec.recommended_order_qty % p.pack_size == 0
    assert rec.recommended_order_qty >= p.min_order_qty
    assert rec.inventory_position + rec.recommended_order_qty >= rec.order_up_to


def test_higher_service_level_orders_more(data):
    lo = inv.reorder_policy(data, "SKU-1001", service_level=0.90)
    hi = inv.reorder_policy(data, "SKU-1001", service_level=0.99)
    assert hi.safety_stock > lo.safety_stock
    assert hi.recommended_order_qty >= lo.recommended_order_qty


def test_ok_sku_orders_nothing(data):
    rec = inv.reorder_policy(data, "SKU-1004")
    assert rec.status == "ok" and rec.recommended_order_qty == 0 and rec.order_cost == 0


def test_report_sorted_by_urgency(data):
    report = inv.reorder_report(data)
    statuses = [r["status"] for r in report["recommendations"]]
    assert statuses == sorted(statuses, key=["critical", "reorder", "ok"].index)
    assert report["skus_to_order"] == sum(r["recommended_order_qty"] > 0
                                          for r in report["recommendations"])


def test_custom_data_dir(tmp_path):
    (tmp_path / "stock.csv").write_text("sku,name,on_hand,on_order,unit_cost\nA,Widget,0,0,2.00\n")
    (tmp_path / "lead_times.csv").write_text(
        "sku,supplier,lead_time_days,lead_time_std_days,min_order_qty,pack_size\n"
        "A,Acme,7,0,10,5\n")
    with open(tmp_path / "sales_history.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "sku", "units_sold"])
        w.writerows([[f"2026-01-{d:02d}", "A", 4] for d in range(1, 29)])
    reg = inv.build_registry(tmp_path)
    out, err = reg.execute(ToolCall("1", "compute_reorder", {"sku": "A", "method": "sma"}))
    rec = json.loads(out)
    assert not err
    # constant demand 4/day, LT 7, R 7, no variability -> SS 0, S = 56, order 56 rounded to 60
    assert rec["safety_stock"] == 0 and rec["reorder_point"] == 28
    assert rec["recommended_order_qty"] == 60 and rec["status"] == "critical"


def test_tool_schemas(data):
    reg = inv.build_registry(data=data)
    assert set(reg.names) == {"list_products", "get_sales_history", "forecast_demand",
                              "compute_reorder", "reorder_report"}
    out, err = reg.execute(ToolCall("1", "compute_reorder", {"sku": "SKU-1001",
                                                             "service_level": 1.5}))
    assert err and "service_level" in out
    out, err = reg.execute(ToolCall("1", "forecast_demand", {"sku": "SKU-1001",
                                                             "method": "arima"}))
    assert err


def test_agent_with_mock_provider(data):
    provider = ScriptedProvider(mock_script())
    result = Agent(provider, inv.build_registry(data=data), SYSTEM_PROMPT).run("reorder?")
    assert result.stop_reason == "final"
    assert result.trace.tool_calls[0] == "reorder_report"
    assert "forecast_demand" in result.trace.tool_calls
    assert "Purchase plan" in result.output and "SKU-1006" in result.output
    assert not any(e.data.get("is_error") for e in result.trace.events if e.kind == "tool")
