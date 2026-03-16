"""Simple forward inventory simulation engine.

Simulates a (s, S) inventory policy for 90 days per SKU-store (spec 10.2 V1
variant — deterministic, no Monte Carlo; Monte Carlo is V2).

Policy:
  - Each day: consume demand from stock.
  - If stock falls to or below the reorder point (s), place a replenishment
    order.  The order quantity restores stock to the initial_stock_on_hand
    level (S).  Orders arrive in ``round(lead_time_days)`` days.
  - Unmet demand (stockout) is treated as lost sales.

Outputs per SKU-store (aggregated over the simulation window):
  fill_rate        : 1 - total_lost_sales / total_demand
  stockout_days    : number of days stock = 0 when demand > 0
  days_of_supply   : mean(stock_on_hand / max(1, avg_daily_demand))
  avg_inventory    : mean daily stock on hand
  total_lost_sales : cumulative units of lost demand

SYNTHETIC label: all outputs are tagged as synthetic (spec rule 3.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

SYNTHETIC_TAG = "SYNTHETIC"
SIMULATION_HORIZON = 90  # days


# ---------------------------------------------------------------------------
# Per-series simulation
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Aggregated results for one SKU-store simulation."""

    id: str
    fill_rate: float            # fraction of demand fulfilled (0-1)
    stockout_days: int          # days with unfulfilled demand
    days_of_supply: float       # mean(stock / avg_daily_demand)
    avg_inventory: float        # mean daily on-hand stock
    total_demand: float
    total_fulfilled: float
    total_lost_sales: float
    total_orders_placed: int
    is_synthetic: bool = True
    synthetic_tag: str = SYNTHETIC_TAG


def simulate_series(
    series_id: str,
    demand_series: list[float],
    initial_stock: float,
    reorder_point: float,
    order_quantity: float,
    lead_time_days: int,
) -> SimulationResult:
    """Simulate one SKU-store over the demand series.

    Parameters
    ----------
    series_id:
        Unique series identifier.
    demand_series:
        Daily demand values (actual or forecast).  Length = simulation horizon.
    initial_stock:
        Starting inventory (units).
    reorder_point:
        Trigger level: place an order when stock ≤ reorder_point.
    order_quantity:
        Units per replenishment order (= initial_stock − current_stock, S-policy).
    lead_time_days:
        Fixed lead time in days (deterministic for V1).

    Returns
    -------
    SimulationResult
    """
    horizon = len(demand_series)
    stock = float(initial_stock)
    avg_d = max(1.0, np.mean(demand_series) if demand_series else 1.0)

    # Track pending orders: list of (arrival_day, quantity)
    pending_orders: list[tuple[int, float]] = []
    order_placed_this_period: bool = False  # 1 order per trigger, no duplicates

    # Metrics accumulators
    total_demand = 0.0
    total_fulfilled = 0.0
    total_lost = 0.0
    stockout_days = 0
    stock_levels: list[float] = []
    orders_placed = 0

    for day in range(horizon):
        # Receive any orders arriving today
        arrived = [qty for (arr_day, qty) in pending_orders if arr_day == day]
        for qty in arrived:
            stock += qty
        pending_orders = [(arr, qty) for (arr, qty) in pending_orders if arr != day]

        # Demand for today
        demand = max(0.0, float(demand_series[day]))
        total_demand += demand

        # Fulfill as much as possible
        fulfilled = min(stock, demand)
        lost = demand - fulfilled
        stock -= fulfilled

        total_fulfilled += fulfilled
        total_lost += lost
        if lost > 0:
            stockout_days += 1

        # Record stock level
        stock_on_hand = max(0.0, stock)
        stock_levels.append(stock_on_hand)

        # Reorder trigger: if stock ≤ ROP and no order already en-route
        # (simplified: only one outstanding order allowed at a time for V1)
        has_pending = len(pending_orders) > 0
        if stock <= reorder_point and not has_pending:
            qty_to_order = max(0.0, initial_stock - stock)
            if qty_to_order > 0:
                arrival_day = day + max(1, lead_time_days)
                pending_orders.append((arrival_day, qty_to_order))
                orders_placed += 1

    # Aggregate metrics
    fill_rate = total_fulfilled / total_demand if total_demand > 0 else 1.0
    avg_inv = float(np.mean(stock_levels)) if stock_levels else 0.0
    dos = avg_inv / avg_d  # days of supply

    return SimulationResult(
        id=series_id,
        fill_rate=round(fill_rate, 6),
        stockout_days=stockout_days,
        days_of_supply=round(dos, 4),
        avg_inventory=round(avg_inv, 4),
        total_demand=round(total_demand, 4),
        total_fulfilled=round(total_fulfilled, 4),
        total_lost_sales=round(total_lost, 4),
        total_orders_placed=orders_placed,
    )


# ---------------------------------------------------------------------------
# Batch simulation engine
# ---------------------------------------------------------------------------


def run_inventory_simulation(
    sales_df: pl.DataFrame,
    params_df: pl.DataFrame,
    ss_df: pl.DataFrame,
    rop_df: pl.DataFrame,
    forecast_df: pl.DataFrame | None = None,
    horizon: int = SIMULATION_HORIZON,
    cutoff_date=None,
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
) -> pl.DataFrame:
    """Simulate inventory for all SKU-stores.

    Parameters
    ----------
    sales_df:
        Long-format sales DataFrame (id, date, sales).
    params_df:
        Synthetic inventory parameters (id, initial_stock_on_hand,
        lead_time_days, …).
    ss_df:
        Safety stock per series (id, safety_stock).
    rop_df:
        Reorder point per series (id, reorder_point).
    forecast_df:
        Optional forecast DataFrame (id, date, forecast_p50) for future days.
    horizon:
        Number of days to simulate (default 90).
    cutoff_date:
        Last date of actual sales.  Days after this use forecast_p50.
    id_col, date_col, sales_col:
        Column name overrides.

    Returns
    -------
    pl.DataFrame  (SYNTHETIC-tagged) with one row per SKU-store:
        id, fill_rate, stockout_days, days_of_supply, avg_inventory,
        total_demand, total_fulfilled, total_lost_sales, total_orders_placed,
        safety_stock, reorder_point, initial_stock_on_hand,
        is_synthetic, synthetic_tag.
    """
    # Build lookup maps
    params_map: dict[str, dict] = {
        row[id_col]: row for row in params_df.iter_rows(named=True)
    }
    ss_map: dict[str, float] = {
        row[id_col]: float(row["safety_stock"])
        for row in ss_df.iter_rows(named=True)
    }
    rop_map: dict[str, float] = {
        row[id_col]: float(row["reorder_point"])
        for row in rop_df.iter_rows(named=True)
    }

    # Build demand series map (last `horizon` days of actual + forecast)
    demand_map: dict[str, list[float]] = {}
    for (sid,), grp in sales_df.sort([id_col, date_col]).group_by([id_col], maintain_order=True):
        demand_map[sid] = grp[sales_col].tail(horizon).to_list()

    # Extend with forecast if cutoff provided
    if forecast_df is not None and cutoff_date is not None:
        fc_future = forecast_df.filter(pl.col(date_col) > cutoff_date)
        for (sid,), grp in fc_future.sort([id_col, date_col]).group_by([id_col], maintain_order=True):
            existing = demand_map.get(sid, [])
            fc_vals = grp["forecast_p50"].to_list()
            combined = (existing + fc_vals)[:horizon]
            # Pad with last known value if still short
            if len(combined) < horizon and combined:
                combined += [combined[-1]] * (horizon - len(combined))
            demand_map[sid] = combined

    results: list[SimulationResult] = []
    all_ids = list(params_map.keys())

    for sid in all_ids:
        p = params_map[sid]
        initial_stock = float(p.get("initial_stock_on_hand", 30))
        lead_time = max(1, round(float(p.get("lead_time_days", 7.0))))
        ss = ss_map.get(sid, 0.0)
        rop = rop_map.get(sid, initial_stock * 0.25)

        demand_series = demand_map.get(sid, [])
        if not demand_series:
            # No demand data — simulate with near-zero demand
            avg_d = float(p.get("avg_daily_demand", 1.0) or 1.0)
            demand_series = [avg_d] * horizon

        # Pad or trim to exactly horizon length
        if len(demand_series) < horizon:
            fill_val = float(np.mean(demand_series)) if demand_series else 1.0
            demand_series = demand_series + [fill_val] * (horizon - len(demand_series))
        else:
            demand_series = demand_series[:horizon]

        result = simulate_series(
            series_id=sid,
            demand_series=demand_series,
            initial_stock=initial_stock,
            reorder_point=rop,
            order_quantity=initial_stock,  # (s, S) policy: order up to S
            lead_time_days=lead_time,
        )
        results.append(result)

    if not results:
        return pl.DataFrame()

    # Assemble results DataFrame
    sim_rows = [
        {
            id_col: r.id,
            "fill_rate": r.fill_rate,
            "stockout_days": r.stockout_days,
            "days_of_supply": r.days_of_supply,
            "avg_inventory": r.avg_inventory,
            "total_demand": r.total_demand,
            "total_fulfilled": r.total_fulfilled,
            "total_lost_sales": r.total_lost_sales,
            "total_orders_placed": r.total_orders_placed,
            "is_synthetic": True,
            "synthetic_tag": SYNTHETIC_TAG,
        }
        for r in results
    ]

    sim_df = pl.DataFrame(sim_rows)

    # Join back params for context
    context_cols = [id_col, "safety_stock"]
    if "reorder_point" in rop_df.columns:
        context_cols_rop = [id_col, "reorder_point"]
        sim_df = sim_df.join(ss_df.select(context_cols), on=id_col, how="left")
        sim_df = sim_df.join(rop_df.select(context_cols_rop), on=id_col, how="left")

    param_keep = [c for c in ["initial_stock_on_hand", "lead_time_mean", "abc_class", "cat_id"]
                  if c in params_df.columns]
    if param_keep:
        sim_df = sim_df.join(params_df.select([id_col] + param_keep), on=id_col, how="left")

    logger.info(
        "Simulation complete: %d SKUs, avg fill_rate=%.3f, avg stockout_days=%.1f",
        len(results),
        float(sim_df["fill_rate"].mean()),
        float(sim_df["stockout_days"].mean()),
    )

    return sim_df


def save_inventory_snapshot(
    sim_df: pl.DataFrame,
    output_dir: Path,
) -> Path:
    """Write inventory simulation results to Parquet.

    Parameters
    ----------
    sim_df:
        Output of ``run_inventory_simulation``.
    output_dir:
        Destination directory (created if needed).

    Returns
    -------
    Path to the written Parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fact_inventory_snapshot.parquet"
    sim_df.write_parquet(path, compression="snappy")
    logger.info("Inventory snapshot saved: %s (%d rows)", path, len(sim_df))
    return path
