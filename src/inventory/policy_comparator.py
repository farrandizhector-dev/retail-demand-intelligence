"""Inventory policy comparator — V2-Fase 3.

Evaluates four replenishment policies via Monte Carlo simulation:
  1. (s, Q)  — Fixed reorder point + fixed order quantity (EOQ)
  2. (s, S)  — Fixed reorder point + order-up-to level S
  3. (R, S)  — Periodic review every R days, order-up-to S
  4. SL      — Service-level driven: target fill rate ≥ SL

Each policy is evaluated using simulate_inventory_mc and results are
aggregated into a PolicyComparisonResult.  Output saved as
policy_comparison.parquet.

All outputs tagged SYNTHETIC (spec rule 3.4).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.inventory.newsvendor import (
    economic_order_quantity,
    reorder_point,
    safety_stock,
)
from src.inventory.simulator import MonteCarloResult, simulate_inventory_mc

logger = logging.getLogger(__name__)

SYNTHETIC_TAG = "SYNTHETIC"

POLICY_NAMES = ["(s,Q)", "(s,S)", "(R,S)", "SL-driven"]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PolicyComparisonResult:
    """Aggregated comparison of 4 inventory policies for one SKU-store."""

    series_id: str = ""

    # Per-policy results — list of 4 in order of POLICY_NAMES
    policy_names: list[str] = field(default_factory=lambda: list(POLICY_NAMES))
    fill_rates: list[float] = field(default_factory=list)
    stockout_probs: list[float] = field(default_factory=list)
    avg_inventories: list[float] = field(default_factory=list)
    total_costs: list[float] = field(default_factory=list)

    # Best policy name (lowest cost)
    best_policy: str = ""
    best_cost: float = 0.0

    # Raw MC results
    mc_results: list[MonteCarloResult] = field(default_factory=list)

    # Metadata
    is_synthetic: bool = True
    synthetic_tag: str = SYNTHETIC_TAG


# ---------------------------------------------------------------------------
# Policy parameter builders
# ---------------------------------------------------------------------------


def _sq_params(
    mean_daily_demand: float,
    demand_std: float,
    lead_time: float,
    holding_cost_per_unit_day: float,
    fixed_order_cost: float,
    service_level: float,
) -> dict[str, float]:
    """(s, Q) parameters: EOQ for Q, safety-stock ROP for s."""
    q = economic_order_quantity(
        annual_demand=mean_daily_demand * 365,
        order_cost=fixed_order_cost,
        holding_cost_annual=holding_cost_per_unit_day * 365,
    )
    s = reorder_point(mean_daily_demand, lead_time, demand_std, service_level)
    return {"reorder_point": s, "order_quantity": q}


def _ss_params(
    mean_daily_demand: float,
    demand_std: float,
    lead_time: float,
    horizon_days: int,
    service_level: float,
) -> dict[str, float]:
    """(s, S) parameters: s = ROP, order-up-to S = peak demand estimate."""
    s = reorder_point(mean_daily_demand, lead_time, demand_std, service_level)
    # S = expected demand for lead time + review period (here: 2 × LT)
    S_level = mean_daily_demand * (lead_time * 2) + safety_stock(demand_std, lead_time, service_level)
    order_quantity = max(1.0, S_level - s)
    return {"reorder_point": s, "order_quantity": order_quantity}


def _rs_params(
    mean_daily_demand: float,
    demand_std: float,
    lead_time: float,
    review_period: int,
    service_level: float,
) -> dict[str, float]:
    """(R, S) periodic review parameters."""
    effective_lt = lead_time + review_period
    s = reorder_point(mean_daily_demand, effective_lt, demand_std, service_level)
    order_quantity = mean_daily_demand * review_period + safety_stock(demand_std, lead_time, service_level)
    order_quantity = max(1.0, order_quantity)
    return {"reorder_point": s, "order_quantity": order_quantity}


def _sl_driven_params(
    mean_daily_demand: float,
    demand_std: float,
    lead_time: float,
    target_service_level: float,
) -> dict[str, float]:
    """Service-level driven: choose ROP and Q to hit target SL."""
    rop = reorder_point(mean_daily_demand, lead_time, demand_std, target_service_level)
    # Q = expected demand during lead time × (1 + buffer for SL)
    q = max(1.0, mean_daily_demand * lead_time * (1.0 + target_service_level))
    return {"reorder_point": rop, "order_quantity": q}


# ---------------------------------------------------------------------------
# Core policy comparator
# ---------------------------------------------------------------------------


def compare_policies(
    forecast_p10: np.ndarray,
    forecast_p50: np.ndarray,
    forecast_p90: np.ndarray,
    initial_stock: float,
    lead_time_mean: float,
    lead_time_std: float,
    holding_cost_per_unit_day: float = 0.01,
    stockout_cost_per_unit: float = 1.0,
    fixed_order_cost: float = 10.0,
    service_level: float = 0.95,
    n_simulations: int = 1000,
    horizon_days: int = 90,
    review_period: int = 7,
    seed: int = 42,
    series_id: str = "",
) -> PolicyComparisonResult:
    """Run Monte Carlo simulation for 4 inventory policies and compare.

    Parameters
    ----------
    forecast_p10, forecast_p50, forecast_p90:
        Quantile forecasts for each day of the horizon.
    initial_stock:
        Starting inventory level.
    lead_time_mean, lead_time_std:
        Lead-time distribution parameters.
    holding_cost_per_unit_day:
        Holding cost per unit per day.
    stockout_cost_per_unit:
        Stockout penalty per unit of lost sales.
    fixed_order_cost:
        Fixed cost per order placed.
    service_level:
        Target fill rate for policies that use it.
    n_simulations:
        MC paths per policy.
    horizon_days:
        Simulation horizon.
    review_period:
        Review period in days for (R, S) policy.
    seed:
        Base random seed (each policy gets seed + offset).
    series_id:
        Identifier for logging.

    Returns
    -------
    PolicyComparisonResult
    """
    p50 = np.asarray(forecast_p50, dtype=float)
    p10 = np.asarray(forecast_p10, dtype=float)
    p90 = np.asarray(forecast_p90, dtype=float)

    # Derive demand statistics from forecasts
    mean_daily = float(np.mean(p50))
    # Approximate std from (p90 - p10) / 2.56 (normal quantile spread)
    demand_std = float(np.mean((p90 - p10) / 2.56))
    demand_std = max(0.0, demand_std)

    # Common MC kwargs
    common_kw: dict[str, Any] = dict(
        forecast_p10=p10,
        forecast_p50=p50,
        forecast_p90=p90,
        initial_stock=initial_stock,
        lead_time_mean=lead_time_mean,
        lead_time_std=lead_time_std,
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        holding_cost_per_unit_day=holding_cost_per_unit_day,
        stockout_cost_per_unit=stockout_cost_per_unit,
        fixed_order_cost=fixed_order_cost,
    )

    # Build policy-specific parameters
    policy_params = [
        _sq_params(mean_daily, demand_std, lead_time_mean, holding_cost_per_unit_day, fixed_order_cost, service_level),
        _ss_params(mean_daily, demand_std, lead_time_mean, horizon_days, service_level),
        _rs_params(mean_daily, demand_std, lead_time_mean, review_period, service_level),
        _sl_driven_params(mean_daily, demand_std, lead_time_mean, service_level),
    ]

    mc_results: list[MonteCarloResult] = []
    for i, (name, params) in enumerate(zip(POLICY_NAMES, policy_params)):
        result = simulate_inventory_mc(
            **common_kw,
            reorder_point=params["reorder_point"],
            order_quantity=params["order_quantity"],
            seed=seed + i,
            series_id=f"{series_id}_{name}",
        )
        mc_results.append(result)
        logger.debug(
            "Policy %s: fill_rate=%.3f, stockout_prob=%.3f, cost=%.2f",
            name, result.fill_rate_mean, result.stockout_probability, result.total_cost_mean,
        )

    # Aggregate
    fill_rates = [r.fill_rate_mean for r in mc_results]
    stockout_probs = [r.stockout_probability for r in mc_results]
    avg_inventories = [r.avg_inventory_mean for r in mc_results]
    total_costs = [r.total_cost_mean for r in mc_results]

    best_idx = int(np.argmin(total_costs))
    best_policy = POLICY_NAMES[best_idx]
    best_cost = total_costs[best_idx]

    return PolicyComparisonResult(
        series_id=series_id,
        policy_names=list(POLICY_NAMES),
        fill_rates=fill_rates,
        stockout_probs=stockout_probs,
        avg_inventories=avg_inventories,
        total_costs=total_costs,
        best_policy=best_policy,
        best_cost=best_cost,
        mc_results=mc_results,
        is_synthetic=True,
        synthetic_tag=SYNTHETIC_TAG,
    )


# ---------------------------------------------------------------------------
# Batch runner + parquet export
# ---------------------------------------------------------------------------


def run_policy_comparison_batch(
    series_list: list[dict],
    n_simulations: int = 1000,
    horizon_days: int = 90,
    output_dir: Path | None = None,
) -> list[PolicyComparisonResult]:
    """Run policy comparison for a batch of SKU-store series.

    Parameters
    ----------
    series_list:
        List of dicts with keys matching ``compare_policies`` parameters
        plus ``series_id``.  Required: ``forecast_p10``, ``forecast_p50``,
        ``forecast_p90``, ``initial_stock``, ``lead_time_mean``,
        ``lead_time_std``.
    n_simulations:
        MC paths per policy per series.
    horizon_days:
        Simulation horizon.
    output_dir:
        If provided, saves ``policy_comparison.parquet`` here.

    Returns
    -------
    list[PolicyComparisonResult]
    """
    t0 = time.perf_counter()
    results: list[PolicyComparisonResult] = []

    for i, s in enumerate(series_list):
        r = compare_policies(
            forecast_p10=s["forecast_p10"],
            forecast_p50=s["forecast_p50"],
            forecast_p90=s["forecast_p90"],
            initial_stock=s.get("initial_stock", 30.0),
            lead_time_mean=s.get("lead_time_mean", 7.0),
            lead_time_std=s.get("lead_time_std", 2.0),
            holding_cost_per_unit_day=s.get("holding_cost_per_unit_day", 0.01),
            stockout_cost_per_unit=s.get("stockout_cost_per_unit", 1.0),
            fixed_order_cost=s.get("fixed_order_cost", 10.0),
            service_level=s.get("service_level", 0.95),
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            seed=42 + i,
            series_id=s.get("series_id", f"series_{i}"),
        )
        results.append(r)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Policy comparison: %d series × 4 policies × %d sims in %.2f s",
        len(series_list), n_simulations, elapsed,
    )

    if output_dir is not None and results:
        _save_parquet(results, Path(output_dir))

    return results


def _save_parquet(results: list[PolicyComparisonResult], output_dir: Path) -> Path:
    """Flatten PolicyComparisonResult list to parquet."""
    try:
        import polars as pl
    except ImportError:
        import pandas as pd
        rows = _flatten_results(results)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "policy_comparison.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        logger.info("Saved policy_comparison.parquet (%d rows)", len(rows))
        return path

    rows = _flatten_results(results)
    df = pl.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "policy_comparison.parquet"
    df.write_parquet(path)
    logger.info("Saved policy_comparison.parquet (%d rows)", len(rows))
    return path


def _flatten_results(results: list[PolicyComparisonResult]) -> list[dict]:
    rows: list[dict] = []
    for r in results:
        for i, name in enumerate(r.policy_names):
            rows.append({
                "series_id": r.series_id,
                "policy": name,
                "fill_rate": r.fill_rates[i] if i < len(r.fill_rates) else None,
                "stockout_prob": r.stockout_probs[i] if i < len(r.stockout_probs) else None,
                "avg_inventory": r.avg_inventories[i] if i < len(r.avg_inventories) else None,
                "total_cost": r.total_costs[i] if i < len(r.total_costs) else None,
                "is_best": name == r.best_policy,
                "synthetic_tag": SYNTHETIC_TAG,
            })
    return rows


def build_policy_comparison_summary(
    results: list[PolicyComparisonResult],
) -> dict:
    """Build a JSON-serialisable summary dict for the serving exporter."""
    if not results:
        return {"policies": POLICY_NAMES, "series": [], "synthetic_tag": SYNTHETIC_TAG}

    series_summaries = []
    for r in results:
        series_summaries.append({
            "series_id": r.series_id,
            "best_policy": r.best_policy,
            "best_cost": round(r.best_cost, 4),
            "policies": {
                name: {
                    "fill_rate": round(r.fill_rates[i], 4),
                    "stockout_prob": round(r.stockout_probs[i], 4),
                    "avg_inventory": round(r.avg_inventories[i], 4),
                    "total_cost": round(r.total_costs[i], 4),
                }
                for i, name in enumerate(r.policy_names)
            },
        })

    # Aggregate best-policy distribution
    from collections import Counter
    best_counts = Counter(r.best_policy for r in results)

    return {
        "n_series": len(results),
        "policies": POLICY_NAMES,
        "best_policy_distribution": dict(best_counts),
        "series": series_summaries,
        "synthetic_tag": SYNTHETIC_TAG,
    }
