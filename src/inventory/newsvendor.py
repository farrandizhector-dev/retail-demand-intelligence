"""Newsvendor model and inventory policy helpers — V2-Fase 3.

Implements:
- Newsvendor optimal quantity: Q* = F⁻¹(Cu / (Cu + Co))
- EOQ: Economic Order Quantity = √(2DS/H)
- Safety stock: SS = z(SL) × σ_demand × √(lead_time)
- Reorder point: ROP = mean_demand_during_LT + SS
- Comparison across ABC segments

All outputs tagged SYNTHETIC (spec rule 3.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

SYNTHETIC_TAG = "SYNTHETIC"

AbcSegment = Literal["A", "B", "C"]

# Default service levels per ABC segment (spec §10.1)
ABC_SERVICE_LEVELS: dict[str, float] = {
    "A": 0.95,
    "B": 0.90,
    "C": 0.85,
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class NewsvendorResult:
    """Optimal quantity and cost comparison for a single SKU-period."""

    series_id: str = ""

    # Newsvendor
    optimal_quantity: float = 0.0
    critical_ratio: float = 0.0       # Cu / (Cu + Co)
    cu: float = 0.0                   # underage cost per unit
    co: float = 0.0                   # overage cost per unit

    # EOQ
    eoq: float = 0.0
    annual_demand: float = 0.0
    order_cost: float = 0.0
    holding_cost_annual: float = 0.0

    # Safety stock & ROP
    safety_stock: float = 0.0
    reorder_point: float = 0.0
    service_level: float = 0.0
    demand_std: float = 0.0
    lead_time: float = 0.0

    # Metadata
    is_synthetic: bool = True
    synthetic_tag: str = SYNTHETIC_TAG


# ---------------------------------------------------------------------------
# Core newsvendor
# ---------------------------------------------------------------------------


def optimal_newsvendor_quantity(
    forecast_quantiles: np.ndarray,
    cu: float,
    co: float,
) -> float:
    """Compute newsvendor optimal order quantity Q*.

    Q* = F⁻¹(Cu / (Cu + Co))

    where F is the empirical CDF of the forecast distribution,
    estimated from the provided quantile samples.

    Parameters
    ----------
    forecast_quantiles:
        Array of demand samples or quantile values representing the
        demand distribution (e.g., MC simulated demands for one period).
    cu:
        Underage (stockout) cost per unit.  Must be > 0.
    co:
        Overage (holding/waste) cost per unit.  Must be > 0.

    Returns
    -------
    float
        Optimal order quantity Q*.  Clipped at 0.
    """
    if cu <= 0 or co <= 0:
        raise ValueError(f"cu and co must be positive, got cu={cu}, co={co}")

    critical_ratio = cu / (cu + co)

    q_star = float(np.quantile(forecast_quantiles, critical_ratio))
    return max(0.0, q_star)


def compute_critical_ratio(cu: float, co: float) -> float:
    """Return Cu / (Cu + Co), the newsvendor critical ratio."""
    if cu <= 0 or co <= 0:
        raise ValueError(f"cu and co must be positive, got cu={cu}, co={co}")
    return cu / (cu + co)


# ---------------------------------------------------------------------------
# EOQ
# ---------------------------------------------------------------------------


def economic_order_quantity(
    annual_demand: float,
    order_cost: float,
    holding_cost_annual: float,
) -> float:
    """Classic EOQ = √(2 × D × S / H).

    Parameters
    ----------
    annual_demand:
        Expected annual demand (units/year).
    order_cost:
        Fixed cost per order placed ($).
    holding_cost_annual:
        Annual holding cost per unit ($/unit/year).

    Returns
    -------
    float
        EOQ in units.  Returns 1.0 if any input is ≤ 0.
    """
    if annual_demand <= 0 or order_cost <= 0 or holding_cost_annual <= 0:
        return 1.0
    return float(np.sqrt(2 * annual_demand * order_cost / holding_cost_annual))


# ---------------------------------------------------------------------------
# Safety stock & ROP
# ---------------------------------------------------------------------------


def safety_stock(
    demand_std: float,
    lead_time: float,
    service_level: float = 0.95,
) -> float:
    """Safety stock = z(SL) × σ_demand × √(LT).

    Parameters
    ----------
    demand_std:
        Standard deviation of daily demand.
    lead_time:
        Lead time in days.
    service_level:
        Desired service level (0–1).

    Returns
    -------
    float
        Safety stock in units (non-negative).
    """
    if demand_std < 0:
        raise ValueError(f"demand_std must be >= 0, got {demand_std}")
    if lead_time < 0:
        raise ValueError(f"lead_time must be >= 0, got {lead_time}")
    service_level = max(0.0, min(1.0 - 1e-9, service_level))

    z = float(stats.norm.ppf(service_level))
    ss = z * demand_std * np.sqrt(max(0.0, lead_time))
    return max(0.0, ss)


def reorder_point(
    mean_daily_demand: float,
    lead_time: float,
    demand_std: float = 0.0,
    service_level: float = 0.95,
) -> float:
    """ROP = mean_demand_during_LT + safety_stock.

    Parameters
    ----------
    mean_daily_demand:
        Average daily demand (units/day).
    lead_time:
        Lead time in days.
    demand_std:
        Std dev of daily demand for safety stock calculation.
    service_level:
        Target service level.

    Returns
    -------
    float
        Reorder point (non-negative).
    """
    demand_during_lt = max(0.0, mean_daily_demand) * max(0.0, lead_time)
    ss = safety_stock(demand_std, lead_time, service_level)
    return max(0.0, demand_during_lt + ss)


# ---------------------------------------------------------------------------
# Full newsvendor analysis
# ---------------------------------------------------------------------------


def run_newsvendor_analysis(
    forecast_samples: np.ndarray,
    cu: float,
    co: float,
    annual_demand: float | None = None,
    order_cost: float = 10.0,
    holding_cost_annual: float = 3.65,  # ≈ 0.01/day × 365
    lead_time: float = 7.0,
    demand_std: float | None = None,
    service_level: float = 0.95,
    series_id: str = "",
) -> NewsvendorResult:
    """Full newsvendor + EOQ + safety-stock analysis for one SKU.

    Parameters
    ----------
    forecast_samples:
        Demand samples representing the distribution for one period
        (e.g., 1,000 MC demand values).
    cu:
        Underage cost per unit.
    co:
        Overage cost per unit.
    annual_demand:
        Annual demand for EOQ.  Defaults to mean(forecast_samples) × 365.
    order_cost:
        Fixed order cost for EOQ.
    holding_cost_annual:
        Annual holding cost per unit for EOQ.
    lead_time:
        Lead time in days for safety stock / ROP.
    demand_std:
        Demand std dev for safety stock.  Defaults to std(forecast_samples).
    service_level:
        Service level for safety stock.
    series_id:
        Identifier stored in the result.
    """
    forecast_samples = np.asarray(forecast_samples, dtype=float)
    if len(forecast_samples) == 0:
        raise ValueError("forecast_samples must be non-empty")

    q_star = optimal_newsvendor_quantity(forecast_samples, cu, co)
    cr = compute_critical_ratio(cu, co)

    if annual_demand is None:
        annual_demand = float(np.mean(forecast_samples)) * 365.0
    annual_demand = max(0.0, annual_demand)

    eoq = economic_order_quantity(annual_demand, order_cost, holding_cost_annual)

    if demand_std is None:
        demand_std = float(np.std(forecast_samples))

    ss = safety_stock(demand_std, lead_time, service_level)
    mean_daily = float(np.mean(forecast_samples))
    rop = reorder_point(mean_daily, lead_time, demand_std, service_level)

    return NewsvendorResult(
        series_id=series_id,
        optimal_quantity=q_star,
        critical_ratio=cr,
        cu=cu,
        co=co,
        eoq=eoq,
        annual_demand=annual_demand,
        order_cost=order_cost,
        holding_cost_annual=holding_cost_annual,
        safety_stock=ss,
        reorder_point=rop,
        service_level=service_level,
        demand_std=demand_std,
        lead_time=lead_time,
        is_synthetic=True,
        synthetic_tag=SYNTHETIC_TAG,
    )


# ---------------------------------------------------------------------------
# ABC segment comparison
# ---------------------------------------------------------------------------


def compare_by_abc_segment(
    forecast_samples: np.ndarray,
    cu: float,
    co: float,
    order_cost: float = 10.0,
    holding_cost_annual: float = 3.65,
    lead_time: float = 7.0,
    demand_std: float | None = None,
) -> dict[str, NewsvendorResult]:
    """Run newsvendor analysis at A / B / C service levels.

    Returns a dict keyed by segment letter with a NewsvendorResult for each.
    """
    results: dict[str, NewsvendorResult] = {}
    for seg, sl in ABC_SERVICE_LEVELS.items():
        results[seg] = run_newsvendor_analysis(
            forecast_samples=forecast_samples,
            cu=cu,
            co=co,
            order_cost=order_cost,
            holding_cost_annual=holding_cost_annual,
            lead_time=lead_time,
            demand_std=demand_std,
            service_level=sl,
            series_id=f"segment_{seg}",
        )
    return results
