"""Monte Carlo inventory simulator — V2-Fase 3.

Replaces the deterministic engine from V1 with a fully stochastic simulation:
  - 1,000 paths × 90 days per SKU-store (spec §10.2)
  - Demand sampled from piecewise-linear CDF defined by (p10, p50, p90)
  - Lead time sampled per order from LogNormal(μ, σ)
  - Policy: (s, Q) by default — single pending order allowed at a time
  - Full NumPy vectorisation: outer loop over 90 days, inner dimension = n_sims

Performance notes
-----------------
For 500 sample series × 1,000 sims × 90 days:
  - Demand pre-sampling: vectorised O(n_sims × horizon) per series.
  - Inventory simulation: 90 iterations × O(n_sims) NumPy ops.
  - joblib optional parallelism across series.

All outputs are tagged SYNTHETIC (spec rule 3.4).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SYNTHETIC_TAG = "SYNTHETIC"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloResult:
    """Aggregated Monte Carlo simulation results for one SKU-store."""

    series_id: str = ""

    # Fill-rate statistics
    fill_rate_mean: float = 0.0
    fill_rate_p5: float = 0.0
    fill_rate_p95: float = 0.0
    fill_rate_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    # Stockout statistics
    stockout_probability: float = 0.0   # fraction of sims with ≥ 1 stockout day
    expected_stockout_days: float = 0.0 # mean across sims

    # Inventory statistics
    avg_inventory_mean: float = 0.0

    # Cost statistics
    total_cost_mean: float = 0.0
    total_cost_p5: float = 0.0
    total_cost_p95: float = 0.0
    total_cost_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    # Daily percentiles for fan chart (shape = 5 × horizon_days)
    # rows: p5, p25, p50, p75, p95
    daily_stock_percentiles: np.ndarray = field(default_factory=lambda: np.zeros((5, 90)))

    # Metadata
    n_simulations: int = 0
    horizon_days: int = 90
    is_synthetic: bool = True
    synthetic_tag: str = SYNTHETIC_TAG


# ---------------------------------------------------------------------------
# Demand sampling helpers
# ---------------------------------------------------------------------------


def _sample_demands(
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample demand matrix from piecewise-linear quantile distribution.

    Returns shape ``(n_sims, len(p50))``.  Demand is clipped at 0.

    The CDF is defined by three quantile points:
      P[D ≤ p10] = 0.10,  P[D ≤ p50] = 0.50,  P[D ≤ p90] = 0.90
    with linear interpolation between segments and linear extrapolation
    into the tails (floored at 0 for the lower tail).
    """
    horizon = len(p50)
    u = rng.uniform(0.0, 1.0, (n_sims, horizon))
    demands = np.empty((n_sims, horizon), dtype=float)

    # Pre-compute upper tail anchor: linear extrapolation past p90
    ub = np.maximum(2.0 * p90 - p50, p90 * 1.1)

    for t in range(horizon):
        # Five anchor points for interpolation
        probs = np.array([0.0, 0.10, 0.50, 0.90, 1.0])
        vals = np.array([
            0.0,
            max(0.0, p10[t]),
            max(0.0, p50[t]),
            max(0.0, p90[t]),
            max(0.0, ub[t]),
        ])
        demands[:, t] = np.interp(u[:, t], probs, vals)

    return np.maximum(0.0, demands)


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------


def simulate_inventory_mc(
    forecast_p10: np.ndarray,
    forecast_p50: np.ndarray,
    forecast_p90: np.ndarray,
    initial_stock: float,
    lead_time_mean: float,
    lead_time_std: float,
    reorder_point: float,
    order_quantity: float,
    n_simulations: int = 1000,
    horizon_days: int = 90,
    seed: int = 42,
    holding_cost_per_unit_day: float = 0.01,
    stockout_cost_per_unit: float = 1.0,
    fixed_order_cost: float = 10.0,
    series_id: str = "",
) -> MonteCarloResult:
    """Run Monte Carlo inventory simulation for one SKU-store.

    Parameters
    ----------
    forecast_p10, forecast_p50, forecast_p90:
        Quantile forecasts for each day of the horizon.  Arrays of length
        ``horizon_days``.
    initial_stock:
        Starting stock-on-hand.
    lead_time_mean, lead_time_std:
        Parameters for the LogNormal lead-time distribution.
        LogNormal is parameterised by (μ, σ) of the underlying normal:
        μ = log(lt_mean²  / √(lt_mean² + lt_std²))
        σ = √(log(1 + lt_std² / lt_mean²))
    reorder_point:
        Trigger level s.  Order placed when stock ≤ s.
    order_quantity:
        Fixed quantity Q per replenishment order.
    n_simulations:
        Number of Monte Carlo paths.
    horizon_days:
        Simulation horizon (default 90 days).
    seed:
        Random seed for reproducibility.
    holding_cost_per_unit_day:
        Cost of holding one unit for one day.
    stockout_cost_per_unit:
        Cost of one unit of lost sales.
    fixed_order_cost:
        Fixed cost per order placed.
    series_id:
        Identifier stored in the result.

    Returns
    -------
    MonteCarloResult
    """
    # ── Input validation ────────────────────────────────────────────────────
    p10 = np.asarray(forecast_p10, dtype=float)[:horizon_days]
    p50 = np.asarray(forecast_p50, dtype=float)[:horizon_days]
    p90 = np.asarray(forecast_p90, dtype=float)[:horizon_days]

    # Pad if shorter than horizon
    if len(p50) < horizon_days:
        fill = p50[-1] if len(p50) > 0 else 0.0
        p10 = np.pad(p10, (0, horizon_days - len(p10)), constant_values=max(0, fill * 0.5))
        p50 = np.pad(p50, (0, horizon_days - len(p50)), constant_values=fill)
        p90 = np.pad(p90, (0, horizon_days - len(p90)), constant_values=max(fill, fill * 1.5))

    order_quantity = max(1.0, float(order_quantity))
    reorder_point = float(reorder_point)
    initial_stock = max(0.0, float(initial_stock))

    # ── LogNormal lead-time params ──────────────────────────────────────────
    lt_mean = max(1.0, float(lead_time_mean))
    lt_std = max(0.0, float(lead_time_std))

    if lt_std > 0:
        variance = lt_std ** 2
        sigma = np.sqrt(np.log(1 + variance / lt_mean ** 2))
        mu = np.log(lt_mean) - sigma ** 2 / 2
    else:
        mu, sigma = np.log(lt_mean), 1e-6

    # ── Random number generation ────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    # Pre-sample demands: shape (n_sims, horizon_days)
    demands = _sample_demands(p10, p50, p90, n_simulations, rng)

    # Pre-sample lead times pool: each sim may place up to max_orders orders
    max_orders = max(5, horizon_days // max(1, int(lt_mean)) + 5)
    raw_lt = rng.lognormal(mu, sigma, (n_simulations, max_orders))
    lead_times = np.maximum(1, np.round(raw_lt).astype(int))

    # ── Simulation state ────────────────────────────────────────────────────
    n = n_simulations

    stock = np.full(n, initial_stock, dtype=float)
    pending_qty = np.zeros(n, dtype=float)   # qty of in-flight order (0 = none)
    pending_arr = np.full(n, -1, dtype=int)  # arrival day (-1 = no order)
    order_cnt = np.zeros(n, dtype=int)       # which LT sample to use next

    # Accumulators (shape = n_sims)
    total_demand = np.zeros(n)
    total_fulfilled = np.zeros(n)
    total_lost = np.zeros(n)
    stockout_days = np.zeros(n, dtype=int)
    orders_placed = np.zeros(n, dtype=int)
    daily_stock_sum = np.zeros(n)

    # Daily stock tracking for percentiles: shape (n_sims, horizon_days)
    daily_stock_all = np.empty((n, horizon_days), dtype=float)

    # ── Day-by-day simulation (vectorised across sims) ──────────────────────
    for day in range(horizon_days):
        # Receive orders arriving today
        arrived = pending_arr == day
        stock[arrived] += pending_qty[arrived]
        pending_arr[arrived] = -1
        pending_qty[arrived] = 0.0

        # Consume demand
        demand = demands[:, day]
        total_demand += demand
        fulfilled = np.minimum(stock, demand)
        lost = demand - fulfilled
        stock -= fulfilled

        total_fulfilled += fulfilled
        total_lost += lost
        stockout_days += (lost > 0).astype(int)

        # Record stock level
        daily_stock_all[:, day] = np.maximum(0.0, stock)
        daily_stock_sum += np.maximum(0.0, stock)

        # Reorder trigger: stock ≤ ROP and no order in flight
        can_order = pending_arr < 0
        triggers = can_order & (stock <= reorder_point)

        if triggers.any():
            idx = np.where(triggers)[0]
            # Sample lead time from pre-sampled pool
            cnt = order_cnt[idx]
            cnt_clipped = np.minimum(cnt, max_orders - 1)
            lt = lead_times[idx, cnt_clipped]

            pending_arr[idx] = day + lt
            pending_qty[idx] = order_quantity
            order_cnt[idx] = np.minimum(cnt + 1, max_orders - 1)
            orders_placed[idx] += 1

    # ── Compute aggregate metrics ───────────────────────────────────────────
    fill_rates = np.where(
        total_demand > 0,
        total_fulfilled / total_demand,
        1.0,
    )

    avg_inv = daily_stock_sum / horizon_days

    # Cost per simulation
    holding_cost = avg_inv * horizon_days * holding_cost_per_unit_day
    stockout_cost_total = total_lost * stockout_cost_per_unit
    order_cost_total = orders_placed * fixed_order_cost
    total_cost = holding_cost + stockout_cost_total + order_cost_total

    # Daily stock percentiles (5, horizon_days)
    pcts = np.percentile(daily_stock_all, [5, 25, 50, 75, 95], axis=0)

    return MonteCarloResult(
        series_id=series_id,
        fill_rate_mean=float(np.mean(fill_rates)),
        fill_rate_p5=float(np.percentile(fill_rates, 5)),
        fill_rate_p95=float(np.percentile(fill_rates, 95)),
        fill_rate_distribution=fill_rates,
        stockout_probability=float(np.mean(stockout_days > 0)),
        expected_stockout_days=float(np.mean(stockout_days)),
        avg_inventory_mean=float(np.mean(avg_inv)),
        total_cost_mean=float(np.mean(total_cost)),
        total_cost_p5=float(np.percentile(total_cost, 5)),
        total_cost_p95=float(np.percentile(total_cost, 95)),
        total_cost_distribution=total_cost,
        daily_stock_percentiles=pcts,
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        is_synthetic=True,
        synthetic_tag=SYNTHETIC_TAG,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_mc_batch(
    series_list: list[dict],
    n_simulations: int = 1000,
    horizon_days: int = 90,
    n_jobs: int = 1,
) -> list[MonteCarloResult]:
    """Run Monte Carlo simulation for a batch of SKU-store series.

    Parameters
    ----------
    series_list:
        List of dicts, each with keys matching ``simulate_inventory_mc``
        parameters plus ``series_id``.  Required keys:
        ``forecast_p10``, ``forecast_p50``, ``forecast_p90``,
        ``initial_stock``, ``lead_time_mean``, ``lead_time_std``,
        ``reorder_point``, ``order_quantity``.
    n_simulations:
        Paths per series.
    horizon_days:
        Simulation horizon.
    n_jobs:
        Parallel jobs (requires joblib).  -1 = all CPUs.

    Returns
    -------
    list[MonteCarloResult] in the same order as input.
    """
    def _run_one(s: dict, sim_idx: int) -> MonteCarloResult:
        return simulate_inventory_mc(
            forecast_p10=s["forecast_p10"],
            forecast_p50=s["forecast_p50"],
            forecast_p90=s["forecast_p90"],
            initial_stock=s.get("initial_stock", 30.0),
            lead_time_mean=s.get("lead_time_mean", 7.0),
            lead_time_std=s.get("lead_time_std", 2.0),
            reorder_point=s.get("reorder_point", 10.0),
            order_quantity=s.get("order_quantity", 50.0),
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            seed=42 + sim_idx,
            holding_cost_per_unit_day=s.get("holding_cost_per_unit_day", 0.01),
            stockout_cost_per_unit=s.get("stockout_cost_per_unit", 1.0),
            fixed_order_cost=s.get("fixed_order_cost", 10.0),
            series_id=s.get("series_id", f"series_{sim_idx}"),
        )

    if n_jobs != 1 and len(series_list) > 1:
        try:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run_one)(s, i) for i, s in enumerate(series_list)
            )
            return results
        except ImportError:
            logger.warning("joblib not available — falling back to sequential.")

    t0 = time.perf_counter()
    results = [_run_one(s, i) for i, s in enumerate(series_list)]
    elapsed = time.perf_counter() - t0
    logger.info(
        "MC batch: %d series × %d sims × %d days in %.2f s (%.1f ms/series)",
        len(series_list), n_simulations, horizon_days,
        elapsed, 1000 * elapsed / max(1, len(series_list)),
    )
    return results
