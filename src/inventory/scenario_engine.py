"""Scenario engine for what-if inventory analysis — V2-Fase 3.

Implements 5 what-if scenarios (spec §10.4):
  1. demand_surge:    Demand +30%
  2. lead_time_delay: Lead time +5 days
  3. cost_increase:   Holding cost +15%
  4. high_service:    Service level 95% → 99%
  5. combined_stress: Demand +20% + Lead time +3 days (combined stress)

Each scenario is evaluated by:
  - Scaling the relevant inputs
  - Running simulate_inventory_mc (same seed as baseline for comparability)
  - Returning delta metrics vs baseline

Output: scenario_results.json (< 200 KB per spec §10.4).
All outputs tagged SYNTHETIC (spec rule 3.4).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.inventory.simulator import MonteCarloResult, simulate_inventory_mc

logger = logging.getLogger(__name__)

SYNTHETIC_TAG = "SYNTHETIC"

# Scenario definitions: name → (description, parameter perturbations)
SCENARIO_DEFINITIONS: dict[str, dict[str, Any]] = {
    "demand_surge": {
        "description": "Demand +30% (supply chain disruption / promotional spike)",
        "demand_scale": 1.30,
        "lead_time_delta": 0.0,
        "holding_cost_scale": 1.0,
        "service_level_target": None,
    },
    "lead_time_delay": {
        "description": "Lead time +5 days (supplier delay / port congestion)",
        "demand_scale": 1.0,
        "lead_time_delta": 5.0,
        "holding_cost_scale": 1.0,
        "service_level_target": None,
    },
    "cost_increase": {
        "description": "Holding cost +15% (storage rate increase)",
        "demand_scale": 1.0,
        "lead_time_delta": 0.0,
        "holding_cost_scale": 1.15,
        "service_level_target": None,
    },
    "high_service": {
        "description": "Service level target 95% → 99% (premium customer SLA)",
        "demand_scale": 1.0,
        "lead_time_delta": 0.0,
        "holding_cost_scale": 1.0,
        "service_level_target": 0.99,
    },
    "combined_stress": {
        "description": "Demand +20% + Lead time +3 days (combined stress test)",
        "demand_scale": 1.20,
        "lead_time_delta": 3.0,
        "holding_cost_scale": 1.0,
        "service_level_target": None,
    },
}

SCENARIO_NAMES = list(SCENARIO_DEFINITIONS.keys())


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Single scenario result for one SKU-store."""

    scenario_name: str = ""
    description: str = ""
    series_id: str = ""

    # Absolute metrics
    fill_rate_mean: float = 0.0
    stockout_probability: float = 0.0
    expected_stockout_days: float = 0.0
    avg_inventory_mean: float = 0.0
    total_cost_mean: float = 0.0

    # Delta vs baseline
    delta_fill_rate: float = 0.0       # scenario - baseline
    delta_stockout_prob: float = 0.0
    delta_stockout_days: float = 0.0
    delta_avg_inventory: float = 0.0
    delta_total_cost: float = 0.0

    is_synthetic: bool = True
    synthetic_tag: str = SYNTHETIC_TAG


@dataclass
class ScenarioEngineResult:
    """Full scenario engine output for one SKU-store."""

    series_id: str = ""

    # Baseline result
    baseline_fill_rate: float = 0.0
    baseline_stockout_prob: float = 0.0
    baseline_stockout_days: float = 0.0
    baseline_avg_inventory: float = 0.0
    baseline_total_cost: float = 0.0

    # 5 scenario results
    scenarios: list[ScenarioResult] = field(default_factory=list)

    # Worst scenario by total cost
    worst_scenario: str = ""
    worst_cost_delta: float = 0.0

    is_synthetic: bool = True
    synthetic_tag: str = SYNTHETIC_TAG


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------


def run_scenario_engine(
    forecast_p10: np.ndarray,
    forecast_p50: np.ndarray,
    forecast_p90: np.ndarray,
    initial_stock: float,
    lead_time_mean: float,
    lead_time_std: float,
    reorder_point_val: float,
    order_quantity: float,
    n_simulations: int = 1000,
    horizon_days: int = 90,
    seed: int = 42,
    holding_cost_per_unit_day: float = 0.01,
    stockout_cost_per_unit: float = 1.0,
    fixed_order_cost: float = 10.0,
    series_id: str = "",
) -> ScenarioEngineResult:
    """Run baseline + 5 what-if scenarios for one SKU-store.

    Parameters
    ----------
    forecast_p10, forecast_p50, forecast_p90:
        Quantile forecasts for each day of the horizon.
    initial_stock:
        Starting inventory level.
    lead_time_mean, lead_time_std:
        Lead-time distribution parameters.
    reorder_point_val:
        Reorder point s for (s, Q) policy.
    order_quantity:
        Order quantity Q for (s, Q) policy.
    n_simulations:
        MC paths per scenario.
    horizon_days:
        Simulation horizon.
    seed:
        Random seed (same across all scenarios for fair comparison).
    holding_cost_per_unit_day:
        Holding cost for baseline.
    stockout_cost_per_unit:
        Stockout penalty per unit.
    fixed_order_cost:
        Fixed cost per order.
    series_id:
        Identifier.

    Returns
    -------
    ScenarioEngineResult
    """
    p10 = np.asarray(forecast_p10, dtype=float)
    p50 = np.asarray(forecast_p50, dtype=float)
    p90 = np.asarray(forecast_p90, dtype=float)

    # ── Baseline ───────────────────────────────────────────────────────────
    baseline = simulate_inventory_mc(
        forecast_p10=p10, forecast_p50=p50, forecast_p90=p90,
        initial_stock=initial_stock,
        lead_time_mean=lead_time_mean,
        lead_time_std=lead_time_std,
        reorder_point=reorder_point_val,
        order_quantity=order_quantity,
        n_simulations=n_simulations,
        horizon_days=horizon_days,
        seed=seed,
        holding_cost_per_unit_day=holding_cost_per_unit_day,
        stockout_cost_per_unit=stockout_cost_per_unit,
        fixed_order_cost=fixed_order_cost,
        series_id=f"{series_id}_baseline",
    )

    # ── Scenarios ──────────────────────────────────────────────────────────
    scenario_results: list[ScenarioResult] = []

    for sc_name, sc_def in SCENARIO_DEFINITIONS.items():
        # Scale forecasts
        ds = float(sc_def["demand_scale"])
        sc_p10 = p10 * ds
        sc_p50 = p50 * ds
        sc_p90 = p90 * ds

        # Adjust lead time
        lt_delta = float(sc_def["lead_time_delta"])
        sc_lt_mean = max(1.0, lead_time_mean + lt_delta)
        sc_lt_std = lead_time_std  # std unchanged

        # Adjust holding cost
        hc_scale = float(sc_def["holding_cost_scale"])
        sc_hc = holding_cost_per_unit_day * hc_scale

        # Adjust reorder point for service level change
        sc_rop = reorder_point_val
        if sc_def.get("service_level_target") is not None:
            # Scale ROP proportionally via z-score ratio
            # (simple approximation: scale by service_level_target / 0.95)
            old_sl = 0.95
            new_sl = float(sc_def["service_level_target"])
            from scipy import stats
            z_old = stats.norm.ppf(old_sl)
            z_new = stats.norm.ppf(new_sl)
            # ROP adjustment: add (z_new - z_old) × σ × √LT
            demand_std = float(np.mean((p90 - p10) / 2.56))
            sc_rop = max(0.0, reorder_point_val + (z_new - z_old) * demand_std * np.sqrt(sc_lt_mean))

        sc_mc = simulate_inventory_mc(
            forecast_p10=sc_p10, forecast_p50=sc_p50, forecast_p90=sc_p90,
            initial_stock=initial_stock,
            lead_time_mean=sc_lt_mean,
            lead_time_std=sc_lt_std,
            reorder_point=sc_rop,
            order_quantity=order_quantity,
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            seed=seed,  # same seed for fair comparison
            holding_cost_per_unit_day=sc_hc,
            stockout_cost_per_unit=stockout_cost_per_unit,
            fixed_order_cost=fixed_order_cost,
            series_id=f"{series_id}_{sc_name}",
        )

        sr = ScenarioResult(
            scenario_name=sc_name,
            description=sc_def["description"],
            series_id=series_id,
            fill_rate_mean=sc_mc.fill_rate_mean,
            stockout_probability=sc_mc.stockout_probability,
            expected_stockout_days=sc_mc.expected_stockout_days,
            avg_inventory_mean=sc_mc.avg_inventory_mean,
            total_cost_mean=sc_mc.total_cost_mean,
            delta_fill_rate=sc_mc.fill_rate_mean - baseline.fill_rate_mean,
            delta_stockout_prob=sc_mc.stockout_probability - baseline.stockout_probability,
            delta_stockout_days=sc_mc.expected_stockout_days - baseline.expected_stockout_days,
            delta_avg_inventory=sc_mc.avg_inventory_mean - baseline.avg_inventory_mean,
            delta_total_cost=sc_mc.total_cost_mean - baseline.total_cost_mean,
            is_synthetic=True,
            synthetic_tag=SYNTHETIC_TAG,
        )
        scenario_results.append(sr)

    # Worst scenario by cost delta
    worst_idx = int(np.argmax([s.delta_total_cost for s in scenario_results]))
    worst_sc = scenario_results[worst_idx]

    return ScenarioEngineResult(
        series_id=series_id,
        baseline_fill_rate=baseline.fill_rate_mean,
        baseline_stockout_prob=baseline.stockout_probability,
        baseline_stockout_days=baseline.expected_stockout_days,
        baseline_avg_inventory=baseline.avg_inventory_mean,
        baseline_total_cost=baseline.total_cost_mean,
        scenarios=scenario_results,
        worst_scenario=worst_sc.scenario_name,
        worst_cost_delta=worst_sc.delta_total_cost,
        is_synthetic=True,
        synthetic_tag=SYNTHETIC_TAG,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_scenario_batch(
    series_list: list[dict],
    n_simulations: int = 1000,
    horizon_days: int = 90,
) -> list[ScenarioEngineResult]:
    """Run scenario engine for a batch of SKU-store series."""
    results: list[ScenarioEngineResult] = []
    for i, s in enumerate(series_list):
        r = run_scenario_engine(
            forecast_p10=s["forecast_p10"],
            forecast_p50=s["forecast_p50"],
            forecast_p90=s["forecast_p90"],
            initial_stock=s.get("initial_stock", 30.0),
            lead_time_mean=s.get("lead_time_mean", 7.0),
            lead_time_std=s.get("lead_time_std", 2.0),
            reorder_point_val=s.get("reorder_point", 10.0),
            order_quantity=s.get("order_quantity", 50.0),
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            seed=42 + i,
            holding_cost_per_unit_day=s.get("holding_cost_per_unit_day", 0.01),
            stockout_cost_per_unit=s.get("stockout_cost_per_unit", 1.0),
            fixed_order_cost=s.get("fixed_order_cost", 10.0),
            series_id=s.get("series_id", f"series_{i}"),
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def build_scenario_results_json(results: list[ScenarioEngineResult]) -> dict:
    """Build JSON-serialisable dict for all scenario results.

    Output is designed to be < 200 KB (spec §10.4) by summarising
    per-series data concisely.
    """
    series_list = []
    for r in results:
        scenarios_dict = {}
        for s in r.scenarios:
            scenarios_dict[s.scenario_name] = {
                "description": s.description,
                "fill_rate_mean": round(s.fill_rate_mean, 4),
                "stockout_probability": round(s.stockout_probability, 4),
                "expected_stockout_days": round(s.expected_stockout_days, 3),
                "avg_inventory_mean": round(s.avg_inventory_mean, 3),
                "total_cost_mean": round(s.total_cost_mean, 4),
                "delta_fill_rate": round(s.delta_fill_rate, 4),
                "delta_stockout_prob": round(s.delta_stockout_prob, 4),
                "delta_stockout_days": round(s.delta_stockout_days, 3),
                "delta_total_cost": round(s.delta_total_cost, 4),
            }
        series_list.append({
            "series_id": r.series_id,
            "baseline": {
                "fill_rate_mean": round(r.baseline_fill_rate, 4),
                "stockout_probability": round(r.baseline_stockout_prob, 4),
                "expected_stockout_days": round(r.baseline_stockout_days, 3),
                "avg_inventory_mean": round(r.baseline_avg_inventory, 3),
                "total_cost_mean": round(r.baseline_total_cost, 4),
            },
            "scenarios": scenarios_dict,
            "worst_scenario": r.worst_scenario,
            "worst_cost_delta": round(r.worst_cost_delta, 4),
        })

    return {
        "n_series": len(results),
        "scenario_names": SCENARIO_NAMES,
        "scenario_descriptions": {k: v["description"] for k, v in SCENARIO_DEFINITIONS.items()},
        "series": series_list,
        "synthetic_tag": SYNTHETIC_TAG,
    }


def export_scenario_results(
    results: list[ScenarioEngineResult],
    output_dir: Path,
) -> Path:
    """Write scenario_results.json to output_dir.

    Validates that output is < 200 KB.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_scenario_results_json(results)
    path = output_dir / "scenario_results.json"
    text = json.dumps(payload, indent=2)
    size_kb = len(text.encode()) / 1024

    if size_kb > 200:
        logger.warning(
            "scenario_results.json is %.1f KB, exceeds 200 KB spec limit. "
            "Consider reducing n_series or precision.", size_kb
        )

    with open(path, "w") as f:
        f.write(text)

    logger.info("Saved scenario_results.json (%.1f KB)", size_kb)
    return path
