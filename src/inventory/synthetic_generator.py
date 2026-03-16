"""Synthetic inventory parameter generator.

Generates per-SKU-store inventory parameters derived from observed demand
patterns (spec section 3.4).  All outputs are **explicitly tagged as
SYNTHETIC** in every output column and file.

Parameter generation rules (spec section 3.4):
  initial_stock_on_hand = round(avg_demand_28d × 30)
  lead_time_days        = LogNormal(μ=ln(cat_mu), σ=cat_sigma)
  service_level_target  = {A:0.98, B:0.95, C:0.90}[abc_class]
  holding_cost_pct      = {FOODS:0.25, HOUSEHOLD:0.20, HOBBIES:0.15}[cat]
  stockout_cost_mult    = {A:5.0, B:3.0, C:1.5}[abc_class]

Seed is fixed at 42 for full reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Parameter tables (from spec section 3.4)
# ---------------------------------------------------------------------------

SYNTHETIC_TAG: str = "SYNTHETIC"
SYNTHETIC_GENERATOR_VERSION: str = "1.0"

# LogNormal(μ, σ) for lead_time_days by category
# mean LT ≈ exp(μ + σ²/2)
LEAD_TIME_PARAMS: dict[str, tuple[float, float]] = {
    "FOODS":     (np.log(7),  0.30),   # E[LT] ≈ 7.3 days
    "HOUSEHOLD": (np.log(10), 0.35),   # E[LT] ≈ 10.6 days
    "HOBBIES":   (np.log(14), 0.40),   # E[LT] ≈ 15.0 days
}

# Fallback when category not in table
_DEFAULT_LT_PARAMS: tuple[float, float] = (np.log(7), 0.30)

# Service level target by ABC class
SERVICE_LEVEL_BY_ABC: dict[str, float] = {
    "A": 0.98,
    "B": 0.95,
    "C": 0.90,
}

# Annual holding cost as % of unit value by category
HOLDING_COST_BY_CAT: dict[str, float] = {
    "FOODS":     0.25,
    "HOUSEHOLD": 0.20,
    "HOBBIES":   0.15,
}

# Stockout cost multiplier by ABC class
STOCKOUT_COST_BY_ABC: dict[str, float] = {
    "A": 5.0,
    "B": 3.0,
    "C": 1.5,
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_synthetic_params(
    classification_df: pl.DataFrame,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic inventory parameters for every SKU-store.

    Parameters
    ----------
    classification_df:
        Output of ``classify_all_series`` (or equivalent) enriched with
        ABC/XYZ classification.  Required columns: id, cat_id, abc_class,
        avg_daily_demand (and optionally item_id, store_id, dept_id).
    seed:
        Random seed for reproducibility (default 42, per spec).

    Returns
    -------
    pl.DataFrame
        One row per series with columns:
        id, item_id, store_id, cat_id, abc_class,
        initial_stock_on_hand, lead_time_days, lead_time_mean,
        service_level_target, holding_cost_pct, stockout_cost_multiplier,
        is_synthetic (literal True), synthetic_tag (literal "SYNTHETIC").
    """
    rng = np.random.default_rng(seed)

    rows: list[dict] = []

    for row in classification_df.iter_rows(named=True):
        cat = str(row.get("cat_id", "FOODS")).upper()
        # Normalize: take first word if dept-style value (e.g., "FOODS_1" → "FOODS")
        cat_key = cat.split("_")[0] if "_" in cat else cat
        abc = str(row.get("abc_class", "C")).upper()
        avg_daily = float(row.get("avg_daily_demand", 1.0) or 1.0)

        # Lead time from category-specific LogNormal
        mu, sigma = LEAD_TIME_PARAMS.get(cat_key, _DEFAULT_LT_PARAMS)
        lead_time_sample = float(rng.lognormal(mu, sigma))
        lead_time = max(1.0, lead_time_sample)
        # Deterministic mean for V1 (no Monte Carlo — that's V2)
        lead_time_mean = np.exp(mu + 0.5 * sigma ** 2)

        rows.append({
            "id": row.get("id", ""),
            "item_id": row.get("item_id", ""),
            "store_id": row.get("store_id", ""),
            "cat_id": cat,
            "abc_class": abc,
            # Derived parameters
            "initial_stock_on_hand": max(1, round(avg_daily * 30)),
            "lead_time_days": round(lead_time, 2),
            "lead_time_mean": round(lead_time_mean, 2),
            "service_level_target": SERVICE_LEVEL_BY_ABC.get(abc, 0.90),
            "holding_cost_pct": HOLDING_COST_BY_CAT.get(cat_key, 0.20),
            "stockout_cost_multiplier": STOCKOUT_COST_BY_ABC.get(abc, 1.5),
            # Synthetic metadata (spec rule: ALWAYS label)
            "is_synthetic": True,
            "synthetic_tag": SYNTHETIC_TAG,
        })

    return pl.DataFrame(rows)


def save_synthetic_params(
    params_df: pl.DataFrame,
    output_dir: Path,
    write_metadata: bool = True,
) -> Path:
    """Persist synthetic parameters to Parquet + optional metadata JSON.

    Parameters
    ----------
    params_df:
        Output of ``generate_synthetic_params``.
    output_dir:
        Destination directory (created if needed).
    write_metadata:
        If True, write a ``synthetic_params_metadata.json`` alongside the
        Parquet file documenting the generation logic.

    Returns
    -------
    Path to the written Parquet file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "synthetic_params.parquet"
    params_df.write_parquet(parquet_path, compression="snappy")

    if write_metadata:
        meta = {
            "synthetic_tag": SYNTHETIC_TAG,
            "generator_version": SYNTHETIC_GENERATOR_VERSION,
            "description": (
                "Inventory parameters derived from observed demand statistics. "
                "All values are SYNTHETIC — not actual supplier/inventory records."
            ),
            "parameter_sources": {
                "initial_stock_on_hand": "avg_demand_28d × 30 (rounded)",
                "lead_time_days": "LogNormal by category (FOODS μ=ln(7), HOUSEHOLD μ=ln(10), HOBBIES μ=ln(14))",
                "service_level_target": "A=0.98, B=0.95, C=0.90 (ABC classification)",
                "holding_cost_pct": "FOODS=0.25, HOUSEHOLD=0.20, HOBBIES=0.15",
                "stockout_cost_multiplier": "A=5.0, B=3.0, C=1.5",
            },
            "n_records": len(params_df),
        }
        meta_path = output_dir / "synthetic_params_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    return parquet_path
