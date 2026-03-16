"""Data drift detection for demand forecasting pipelines (spec §11.1).

Four checks:
  1. sales_distribution_drift  — KS test on daily sales
  2. feature_distribution_drift — PSI per feature
  3. zero_inflation_shift       — % zero sales per category
  4. price_regime_change        — mean sell_price by dept
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Sales distribution drift (KS test)
# ---------------------------------------------------------------------------

def sales_distribution_drift(
    reference_df: pl.DataFrame,
    current_df: pl.DataFrame,
    date_col: str = "date",
    sales_col: str = "sales",
) -> dict[str, Any]:
    """KS test on daily aggregate sales between reference and current windows.

    Parameters
    ----------
    reference_df:
        DataFrame with at least (date_col, sales_col). Typically last 90 days
        of training data.
    current_df:
        DataFrame with the same schema. Typically last 28 days of production.
    date_col:
        Name of the date column.
    sales_col:
        Name of the sales column.

    Returns
    -------
    dict with keys: ks_statistic, p_value, is_drift, reference_days, current_days.
    """
    from scipy.stats import ks_2samp  # lazy import

    # Aggregate to daily totals
    ref_daily = (
        reference_df.group_by(date_col)
        .agg(pl.col(sales_col).sum().alias("daily_sales"))
        .sort(date_col)
        .get_column("daily_sales")
        .to_numpy()
        .astype(float)
    )
    cur_daily = (
        current_df.group_by(date_col)
        .agg(pl.col(sales_col).sum().alias("daily_sales"))
        .sort(date_col)
        .get_column("daily_sales")
        .to_numpy()
        .astype(float)
    )

    ref_days = int(reference_df[date_col].n_unique())
    cur_days = int(current_df[date_col].n_unique())

    if len(ref_daily) == 0 or len(cur_daily) == 0:
        logger.warning("sales_distribution_drift: empty daily aggregates, returning defaults")
        return {
            "ks_statistic": 0.0,
            "p_value": 1.0,
            "is_drift": False,
            "reference_days": ref_days,
            "current_days": cur_days,
        }

    stat, pval = ks_2samp(ref_daily, cur_daily)

    return {
        "ks_statistic": float(stat),
        "p_value": float(pval),
        "is_drift": bool(pval < 0.01),
        "reference_days": ref_days,
        "current_days": cur_days,
    }


# ---------------------------------------------------------------------------
# 2. PSI computation
# ---------------------------------------------------------------------------

def compute_psi(
    reference_dist: np.ndarray | list[float],
    current_dist: np.ndarray | list[float],
    n_bins: int = 10,
) -> float:
    """Population Stability Index between two distributions.

    Parameters
    ----------
    reference_dist:
        Reference (training) distribution array.
    current_dist:
        Current (production) distribution array.
    n_bins:
        Number of equal-frequency bins derived from reference_dist.

    Returns
    -------
    PSI value (float >= 0). Higher = more drift.
    """
    ref = np.asarray(reference_dist, dtype=float)
    cur = np.asarray(current_dist, dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Build bin edges from reference distribution using equal-frequency binning
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(ref, percentiles))

    # If all values are the same, bin_edges collapses to 1 point → single bin
    if len(bin_edges) < 2:
        return 0.0

    # Clip to avoid inf at boundaries
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    # Convert to percentages, clip to avoid log(0)
    ref_pct = np.clip(ref_counts / len(ref), 0.0001, None)
    cur_pct = np.clip(cur_counts / len(cur), 0.0001, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(psi, 0.0)


# ---------------------------------------------------------------------------
# 3. Feature distribution drift (PSI per feature)
# ---------------------------------------------------------------------------

def feature_distribution_drift(
    ref_features: dict[str, np.ndarray | list[float]],
    cur_features: dict[str, np.ndarray | list[float]],
    feature_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute PSI for each requested feature and assign a status label.

    Parameters
    ----------
    ref_features:
        Mapping of feature_name → array of reference values.
    cur_features:
        Mapping of feature_name → array of current values.
    feature_names:
        List of feature names to evaluate.

    Returns
    -------
    {feature: {"psi": float, "status": str}}
    Status thresholds: <=0.10 → "ok", <=0.20 → "warning", <=0.25 → "alert",
    >0.25 → "retrain".
    """
    results: dict[str, dict[str, Any]] = {}

    for feat in feature_names:
        ref_arr = np.asarray(ref_features.get(feat, []), dtype=float)
        cur_arr = np.asarray(cur_features.get(feat, []), dtype=float)

        psi = compute_psi(ref_arr, cur_arr)

        if psi <= 0.10:
            status = "ok"
        elif psi <= 0.20:
            status = "warning"
        elif psi <= 0.25:
            status = "alert"
        else:
            status = "retrain"

        results[feat] = {"psi": float(psi), "status": status}

    return results


# ---------------------------------------------------------------------------
# 4. Zero-inflation shift
# ---------------------------------------------------------------------------

def zero_inflation_shift(
    ref_df: pl.DataFrame,
    cur_df: pl.DataFrame,
    by: str = "cat_id",
    sales_col: str = "sales",
) -> dict[str, dict[str, Any]]:
    """Detect shift in the fraction of zero sales per category.

    Parameters
    ----------
    ref_df:
        Reference DataFrame with (by, sales_col).
    cur_df:
        Current DataFrame with the same schema.
    by:
        Column used for grouping (e.g. "cat_id").
    sales_col:
        Name of the sales column.

    Returns
    -------
    {category: {"ref_pct": float, "cur_pct": float, "delta": float, "is_shift": bool}}
    Only categories present in BOTH DataFrames are reported.
    """
    def _zero_pct(df: pl.DataFrame) -> dict[str, float]:
        result = (
            df.with_columns(
                (pl.col(sales_col) == 0).cast(pl.Float64).alias("_is_zero")
            )
            .group_by(by)
            .agg(pl.col("_is_zero").mean().alias("pct_zero"))
        )
        return {
            row[by]: float(row["pct_zero"])
            for row in result.iter_rows(named=True)
        }

    ref_pcts = _zero_pct(ref_df)
    cur_pcts = _zero_pct(cur_df)

    common_cats = set(ref_pcts.keys()) & set(cur_pcts.keys())
    output: dict[str, dict[str, Any]] = {}

    for cat in sorted(common_cats):
        ref_pct = ref_pcts[cat]
        cur_pct = cur_pcts[cat]
        delta = cur_pct - ref_pct
        output[cat] = {
            "ref_pct": float(ref_pct),
            "cur_pct": float(cur_pct),
            "delta": float(delta),
            "is_shift": bool(abs(delta) > 0.05),
        }

    return output


# ---------------------------------------------------------------------------
# 5. Price regime change
# ---------------------------------------------------------------------------

def price_regime_change(
    ref_prices: pl.DataFrame,
    cur_prices: pl.DataFrame,
    by: str = "dept_id",
    price_col: str = "sell_price",
) -> dict[str, dict[str, Any]]:
    """Detect significant changes in mean sell_price by department.

    Parameters
    ----------
    ref_prices:
        Reference DataFrame with (by, price_col).
    cur_prices:
        Current DataFrame with the same schema.
    by:
        Column used for grouping (e.g. "dept_id").
    price_col:
        Name of the price column.

    Returns
    -------
    {dept: {"ref_mean": float, "cur_mean": float, "pct_change": float, "is_shift": bool}}
    Only depts present in BOTH DataFrames are reported.
    """
    def _mean_price(df: pl.DataFrame) -> dict[str, float]:
        result = (
            df.group_by(by)
            .agg(pl.col(price_col).mean().alias("mean_price"))
        )
        return {
            row[by]: float(row["mean_price"])
            for row in result.iter_rows(named=True)
        }

    ref_means = _mean_price(ref_prices)
    cur_means = _mean_price(cur_prices)

    common_depts = set(ref_means.keys()) & set(cur_means.keys())
    output: dict[str, dict[str, Any]] = {}

    for dept in sorted(common_depts):
        ref_mean = ref_means[dept]
        cur_mean = cur_means[dept]

        if ref_mean == 0.0:
            pct_change = 0.0
        else:
            pct_change = (cur_mean - ref_mean) / ref_mean

        output[dept] = {
            "ref_mean": float(ref_mean),
            "cur_mean": float(cur_mean),
            "pct_change": float(pct_change),
            "is_shift": bool(abs(pct_change) > 0.10),
        }

    return output
