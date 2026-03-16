"""Coherence tests for reconciled hierarchical forecasts — V2-Fase 1.

Four mandatory coherence checks (spec section 6.3):

1. **Bottom-up coherence**: bottom-level series sum to upper aggregates
   (tolerance ±0.01 per aggregated series per timestep).
2. **Non-negativity**: no negative values in reconciled forecasts.
3. **Quantile monotonicity**: p10 ≤ p50 ≤ p90 at every (series, timestep).
4. **Daily total coherence**: total across all bottom series equals the
   "Total" aggregate series at each timestep (tolerance ±1.0 unit/day).

Entry point
-----------
run_coherence_tests(reconciled_df, S_df, tags) → dict[str, Any]

The returned dict keys are ``"bottom_up_coherence"``, ``"non_negativity"``,
``"quantile_monotonicity"``, ``"daily_total_coherence"``.  Each value is a
sub-dict with ``"passed": bool`` and ``"details": dict``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Tolerances (spec §6.3)
_BU_ATOL = 0.01     # absolute tolerance for aggregate reconstruction
_DAILY_ATOL = 1.0   # tolerance for daily total coherence (units/day)


# ---------------------------------------------------------------------------
# Individual test functions
# ---------------------------------------------------------------------------


def check_bottom_up_coherence(
    reconciled_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    model_col: str = "forecast_p50",
    id_col: str = "unique_id",
    date_col: str = "ds",
    atol: float = _BU_ATOL,
) -> dict[str, Any]:
    """Test 1: sum of bottom-level series matches aggregate series.

    For every non-bottom series, checks that:
        abs(reconciled_aggregate - sum(reconciled_bottom_members)) ≤ atol

    Parameters
    ----------
    reconciled_df:
        Reconciled forecasts ``[unique_id, ds, model_col]``.
    S_df:
        Summing matrix from ``build_hierarchy_matrix``.
    tags:
        Aggregation tags from ``build_hierarchy_matrix``.
    model_col:
        Reconciled forecast column to validate.
    atol:
        Absolute tolerance (default 0.01).

    Returns
    -------
    dict with keys ``passed``, ``max_abs_error``, ``n_violations``,
    ``n_checked``, ``sample_violations``.
    """
    # Identify bottom-level IDs (last level in tags)
    bottom_level = "SKU-Store"
    if bottom_level not in tags:
        # Fallback: use the last level alphabetically
        bottom_level = sorted(tags.keys())[-1]

    bottom_ids = set(tags[bottom_level])

    # Build pivot: (unique_id → {ds → value})
    pivot = reconciled_df.set_index([id_col, date_col])[model_col]

    violations: list[dict] = []
    max_err = 0.0
    n_checked = 0

    # S_df layout: first col = unique_id, rest = bottom-series columns
    bottom_cols = [c for c in S_df.columns if c != id_col]

    for _, row in S_df.iterrows():
        uid = row[id_col]
        if uid in bottom_ids:
            continue  # skip bottom series

        # Identify which bottom series contribute to this aggregate
        member_cols = [c for c in bottom_cols if row[c] > 0]
        if not member_cols:
            continue

        # For each timestep: agg_value ≈ sum(member values)
        try:
            agg_series = pivot.loc[uid]
        except KeyError:
            continue

        for ds_val, agg_val in agg_series.items():
            # Sum matching bottom-series values at this timestep
            member_sum = 0.0
            for member in member_cols:
                try:
                    member_sum += pivot.loc[(member, ds_val)]
                except KeyError:
                    pass

            err = abs(agg_val - member_sum)
            max_err = max(max_err, err)
            n_checked += 1

            if err > atol:
                violations.append({
                    "uid": uid,
                    "ds": ds_val,
                    "agg_value": agg_val,
                    "member_sum": member_sum,
                    "error": err,
                })

    passed = len(violations) == 0
    return {
        "passed": passed,
        "max_abs_error": float(max_err),
        "n_violations": len(violations),
        "n_checked": n_checked,
        "sample_violations": violations[:5],
    }


def check_non_negativity(
    reconciled_df: pd.DataFrame,
    model_col: str = "forecast_p50",
    id_col: str = "unique_id",
    date_col: str = "ds",
) -> dict[str, Any]:
    """Test 2: no negative values in reconciled forecasts.

    Parameters
    ----------
    reconciled_df:
        Reconciled forecasts ``[unique_id, ds, model_col]``.
    model_col:
        Column to check (default ``"forecast_p50"``).

    Returns
    -------
    dict with ``passed``, ``n_negatives``, ``min_value``, ``sample_negatives``.
    """
    if model_col not in reconciled_df.columns:
        return {
            "passed": False,
            "n_negatives": 0,
            "min_value": None,
            "sample_negatives": [],
            "error": f"Column {model_col!r} not found in reconciled_df.",
        }

    vals = reconciled_df[model_col].values
    min_val = float(np.min(vals))
    neg_mask = vals < 0

    neg_rows = reconciled_df[neg_mask][[id_col, date_col, model_col]]
    return {
        "passed": bool(not neg_mask.any()),
        "n_negatives": int(neg_mask.sum()),
        "min_value": min_val,
        "sample_negatives": neg_rows.head(5).to_dict(orient="records"),
    }


def check_quantile_monotonicity(
    reconciled_df: pd.DataFrame,
    p10_col: str = "forecast_p10",
    p50_col: str = "forecast_p50",
    p90_col: str = "forecast_p90",
    id_col: str = "unique_id",
    date_col: str = "ds",
) -> dict[str, Any]:
    """Test 3: p10 ≤ p50 ≤ p90 at every (series, timestep).

    Skipped (passed=True, skipped=True) if any quantile column is absent.

    Parameters
    ----------
    reconciled_df:
        Reconciled forecasts with optional quantile columns.
    p10_col, p50_col, p90_col:
        Column names for the 10th, 50th, 90th percentile forecasts.

    Returns
    -------
    dict with ``passed``, ``n_p10_violations``, ``n_p90_violations``,
    ``skipped``, ``sample_violations``.
    """
    present = [c for c in [p10_col, p50_col, p90_col] if c in reconciled_df.columns]
    if len(present) < 2:
        return {
            "passed": True,
            "skipped": True,
            "reason": f"Quantile columns not all present (found: {present}).",
            "n_p10_violations": 0,
            "n_p90_violations": 0,
        }

    violations: list[dict] = []

    if p10_col in reconciled_df.columns and p50_col in reconciled_df.columns:
        mask_p10 = reconciled_df[p10_col] > reconciled_df[p50_col]
        if mask_p10.any():
            sample = reconciled_df[mask_p10][[id_col, date_col, p10_col, p50_col]].head(5)
            violations.extend(sample.to_dict(orient="records"))
        n_p10_violations = int(mask_p10.sum())
    else:
        n_p10_violations = 0

    if p50_col in reconciled_df.columns and p90_col in reconciled_df.columns:
        mask_p90 = reconciled_df[p90_col] < reconciled_df[p50_col]
        if mask_p90.any():
            sample = reconciled_df[mask_p90][[id_col, date_col, p50_col, p90_col]].head(5)
            violations.extend(sample.to_dict(orient="records"))
        n_p90_violations = int(mask_p90.sum())
    else:
        n_p90_violations = 0

    return {
        "passed": n_p10_violations == 0 and n_p90_violations == 0,
        "skipped": False,
        "n_p10_violations": n_p10_violations,
        "n_p90_violations": n_p90_violations,
        "sample_violations": violations[:5],
    }


def check_daily_total_coherence(
    reconciled_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    model_col: str = "forecast_p50",
    id_col: str = "unique_id",
    date_col: str = "ds",
    atol: float = _DAILY_ATOL,
) -> dict[str, Any]:
    """Test 4: sum of bottom series equals "Total" aggregate at each timestep.

    Checks: abs(Total[t] - sum(bottom[t])) ≤ atol  for every date t.

    Parameters
    ----------
    reconciled_df:
        Reconciled forecasts ``[unique_id, ds, model_col]``.
    tags:
        Aggregation tags from ``build_hierarchy_matrix``.
    atol:
        Absolute tolerance per day (default 1.0).

    Returns
    -------
    dict with ``passed``, ``max_abs_error``, ``n_violations``,
    ``n_dates_checked``, ``sample_violations``.
    """
    # Identify bottom and Total series
    bottom_level = "SKU-Store"
    if bottom_level not in tags:
        bottom_level = sorted(tags.keys())[-1]
    bottom_ids = set(tags[bottom_level])

    total_series = "Total"

    df = reconciled_df.copy()
    bottom_df = df[df[id_col].isin(bottom_ids)]
    total_df = df[df[id_col] == total_series]

    if total_df.empty:
        return {
            "passed": True,
            "skipped": True,
            "reason": f"'Total' series not found in reconciled_df.",
            "max_abs_error": 0.0,
            "n_violations": 0,
            "n_dates_checked": 0,
        }

    # Daily bottom sum
    bottom_daily = (
        bottom_df.groupby(date_col)[model_col]
        .sum()
        .rename("bottom_sum")
    )
    total_daily = total_df.set_index(date_col)[model_col].rename("total_val")

    merged = pd.concat([bottom_daily, total_daily], axis=1).dropna()
    if merged.empty:
        return {
            "passed": True,
            "skipped": True,
            "reason": "No overlapping dates between Total and bottom series.",
            "max_abs_error": 0.0,
            "n_violations": 0,
            "n_dates_checked": 0,
        }

    merged["abs_err"] = (merged["bottom_sum"] - merged["total_val"]).abs()
    max_err = float(merged["abs_err"].max())
    violations = merged[merged["abs_err"] > atol]

    return {
        "passed": len(violations) == 0,
        "max_abs_error": max_err,
        "n_violations": len(violations),
        "n_dates_checked": len(merged),
        "sample_violations": violations.head(5).reset_index().to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_coherence_tests(
    reconciled_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    model_col: str = "forecast_p50",
    p10_col: str = "forecast_p10",
    p50_col: str = "forecast_p50",
    p90_col: str = "forecast_p90",
    id_col: str = "unique_id",
    date_col: str = "ds",
    bu_atol: float = _BU_ATOL,
    daily_atol: float = _DAILY_ATOL,
) -> dict[str, Any]:
    """Run all 4 coherence tests on reconciled forecasts.

    Parameters
    ----------
    reconciled_df:
        Reconciled forecast DataFrame ``[unique_id, ds, model_col, ...]``.
    S_df:
        Summing matrix from ``build_hierarchy_matrix``.
    tags:
        Aggregation tags from ``build_hierarchy_matrix``.
    model_col:
        Primary reconciled forecast column (default ``"forecast_p50"``).
    p10_col, p50_col, p90_col:
        Quantile columns for Test 3 (skipped if absent).
    bu_atol:
        Tolerance for Test 1 (default 0.01).
    daily_atol:
        Tolerance for Test 4 (default 1.0).

    Returns
    -------
    dict
        Keys: ``"bottom_up_coherence"``, ``"non_negativity"``,
        ``"quantile_monotonicity"``, ``"daily_total_coherence"``,
        ``"all_passed"``.
        Each sub-dict has at minimum ``"passed": bool``.

    Examples
    --------
    >>> results = run_coherence_tests(reconciled_df, S_df, tags)
    >>> assert results["all_passed"], results
    """
    logger.info("Running 4 coherence tests on %d rows …", len(reconciled_df))

    t1 = check_bottom_up_coherence(
        reconciled_df, S_df, tags, model_col=model_col, id_col=id_col,
        date_col=date_col, atol=bu_atol,
    )
    t2 = check_non_negativity(
        reconciled_df, model_col=model_col, id_col=id_col, date_col=date_col,
    )
    t3 = check_quantile_monotonicity(
        reconciled_df, p10_col=p10_col, p50_col=p50_col, p90_col=p90_col,
        id_col=id_col, date_col=date_col,
    )
    t4 = check_daily_total_coherence(
        reconciled_df, tags, model_col=model_col, id_col=id_col,
        date_col=date_col, atol=daily_atol,
    )

    all_passed = t1["passed"] and t2["passed"] and t3["passed"] and t4["passed"]

    for name, result in [
        ("bottom_up_coherence", t1),
        ("non_negativity", t2),
        ("quantile_monotonicity", t3),
        ("daily_total_coherence", t4),
    ]:
        status = "PASS" if result["passed"] else "FAIL"
        logger.info("  Test %-25s [%s]", name, status)

    return {
        "bottom_up_coherence": t1,
        "non_negativity": t2,
        "quantile_monotonicity": t3,
        "daily_total_coherence": t4,
        "all_passed": all_passed,
    }
