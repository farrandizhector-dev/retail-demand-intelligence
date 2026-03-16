"""Hierarchical forecast reconciliation — V2-Fase 1.

Implements BottomUp, TopDown (average_proportions), and MinTrace (mint_shrink)
reconciliation via ``hierarchicalforecast`` (Nixtla).

Public entry points
-------------------
reconcile_forecasts(Y_hat_df, S_df, tags, method, Y_df)
    Reconcile a single method on the full or sub-hierarchy.

reconcile_all_methods(Y_hat_df, S_df, tags, Y_df, model_col)
    Run all three methods and return a combined DataFrame.

run_reconciliation_backtest(sales_df, forecast_df, catalog_df, ...)
    5-fold backtesting comparison: BU vs TD vs MinT by hierarchy level.

Notes on scalability
--------------------
MinTrace (mint_shrink) requires estimating a covariance matrix of shape
(n_bottom × n_bottom).  For the full M5 hierarchy (n_bottom = 30,490) this
is infeasible.  The function ``_reconcile_mint_sub_hierarchy`` partitions the
catalog by (store_id, dept_id) into 70 groups of ≤ 500 series each, reconciles
each independently, and merges the results.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

from src.reconciliation.hierarchy import (
    HierarchyMatrix,
    build_hierarchy_matrix,
    build_sub_hierarchy,
    _bottom_id,
)

logger = logging.getLogger(__name__)

ReconciliationMethod = Literal["bu", "td", "mint_shrink"]

# Output column suffix mapping (mirrors hierarchicalforecast naming)
METHOD_COL: dict[str, str] = {
    "bu": "BottomUp",
    "td": "TopDown_method-average_proportions",
    "mint_shrink": "MinTrace_method-mint_shrink",
}


# ---------------------------------------------------------------------------
# Core reconciliation
# ---------------------------------------------------------------------------


def reconcile_forecasts(
    Y_hat_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    method: ReconciliationMethod,
    Y_df: pd.DataFrame | None = None,
    model_col: str = "forecast_p50",
) -> pd.DataFrame:
    """Reconcile base forecasts with one method.

    Parameters
    ----------
    Y_hat_df:
        Base forecasts.  Must contain ``[unique_id, ds, {model_col}]`` for all
        series (bottom + aggregates).
    S_df:
        Summing matrix from ``build_hierarchy_matrix``.  First column is
        ``unique_id``; remaining columns are bottom-series IDs.
    tags:
        Aggregation tags dict from ``build_hierarchy_matrix``.
    method:
        ``"bu"``, ``"td"``, or ``"mint_shrink"``.
    Y_df:
        Actuals + in-sample predictions ``[unique_id, ds, y, {model_col}]``.
        Required for ``"mint_shrink"``; ignored otherwise.
    model_col:
        Forecast column name in ``Y_hat_df`` (default ``"forecast_p50"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``[unique_id, ds, {model_col}, {model_col}/{suffix}]``.
    """
    from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
    from hierarchicalforecast.core import HierarchicalReconciliation

    if method == "bu":
        reconciler = BottomUp()
    elif method == "td":
        reconciler = TopDown(method="average_proportions")
    elif method == "mint_shrink":
        reconciler = MinTrace(method="mint_shrink")
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose from 'bu', 'td', 'mint_shrink'."
        )

    hrec = HierarchicalReconciliation(reconcilers=[reconciler])
    return hrec.reconcile(
        Y_hat_df=Y_hat_df,
        S_df=S_df,
        tags=tags,
        Y_df=Y_df,
    )


def reconcile_all_methods(
    Y_hat_df: pd.DataFrame,
    S_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    Y_df: pd.DataFrame | None = None,
    model_col: str = "forecast_p50",
) -> pd.DataFrame:
    """Reconcile BU, TD, and MinT-Shrink, returning a merged DataFrame.

    Output columns (``{model_col}`` = base name, e.g. ``Forecast``):

    - ``{model_col}/BottomUp``
    - ``{model_col}/TopDown_method-average_proportions``
    - ``{model_col}/MinTrace_method-mint_shrink``  *(if Y_df provided)*

    MinTrace is skipped with a warning when ``Y_df`` is ``None``.
    """
    from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
    from hierarchicalforecast.core import HierarchicalReconciliation

    # TopDown(average_proportions) is also in-sample (requires Y_df).
    # BottomUp is the only method that works without actuals.
    reconcilers = [BottomUp()]
    if Y_df is not None:
        reconcilers.append(TopDown(method="average_proportions"))
        reconcilers.append(MinTrace(method="mint_shrink"))
    else:
        logger.warning(
            "Y_df not provided — TopDown and MinTrace skipped (only BottomUp will run)."
        )

    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    return hrec.reconcile(
        Y_hat_df=Y_hat_df,
        S_df=S_df,
        tags=tags,
        Y_df=Y_df,
    )


# ---------------------------------------------------------------------------
# Sub-hierarchy MinTrace (scalable for large bottom sets)
# ---------------------------------------------------------------------------


def reconcile_mint_sub_hierarchy(
    Y_hat_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    Y_df: pd.DataFrame | None = None,
    model_col: str = "forecast_p50",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """MinTrace via per-group sub-hierarchy reconciliation.

    Partitions the catalog by ``group_cols`` (default ``store_id × dept_id``),
    runs MinTrace independently per group, then concatenates bottom-level
    results.

    Parameters
    ----------
    Y_hat_df:
        Base forecast DataFrame (bottom-level series only).
    catalog_df:
        M5 catalog with hierarchy columns.
    Y_df:
        In-sample actuals + predictions ``[unique_id, ds, y, {model_col}]``.
    model_col:
        Forecast column name.
    group_cols:
        Partition columns (default ``["store_id", "dept_id"]``).

    Returns
    -------
    pd.DataFrame with bottom-level reconciled column
    ``{model_col}/MinTrace_method-mint_shrink``.
    """
    from hierarchicalforecast.methods import MinTrace
    from hierarchicalforecast.core import HierarchicalReconciliation

    if group_cols is None:
        group_cols = ["store_id", "dept_id"]

    sub_hierarchies = build_sub_hierarchy(catalog_df, group_cols)
    reconciled_parts: list[pd.DataFrame] = []

    for group_key, hm in sub_hierarchies.items():
        # Bottom-series IDs for this group (skip 'unique_id' column)
        bottom_ids = list(hm.S_df.columns[1:])

        Y_hat_sub = Y_hat_df[Y_hat_df["unique_id"].isin(bottom_ids)].copy()
        if Y_hat_sub.empty:
            continue

        # Group total series
        grp_label = "/".join(str(v) for v in group_key)
        grp_total = f"Total_{grp_label}"

        total_rows = (
            Y_hat_sub.groupby("ds")[model_col]
            .sum()
            .reset_index()
            .assign(unique_id=grp_total)
        )
        Y_hat_sub_full = pd.concat([Y_hat_sub, total_rows], ignore_index=True)

        sub_tags = {
            f"Total_{grp_label}": np.array([grp_total]),
            "SKU-Store": np.array(bottom_ids),
        }

        n_bottom = len(bottom_ids)
        S_data = np.vstack([
            np.ones((1, n_bottom), dtype=float),
            np.eye(n_bottom, dtype=float),
        ])
        sub_all_ids = [grp_total] + list(bottom_ids)
        sub_S_df = pd.DataFrame(
            S_data, index=sub_all_ids, columns=bottom_ids
        ).reset_index(names="unique_id")

        Y_df_sub = None
        if Y_df is not None:
            bottom_y = Y_df[Y_df["unique_id"].isin(bottom_ids)].copy()
            if not bottom_y.empty and model_col in bottom_y.columns:
                total_y = (
                    bottom_y.groupby("ds")[["y", model_col]]
                    .sum()
                    .reset_index()
                    .assign(unique_id=grp_total)
                )
                Y_df_sub = pd.concat([bottom_y, total_y], ignore_index=True)

        try:
            hrec = HierarchicalReconciliation(
                reconcilers=[MinTrace(method="mint_shrink")]
            )
            out = hrec.reconcile(
                Y_hat_df=Y_hat_sub_full,
                S_df=sub_S_df,
                tags=sub_tags,
                Y_df=Y_df_sub,
            )
            reconciled_parts.append(out[out["unique_id"].isin(bottom_ids)])
        except Exception as exc:
            logger.warning(
                "MinT sub-hierarchy %s failed: %s — falling back to base forecast.",
                group_key,
                exc,
            )
            fallback = Y_hat_sub[["unique_id", "ds", model_col]].copy()
            rec_col = f"{model_col}/MinTrace_method-mint_shrink"
            fallback[rec_col] = fallback[model_col]
            reconciled_parts.append(fallback)

    if not reconciled_parts:
        return Y_hat_df[["unique_id", "ds", model_col]].copy()

    return pd.concat(reconciled_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregate bottom-level forecasts to all hierarchy levels
# ---------------------------------------------------------------------------


def aggregate_base_forecasts(
    bottom_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    tags: dict[str, np.ndarray],
    model_col: str = "forecast_p50",
    date_col: str = "ds",
    id_col: str = "unique_id",
) -> pd.DataFrame:
    """Sum bottom-level forecasts upward through the M5 hierarchy.

    Parameters
    ----------
    bottom_df:
        Bottom-level (SKU-Store) forecasts: ``[unique_id, ds, model_col]``.
    catalog_df:
        M5 catalog with 5 hierarchy columns.
    tags:
        Tags from ``build_hierarchy_matrix`` (used to validate level names).
    model_col:
        Forecast column name.

    Returns
    -------
    pd.DataFrame with all hierarchy levels concatenated
    (``[unique_id, ds, model_col]``).
    """
    catalog_clean = catalog_df.drop_duplicates(subset=["item_id", "store_id"]).copy()
    catalog_clean["unique_id"] = catalog_clean.apply(
        lambda r: _bottom_id(r["item_id"], r["store_id"]), axis=1
    )

    merged = bottom_df.merge(
        catalog_clean[[
            "unique_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"
        ]],
        on="unique_id",
        how="left",
    )

    parts: list[pd.DataFrame] = [
        bottom_df[[id_col, date_col, model_col]].copy()
    ]

    level_specs: list[tuple[str, list[str]]] = [
        ("Total",            []),
        ("State",            ["state_id"]),
        ("Store",            ["store_id"]),
        ("Category",         ["cat_id"]),
        ("Department",       ["dept_id"]),
        ("State/Category",   ["state_id", "cat_id"]),
        ("State/Department", ["state_id", "dept_id"]),
        ("Store/Category",   ["store_id", "cat_id"]),
        ("Store/Department", ["store_id", "dept_id"]),
        ("Product",          ["item_id"]),
        ("Product/State",    ["item_id", "state_id"]),
    ]

    for _level_name, grp_cols in level_specs:
        if not grp_cols:
            agg = (
                merged.groupby(date_col)[model_col]
                .sum()
                .reset_index()
                .assign(**{id_col: "Total"})
            )
        else:
            agg = (
                merged.groupby(grp_cols + [date_col])[model_col]
                .sum()
                .reset_index()
            )
            agg[id_col] = agg[grp_cols].apply(
                lambda row: "/".join(str(row[c]) for c in grp_cols), axis=1
            )
        parts.append(agg[[id_col, date_col, model_col]])

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Backtesting comparison (BU vs TD vs MinT)
# ---------------------------------------------------------------------------


def run_reconciliation_backtest(
    sales_df: pl.DataFrame,
    forecast_df: pl.DataFrame,
    catalog_df: pd.DataFrame,
    output_dir: str = "data/gold/metrics",
    model_col: str = "forecast_p50",
    id_col: str = "id",
    date_col: str = "date",
    sales_col: str = "sales",
    n_folds: int = 5,
) -> pd.DataFrame:
    """Compare BU, TD, MinT on rolling-origin folds by hierarchy level.

    Parameters
    ----------
    sales_df:
        Polars DataFrame ``[id, date, sales]``.  The ``id`` must follow the
        ``{item_id}_{store_id}`` bottom-level convention.
    forecast_df:
        Polars DataFrame ``[id, date, forecast_p50]`` — LightGBM base forecasts
        at the bottom level.
    catalog_df:
        Pandas DataFrame with the 5 M5 hierarchy columns.
    output_dir:
        Directory for ``reconciliation_comparison.parquet``.
    n_folds:
        Number of rolling-origin folds.

    Returns
    -------
    pd.DataFrame with columns ``[fold, method, level, mae]``.
    Also written to ``{output_dir}/reconciliation_comparison.parquet``.
    """
    from pathlib import Path
    from hierarchicalforecast.methods import BottomUp, MinTrace, TopDown
    from hierarchicalforecast.core import HierarchicalReconciliation

    hm = build_hierarchy_matrix(catalog_df)
    S_df = hm.S_df
    tags = hm.tags

    # Polars → pandas
    s_pd = sales_df.to_pandas() if isinstance(sales_df, pl.DataFrame) else sales_df
    f_pd = forecast_df.to_pandas() if isinstance(forecast_df, pl.DataFrame) else forecast_df
    s_pd = s_pd.rename(columns={id_col: "unique_id", date_col: "ds", sales_col: "y"})
    f_pd = f_pd.rename(columns={id_col: "unique_id", date_col: "ds"})

    all_dates = sorted(s_pd["ds"].unique())
    n_dates = len(all_dates)
    test_window = max(1, n_dates // (n_folds + 1))

    results: list[dict] = []

    uid_to_level: dict[str, str] = {
        uid: lev for lev, arr in tags.items() for uid in arr
    }

    for fold_i in range(n_folds):
        cutoff_idx = n_dates - (n_folds - fold_i) * test_window
        if cutoff_idx <= 0 or cutoff_idx >= n_dates:
            continue

        cutoff_date = all_dates[cutoff_idx - 1]
        test_start = all_dates[cutoff_idx]
        test_end = all_dates[min(cutoff_idx + test_window - 1, n_dates - 1)]

        fc_mask = (f_pd["ds"] >= test_start) & (f_pd["ds"] <= test_end)
        fc_bottom = f_pd[fc_mask].copy()
        if fc_bottom.empty:
            continue

        act_test_mask = (s_pd["ds"] >= test_start) & (s_pd["ds"] <= test_end)
        actuals_bottom = s_pd[act_test_mask].copy()

        try:
            fc_all = aggregate_base_forecasts(
                fc_bottom, catalog_df, tags, model_col=model_col
            ).rename(columns={model_col: "Forecast"})
            act_all = aggregate_base_forecasts(
                actuals_bottom.rename(columns={"y": model_col}),
                catalog_df, tags, model_col=model_col,
            ).rename(columns={model_col: "y"})
        except Exception as exc:
            logger.warning("Fold %d aggregation failed: %s", fold_i, exc)
            continue

        # Build Y_df for MinTrace (use same test-period data as proxy)
        try:
            fc_insample = f_pd[f_pd["ds"] > cutoff_date].copy()
            insample_all = aggregate_base_forecasts(
                fc_insample, catalog_df, tags, model_col=model_col,
            ).rename(columns={model_col: "Forecast"})
            act_insample = aggregate_base_forecasts(
                s_pd[s_pd["ds"] > cutoff_date].rename(columns={"y": model_col}),
                catalog_df, tags, model_col=model_col,
            ).rename(columns={model_col: "y"})
            Y_df_fold = act_insample.merge(
                insample_all[["unique_id", "ds", "Forecast"]],
                on=["unique_id", "ds"], how="left"
            )
        except Exception as exc:
            logger.warning("Fold %d Y_df build failed: %s", fold_i, exc)
            Y_df_fold = None

        reconcilers = [BottomUp(), TopDown(method="average_proportions")]
        if Y_df_fold is not None:
            reconcilers.append(MinTrace(method="mint_shrink"))

        try:
            hrec = HierarchicalReconciliation(reconcilers=reconcilers)
            reconciled = hrec.reconcile(
                Y_hat_df=fc_all, S_df=S_df, tags=tags, Y_df=Y_df_fold
            )
        except Exception as exc:
            logger.warning("Fold %d reconciliation failed: %s", fold_i, exc)
            continue

        eval_df = reconciled.merge(act_all, on=["unique_id", "ds"], how="inner")
        eval_df["level"] = eval_df["unique_id"].map(uid_to_level).fillna("Unknown")

        method_cols = {
            "bu":         "Forecast/BottomUp",
            "td":         "Forecast/TopDown_method-average_proportions",
            "mint_shrink": "Forecast/MinTrace_method-mint_shrink",
        }
        for method_key, col in method_cols.items():
            if col not in eval_df.columns:
                continue
            level_mae = (
                eval_df.groupby("level")
                .apply(
                    lambda g, _col=col: float(np.mean(np.abs(g["y"] - g[_col]))),
                    include_groups=False,
                )
                .reset_index(name="mae")
            )
            for _, row in level_mae.iterrows():
                results.append({
                    "fold": fold_i,
                    "method": method_key,
                    "level": row["level"],
                    "mae": float(row["mae"]),
                })

    if not results:
        logger.warning("No reconciliation backtest results produced.")
        return pd.DataFrame(columns=["fold", "method", "level", "mae"])

    comparison_df = pd.DataFrame(results)

    # Best method selection: Total-level revenue-weighted MAE
    total_mae = (
        comparison_df[comparison_df["level"] == "Total"]
        .groupby("method")["mae"]
        .mean()
    )
    if not total_mae.empty:
        best = total_mae.idxmin()
        logger.info(
            "Best reconciliation method (Total-level MAE): %s (%.4f)", best, total_mae[best]
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pq_file = out_path / "reconciliation_comparison.parquet"
    comparison_df.to_parquet(pq_file, index=False)
    logger.info("Saved reconciliation comparison: %s (%d rows)", pq_file, len(comparison_df))

    return comparison_df
