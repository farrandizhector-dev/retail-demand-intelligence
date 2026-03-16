"""Segmented forecast evaluation report.

Computes metrics broken down by:
  - category (cat_id)
  - department (dept_id)
  - store (store_id)
  - state (state_id)
  - demand classification class (demand_class)
  - ABC class (abc_class)

Output:
  - data/gold/metrics/segmented_report.parquet  — full table
  - data/gold/metrics/model_metrics.json        — frontend-ready summary
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from src.evaluation.metrics import (
    bias,
    coverage_80,
    mae,
    mean_pinball_loss,
    rmse,
    smape,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Segment definitions
# ---------------------------------------------------------------------------

SEGMENT_COLS = ["cat_id", "dept_id", "store_id", "state_id", "demand_class", "abc_class"]


# ---------------------------------------------------------------------------
# Core segmented evaluation
# ---------------------------------------------------------------------------


def _compute_segment_metrics(group_df: pl.DataFrame) -> dict[str, float]:
    """Compute all metrics for a sub-DataFrame."""
    y_true = group_df["actual"].to_numpy()
    y_p50 = group_df["forecast_p50"].to_numpy()

    has_quantiles = (
        "forecast_p10" in group_df.columns
        and "forecast_p90" in group_df.columns
    )

    result: dict[str, float] = {
        "n_rows": float(len(group_df)),
        "mae": mae(y_true, y_p50),
        "rmse": rmse(y_true, y_p50),
        "smape": smape(y_true, y_p50),
        "bias": bias(y_true, y_p50),
    }

    if has_quantiles:
        y_p10 = group_df["forecast_p10"].to_numpy()
        y_p90 = group_df["forecast_p90"].to_numpy()
        result["coverage_80"] = coverage_80(y_true, y_p10, y_p90)
        result["pinball_loss"] = mean_pinball_loss(
            y_true, y_p10, y_p50, y_p90
        )

    return result


def generate_segmented_report(
    predictions_df: pl.DataFrame,
    classification_df: pl.DataFrame | None = None,
    sales_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute metrics segmented by each dimension.

    Parameters
    ----------
    predictions_df:
        DataFrame with columns:
        id, date, forecast_p10, forecast_p50, forecast_p90, actual.
        May optionally include cat_id, dept_id, store_id, state_id.
    classification_df:
        Optional demand classification table (id, demand_class, abc_class, …).
        Joined onto predictions to enable demand_class / abc_class segments.
    sales_df:
        Optional raw sales DataFrame used to attach cat_id / dept_id if
        not already in predictions_df.

    Returns
    -------
    pl.DataFrame
        Columns: segment_col, segment_value, mae, rmse, smape, bias,
        coverage_80 (if quantiles available), pinball_loss, n_rows.
    """
    # Enrich predictions with classification and metadata columns
    df = predictions_df.clone()

    if classification_df is not None and not classification_df.is_empty():
        class_cols = [c for c in ["id", "demand_class", "abc_class"] if c in classification_df.columns]
        if "id" in class_cols:
            df = df.join(
                classification_df.select(class_cols).unique("id"),
                on="id",
                how="left",
            )

    if sales_df is not None and not sales_df.is_empty():
        meta_cols = [c for c in ["id", "item_id", "cat_id", "dept_id", "store_id", "state_id"]
                     if c in sales_df.columns]
        if "id" in meta_cols:
            meta = sales_df.select(meta_cols).unique("id")
            for col in ["cat_id", "dept_id", "store_id", "state_id"]:
                if col not in df.columns and col in meta.columns:
                    df = df.join(meta.select(["id", col]), on="id", how="left")

    # Which segment columns actually exist in df?
    available_segments = [c for c in SEGMENT_COLS if c in df.columns]

    rows: list[dict] = []

    # Overall metrics
    overall = _compute_segment_metrics(df)
    rows.append({"segment_col": "overall", "segment_value": "all", **overall})

    # Per-segment metrics
    for seg_col in available_segments:
        for (seg_val,), grp in df.group_by([seg_col], maintain_order=True):
            if grp.is_empty():
                continue
            seg_metrics = _compute_segment_metrics(grp)
            rows.append({
                "segment_col": seg_col,
                "segment_value": str(seg_val),
                **seg_metrics,
            })

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------


def aggregate_fold_reports(
    fold_reports: list[pl.DataFrame],
) -> pl.DataFrame:
    """Average metrics across backtesting folds, grouped by segment.

    Parameters
    ----------
    fold_reports:
        List of DataFrames returned by ``generate_segmented_report``,
        one per fold.

    Returns
    -------
    pl.DataFrame
        Columns: segment_col, segment_value, mean_{metric}, std_{metric}.
    """
    if not fold_reports:
        return pl.DataFrame()

    combined = pl.concat(fold_reports, how="diagonal")
    metric_cols = [c for c in combined.columns
                   if c not in ("segment_col", "segment_value")]

    agg_exprs = []
    for col in metric_cols:
        agg_exprs += [
            pl.col(col).mean().alias(f"mean_{col}"),
            pl.col(col).std().alias(f"std_{col}"),
        ]

    return (
        combined
        .group_by(["segment_col", "segment_value"], maintain_order=True)
        .agg(agg_exprs)
    )


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------


def save_segmented_report(
    report_df: pl.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write segmented report to Parquet + model_metrics.json.

    Parameters
    ----------
    report_df:
        Output of ``generate_segmented_report`` or ``aggregate_fold_reports``.
    output_dir:
        Destination directory (created if needed).

    Returns
    -------
    dict with paths: "parquet" and "json".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "segmented_report.parquet"
    report_df.write_parquet(parquet_path, compression="snappy")

    # Build a JSON structure for the frontend
    metrics_json: dict = {"segments": {}}
    for (seg_col,), grp in report_df.group_by(["segment_col"], maintain_order=True):
        seg_dict: dict = {}
        for row in grp.iter_rows(named=True):
            seg_val = row["segment_value"]
            seg_dict[seg_val] = {
                k: v for k, v in row.items()
                if k not in ("segment_col", "segment_value")
                and isinstance(v, (int, float))
            }
        metrics_json["segments"][str(seg_col)] = seg_dict

    json_path = output_dir / "model_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)

    logger.info("Segmented report saved to %s", output_dir)
    return {"parquet": parquet_path, "json": json_path}
