"""SHAP-based feature importance analysis for the LightGBM global model.

Generates interpretability artefacts using ``shap.TreeExplainer`` (the
fastest and most accurate explainer for gradient-boosted trees):

- mean_abs_shap per feature → top-30 ranking
- shap_by_category: mean |SHAP| per feature × demand category
- shap_by_demand_class: mean |SHAP| per feature × demand class
- JSON exports for the frontend (< 50 KB budget, spec §13.4)

Usage
-----
from src.evaluation.shap_analysis import run_shap_analysis

result = run_shap_analysis(
    model_dir="models/lgbm_final",
    feature_df=silver_features_df,  # Polars DataFrame
    output_dir="data/gold/metrics",
    mlflow_run=active_mlflow_run,   # optional
)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Maximum rows used for SHAP computation (avoid OOM on full M5)
SHAP_SAMPLE_SIZE = 10_000
TOP_N_FEATURES = 30


# ---------------------------------------------------------------------------
# Core SHAP computation
# ---------------------------------------------------------------------------


def compute_shap_values(
    model,
    X_sample: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model:
        Trained ``lightgbm.Booster``.
    X_sample:
        Pandas DataFrame of feature rows (already encoded, same columns as
        used during training).

    Returns
    -------
    np.ndarray of shape ``(n_rows, n_features)`` — raw SHAP values.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)

    # Some versions of shap / LightGBM return a list (multi-output); flatten
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    return np.asarray(shap_vals, dtype=float)


def generate_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = TOP_N_FEATURES,
) -> dict[str, Any]:
    """Aggregate SHAP values into a summary dict.

    Parameters
    ----------
    shap_values:
        Array of shape ``(n_samples, n_features)``.
    feature_names:
        Ordered list of feature names (must match axis-1 of shap_values).
    top_n:
        Number of top features to include in the summary.

    Returns
    -------
    dict with keys:
        ``top_features`` — list of ``{feature, mean_abs_shap, rank}`` for top N.
        ``all_features``  — dict ``{feature: mean_abs_shap}`` for all features.
        ``n_samples``     — number of rows used.
        ``n_features``    — total feature count.
    """
    if len(feature_names) != shap_values.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) != "
            f"shap_values.shape[1] ({shap_values.shape[1]})."
        )

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]  # descending

    all_features = {
        feature_names[i]: float(mean_abs[i]) for i in range(len(feature_names))
    }
    top_features = [
        {
            "feature": feature_names[i],
            "mean_abs_shap": float(mean_abs[i]),
            "rank": rank + 1,
        }
        for rank, i in enumerate(order[:top_n])
    ]

    return {
        "top_features": top_features,
        "all_features": all_features,
        "n_samples": int(shap_values.shape[0]),
        "n_features": int(shap_values.shape[1]),
    }


# ---------------------------------------------------------------------------
# Segment-level SHAP
# ---------------------------------------------------------------------------


def shap_by_segment(
    shap_values: np.ndarray,
    feature_names: list[str],
    segment_series: pd.Series,
    top_n: int = TOP_N_FEATURES,
) -> dict[str, list[dict]]:
    """Compute mean |SHAP| per feature for each segment value.

    Parameters
    ----------
    shap_values:
        Shape ``(n_samples, n_features)``.
    feature_names:
        Feature name list (length = n_features).
    segment_series:
        Pandas Series of segment labels (length = n_samples).
        E.g. the ``cat_id`` column (FOODS / HOUSEHOLD / HOBBIES).
    top_n:
        Top features to include per segment.

    Returns
    -------
    dict  ``{segment_value: [{"feature", "mean_abs_shap"}, ...]}``
    """
    segment_series = segment_series.reset_index(drop=True)
    result: dict[str, list[dict]] = {}

    for seg_val in sorted(segment_series.unique()):
        mask = (segment_series == seg_val).values
        if mask.sum() == 0:
            continue
        seg_shap = shap_values[mask]
        mean_abs = np.mean(np.abs(seg_shap), axis=0)
        order = np.argsort(mean_abs)[::-1][:top_n]
        result[str(seg_val)] = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order
        ]

    return result


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_shap_for_frontend(
    shap_summary: dict[str, Any],
    shap_by_cat: dict[str, list[dict]] | None,
    shap_by_demand: dict[str, list[dict]] | None,
    output_dir: Path,
) -> dict[str, Path]:
    """Write SHAP artefacts to JSON files for frontend consumption.

    Parameters
    ----------
    shap_summary:
        Output of ``generate_shap_summary``.
    shap_by_cat:
        Output of ``shap_by_segment`` for cat_id column.
    shap_by_demand:
        Output of ``shap_by_segment`` for demand_class column.
    output_dir:
        Destination directory.

    Returns
    -------
    dict mapping artefact name → Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # shap_summary.json (top 30 only — < 50 KB target)
    summary_payload = {
        "top_features": shap_summary["top_features"],
        "n_samples": shap_summary["n_samples"],
        "n_features": shap_summary["n_features"],
    }
    p = output_dir / "shap_summary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, separators=(",", ":"))
    paths["shap_summary"] = p
    logger.info("shap_summary.json written: %.1f KB", p.stat().st_size / 1024)

    # shap_by_segment.json
    by_seg_payload: dict[str, Any] = {}
    if shap_by_cat is not None:
        by_seg_payload["by_category"] = shap_by_cat
    if shap_by_demand is not None:
        by_seg_payload["by_demand_class"] = shap_by_demand

    if by_seg_payload:
        p2 = output_dir / "shap_by_segment.json"
        with open(p2, "w", encoding="utf-8") as f:
            json.dump(by_seg_payload, f, separators=(",", ":"))
        paths["shap_by_segment"] = p2
        logger.info("shap_by_segment.json written: %.1f KB", p2.stat().st_size / 1024)

    return paths


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_shap_analysis(
    model_dir: str | Path,
    feature_df: pl.DataFrame | pd.DataFrame,
    output_dir: str | Path,
    sample_size: int = SHAP_SAMPLE_SIZE,
    top_n: int = TOP_N_FEATURES,
    mlflow_run=None,
    cat_col: str = "cat_id",
    demand_class_col: str = "demand_class",
    random_seed: int = 42,
) -> dict[str, Any]:
    """End-to-end SHAP analysis: load model, sample, compute, export.

    Parameters
    ----------
    model_dir:
        Directory containing ``model_point.lgb`` and ``model_meta.json``,
        as written by ``TrainedModels.save``.
    feature_df:
        Feature store DataFrame (Polars or pandas).  Must include all
        training feature columns plus (optionally) ``cat_id`` and
        ``demand_class`` for segment analysis.
    output_dir:
        Where to write ``shap_summary.json`` and ``shap_by_segment.json``.
    sample_size:
        Maximum rows to use for SHAP computation.  Full dataset is used
        if smaller than this.
    top_n:
        Number of top features to include in outputs.
    mlflow_run:
        Active MLflow run for logging the SHAP summary as an artifact.
    cat_col, demand_class_col:
        Column names for category and demand-class segmentation.
    random_seed:
        Seed for reproducible sampling.

    Returns
    -------
    dict with keys: ``shap_summary``, ``shap_by_category``,
    ``shap_by_demand_class``, ``output_paths``.
    """
    from src.models.training import TrainedModels, get_feature_cols

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)

    # ── Load model ──────────────────────────────────────────────────────────
    trained = TrainedModels.load(model_dir)
    feature_cols = trained.feature_cols
    cat_cols = trained.cat_cols

    # ── Prepare feature matrix ──────────────────────────────────────────────
    if isinstance(feature_df, pl.DataFrame):
        pdf = feature_df.to_pandas()
    else:
        pdf = feature_df.copy()

    # Keep only feature columns present in the DataFrame
    available = [c for c in feature_cols if c in pdf.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        logger.warning("SHAP: %d feature columns missing in feature_df: %s", len(missing), sorted(missing)[:10])

    X_full = pdf[available].copy()

    # Encode categoricals as pandas 'category' dtype (same as training)
    for col in cat_cols:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype("category")

    # ── Sample ─────────────────────────────────────────────────────────────
    if len(X_full) > sample_size:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(X_full), size=sample_size, replace=False)
        X_sample = X_full.iloc[idx].reset_index(drop=True)
        # Keep segment columns for same rows
        seg_pdf = pdf.iloc[idx].reset_index(drop=True)
    else:
        X_sample = X_full.reset_index(drop=True)
        seg_pdf = pdf.reset_index(drop=True)

    logger.info("SHAP: computing values for %d rows × %d features", len(X_sample), len(available))

    # ── Compute SHAP values ─────────────────────────────────────────────────
    shap_vals = compute_shap_values(trained.point_model, X_sample)

    # ── Summary ────────────────────────────────────────────────────────────
    shap_sum = generate_shap_summary(shap_vals, available, top_n=top_n)

    # ── Segment analysis ────────────────────────────────────────────────────
    shap_cat: dict[str, list[dict]] | None = None
    shap_demand: dict[str, list[dict]] | None = None

    if cat_col in seg_pdf.columns:
        shap_cat = shap_by_segment(shap_vals, available, seg_pdf[cat_col], top_n=top_n)

    if demand_class_col in seg_pdf.columns:
        shap_demand = shap_by_segment(shap_vals, available, seg_pdf[demand_class_col], top_n=top_n)

    # ── Export JSON ─────────────────────────────────────────────────────────
    out_paths = export_shap_for_frontend(shap_sum, shap_cat, shap_demand, output_dir)

    # ── MLflow artifact ─────────────────────────────────────────────────────
    if mlflow_run is not None:
        try:
            import mlflow
            if "shap_summary" in out_paths:
                mlflow.log_artifact(str(out_paths["shap_summary"]), artifact_path="shap")
            if "shap_by_segment" in out_paths:
                mlflow.log_artifact(str(out_paths["shap_by_segment"]), artifact_path="shap")
            mlflow.log_metric(
                "shap_top1_value",
                shap_sum["top_features"][0]["mean_abs_shap"] if shap_sum["top_features"] else 0.0,
            )
        except Exception as exc:
            logger.warning("MLflow SHAP logging failed: %s", exc)

    logger.info(
        "SHAP analysis complete: top feature = '%s' (%.4f)",
        shap_sum["top_features"][0]["feature"] if shap_sum["top_features"] else "N/A",
        shap_sum["top_features"][0]["mean_abs_shap"] if shap_sum["top_features"] else 0.0,
    )

    return {
        "shap_summary": shap_sum,
        "shap_by_category": shap_cat,
        "shap_by_demand_class": shap_demand,
        "output_paths": out_paths,
    }
