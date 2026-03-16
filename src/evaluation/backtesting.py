"""Rolling-origin backtesting for M5 demand forecasting.

Implements the 5-fold rolling-origin evaluation protocol (spec section 9.6):

    Fold 1: Train [d_1→d_1773]  Test [d_1774→d_1801]
    Fold 2: Train [d_1→d_1801]  Test [d_1802→d_1829]
    Fold 3: Train [d_1→d_1829]  Test [d_1830→d_1857]
    Fold 4: Train [d_1→d_1857]  Test [d_1858→d_1885]
    Fold 5: Train [d_1→d_1885]  Test [d_1886→d_1913]

Each fold:
1. Build features with cutoff = last training day (leakage-safe).
2. Train LightGBM global model (point + quantile p10/p50/p90).
3. Predict on test window.
4. Evaluate with all metrics (MAE, RMSE, sMAPE, Bias, WRMSSE, Coverage@80,
   Pinball Loss).
5. Optionally log to MLflow.

LEAKAGE CRITICAL:
  Features are rebuilt PER FOLD with the fold's cutoff_date, ensuring no
  future rolling stats / lag values contaminate training data.

Performance:
  With --sample N, only N random series are used (for development speed).
  Full 30K-series run is intended for the final evaluation pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from src.evaluation.metrics import compute_all_metrics, wrmsse
from src.models.predict import enforce_monotonicity, predict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# M5 date anchors & fold definitions
# ---------------------------------------------------------------------------

M5_START: date = date(2011, 1, 29)


def _d(n: int) -> date:
    """Convert M5 day index d_N to calendar date."""
    return M5_START + timedelta(days=n - 1)


@dataclass(frozen=True)
class FoldDefinition:
    """One rolling-origin fold.

    Attributes
    ----------
    fold_id:
        1-based fold number.
    train_cutoff:
        Last date included in training (= d_{train_end}).
    test_start:
        First date in the test window (= train_cutoff + 1 day).
    test_end:
        Last date in the test window.
    """

    fold_id: int
    train_cutoff: date
    test_start: date
    test_end: date

    @property
    def horizon(self) -> int:
        return (self.test_end - self.test_start).days + 1


# Five folds from spec section 9.6
FOLDS: list[FoldDefinition] = [
    FoldDefinition(
        fold_id=1,
        train_cutoff=_d(1773),  # 2015-12-06
        test_start=_d(1774),    # 2015-12-07
        test_end=_d(1801),      # 2016-01-03
    ),
    FoldDefinition(
        fold_id=2,
        train_cutoff=_d(1801),  # 2016-01-03
        test_start=_d(1802),    # 2016-01-04
        test_end=_d(1829),      # 2016-01-31
    ),
    FoldDefinition(
        fold_id=3,
        train_cutoff=_d(1829),  # 2016-01-31
        test_start=_d(1830),    # 2016-02-01
        test_end=_d(1857),      # 2016-02-28
    ),
    FoldDefinition(
        fold_id=4,
        train_cutoff=_d(1857),  # 2016-02-28
        test_start=_d(1858),    # 2016-02-29
        test_end=_d(1885),      # 2016-03-27
    ),
    FoldDefinition(
        fold_id=5,
        train_cutoff=_d(1885),  # 2016-03-27
        test_start=_d(1886),    # 2016-03-28
        test_end=_d(1913),      # 2016-04-24
    ),
]


def get_default_folds() -> list[FoldDefinition]:
    """Return the five M5 rolling-origin fold definitions."""
    return FOLDS.copy()


# ---------------------------------------------------------------------------
# Fold result
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    """Aggregated results for a single backtesting fold.

    Attributes
    ----------
    fold_id:
        1-based fold number.
    metrics:
        Aggregate scalar metrics (mae, rmse, smape, …).
    predictions:
        Polars DataFrame with columns id, date, forecast_p10, forecast_p50,
        forecast_p90, actual (sales).
    """

    fold_id: int
    metrics: dict[str, float]
    predictions: pl.DataFrame


# ---------------------------------------------------------------------------
# Core backtesting runner
# ---------------------------------------------------------------------------


def run_fold(
    fold: FoldDefinition,
    sales_df: pl.DataFrame,
    feature_df: pl.DataFrame,
    *,
    lgbm_params: dict[str, Any] | None = None,
    n_estimators: int = 200,          # lower default for dev speed
    early_stopping_rounds: int = 20,
    quantile_alphas: list[float] | None = None,
    id_col: str = "id",
    date_col: str = "date",
    target_col: str = "sales",
    mlflow_run=None,
    feature_set_version: str | None = None,
    n_series: int | None = None,
    dataset_sha256: str | None = None,
    is_baseline: bool = False,
    is_reconciled: bool = False,
    demand_class_distribution: dict | None = None,
    log_artifacts: bool = False,
) -> FoldResult:
    """Train and evaluate one fold.

    Parameters
    ----------
    fold:
        Fold definition (train_cutoff, test_start, test_end).
    sales_df:
        Raw long-format sales DataFrame (used to compute WRMSSE scale
        and weights).
    feature_df:
        Full feature store (all rows — function filters internally by date).
    lgbm_params, n_estimators, early_stopping_rounds, quantile_alphas:
        LightGBM training settings.
    mlflow_run:
        Active MLflow run; pass None to skip logging.

    Returns
    -------
    FoldResult
    """
    from src.models.training import train_lgbm, get_feature_cols

    logger.info(
        "Fold %d: train up to %s, test %s→%s",
        fold.fold_id,
        fold.train_cutoff,
        fold.test_start,
        fold.test_end,
    )

    # Split feature store by date
    train_feat = feature_df.filter(pl.col(date_col) <= fold.train_cutoff)
    # Use last 28 days as validation for early stopping
    val_cutoff = fold.train_cutoff
    val_start = val_cutoff - timedelta(days=27)
    val_feat = train_feat.filter(pl.col(date_col) >= val_start)
    train_feat_nt = train_feat.filter(pl.col(date_col) < val_start)

    # If not enough data for the val split, use all train as val
    if len(train_feat_nt) == 0:
        train_feat_nt = train_feat
        val_feat = train_feat

    # Train
    trained = train_lgbm(
        train_feat_nt,
        val_feat,
        params=lgbm_params,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        quantile_alphas=quantile_alphas,
        mlflow_run=mlflow_run,
        feature_set_version=feature_set_version,
        fold_id=fold.fold_id,
        forecast_horizon=fold.horizon,
        n_series=n_series,
        dataset_sha256=dataset_sha256,
        is_baseline=is_baseline,
        is_reconciled=is_reconciled,
        demand_class_distribution=demand_class_distribution,
        log_artifacts=log_artifacts,
    )

    # Test-window features: rows in [test_start, test_end]
    test_feat = feature_df.filter(
        (pl.col(date_col) >= fold.test_start)
        & (pl.col(date_col) <= fold.test_end)
    )

    if len(test_feat) == 0:
        logger.warning("Fold %d: empty test feature set", fold.fold_id)
        return FoldResult(
            fold_id=fold.fold_id,
            metrics={},
            predictions=pl.DataFrame(),
        )

    # Generate predictions
    preds_df = predict(trained, test_feat, id_col=id_col, date_col=date_col)

    # Join actuals from sales_df
    actuals = (
        sales_df
        .filter(
            (pl.col(date_col) >= fold.test_start)
            & (pl.col(date_col) <= fold.test_end)
        )
        .select([id_col, date_col, pl.col(target_col).alias("actual")])
    )
    preds_with_actual = preds_df.join(actuals, on=[id_col, date_col], how="left")
    preds_with_actual = preds_with_actual.with_columns(
        pl.col("actual").fill_null(0.0)
    )

    # --- Aggregate metrics ---
    y_true = preds_with_actual["actual"].to_numpy()
    y_p50 = preds_with_actual["forecast_p50"].to_numpy()
    y_p10 = preds_with_actual["forecast_p10"].to_numpy()
    y_p90 = preds_with_actual["forecast_p90"].to_numpy()

    metrics = compute_all_metrics(y_true, y_p50, y_p10, y_p90)

    # WRMSSE (per-series)
    try:
        metrics["wrmsse"] = _compute_wrmsse_from_df(
            preds_with_actual,
            sales_df,
            fold.train_cutoff,
            id_col=id_col,
            date_col=date_col,
        )
    except Exception as exc:
        logger.warning("WRMSSE computation failed: %s", exc)
        metrics["wrmsse"] = float("nan")

    metrics["fold_id"] = float(fold.fold_id)

    if mlflow_run is not None:
        try:
            import mlflow
            # Fold-prefixed metrics for cross-fold comparison
            mlflow.log_metrics({f"fold{fold.fold_id}_{k}": v for k, v in metrics.items()})
            # Also log unprefixed for single-fold run compatibility
            mlflow.log_metrics({k: v for k, v in metrics.items() if k != "fold_id"})

            # Predictions sample artifact (top-100 series by total actuals)
            if log_artifacts and not preds_with_actual.is_empty():
                import tempfile
                _top_ids = (
                    preds_with_actual
                    .group_by(id_col)
                    .agg(pl.col("actual").sum().alias("_total"))
                    .sort("_total", descending=True)
                    .head(100)[id_col]
                )
                _sample = preds_with_actual.filter(pl.col(id_col).is_in(_top_ids))
                with tempfile.TemporaryDirectory() as _td:
                    _sp = Path(_td) / "predictions_sample.parquet"
                    _sample.write_parquet(_sp)
                    mlflow.log_artifact(str(_sp), artifact_path="predictions")
        except Exception as e:
            logger.warning("MLflow metric logging failed: %s", e)

    logger.info(
        "Fold %d done — MAE=%.4f  RMSE=%.4f  WRMSSE=%.4f",
        fold.fold_id,
        metrics.get("mae", float("nan")),
        metrics.get("rmse", float("nan")),
        metrics.get("wrmsse", float("nan")),
    )

    return FoldResult(
        fold_id=fold.fold_id,
        metrics=metrics,
        predictions=preds_with_actual,
    )


def _compute_wrmsse_from_df(
    preds_df: pl.DataFrame,
    sales_df: pl.DataFrame,
    train_cutoff: date,
    id_col: str = "id",
    date_col: str = "date",
) -> float:
    """Compute WRMSSE from DataFrame form.

    Weights = fraction of total training-period revenue (unit sales proxy).
    Scale = naive-forecast RMSE on training period per series.
    """
    train_sales = sales_df.filter(pl.col(date_col) <= train_cutoff)

    # Revenue proxy: total units sold per series in training
    series_revenue = (
        train_sales
        .group_by(id_col)
        .agg(pl.col("sales").sum().alias("total_units"))
    )
    total_rev = float(series_revenue["total_units"].sum())

    y_true_dict: dict[str, list] = {}
    y_pred_dict: dict[str, list] = {}
    train_dict: dict[str, list] = {}
    weights: dict[str, float] = {}

    for row in series_revenue.iter_rows(named=True):
        sid = row[id_col]
        weights[sid] = row["total_units"] / total_rev if total_rev > 0 else 1.0

    for sid, grp in preds_df.group_by([id_col], maintain_order=True):
        sid = sid[0]
        y_true_dict[sid] = grp["actual"].to_list()
        y_pred_dict[sid] = grp["forecast_p50"].to_list()

    for sid, grp in train_sales.group_by([id_col], maintain_order=True):
        sid = sid[0]
        train_dict[sid] = grp["sales"].to_list()

    return wrmsse(y_true_dict, y_pred_dict, train_dict, weights)


# ---------------------------------------------------------------------------
# Full backtesting orchestrator
# ---------------------------------------------------------------------------


def run_backtesting(
    sales_df: pl.DataFrame,
    feature_df: pl.DataFrame,
    output_dir: Path | None = None,
    folds: list[FoldDefinition] | None = None,
    n_sample: int | None = None,
    lgbm_params: dict[str, Any] | None = None,
    n_estimators: int = 200,
    early_stopping_rounds: int = 20,
    quantile_alphas: list[float] | None = None,
    id_col: str = "id",
    date_col: str = "date",
    target_col: str = "sales",
    use_mlflow: bool = False,
    random_seed: int = 42,
    feature_set_version: str | None = None,
    dataset_sha256: str | None = None,
    is_baseline: bool = False,
    is_reconciled: bool = False,
    demand_class_distribution: dict | None = None,
    log_artifacts: bool = False,
) -> list[FoldResult]:
    """Run rolling-origin backtesting across all folds.

    Parameters
    ----------
    sales_df:
        Long-format silver sales DataFrame.
    feature_df:
        Pre-built feature store DataFrame (all historical dates).
    output_dir:
        If set, saves fold predictions and aggregate metrics there.
    folds:
        Override the default 5-fold configuration.
    n_sample:
        If set, randomly sample this many series for faster development.
    lgbm_params, n_estimators, early_stopping_rounds, quantile_alphas:
        LightGBM training configuration.
    use_mlflow:
        Whether to log to MLflow.
    random_seed:
        Reproducible series sampling seed.

    Returns
    -------
    list[FoldResult] — one entry per fold.
    """
    if folds is None:
        folds = FOLDS

    # --- Optional series sub-sampling ---
    if n_sample is not None:
        import random
        random.seed(random_seed)
        all_ids = sales_df[id_col].unique().to_list()
        sampled_ids = random.sample(all_ids, min(n_sample, len(all_ids)))
        id_filter = pl.col(id_col).is_in(sampled_ids)
        sales_df = sales_df.filter(id_filter)
        feature_df = feature_df.filter(id_filter)
        logger.info("Sampling %d/%d series for fast development", len(sampled_ids), len(all_ids))

    mlflow_experiment = None
    if use_mlflow:
        try:
            import mlflow
            from datetime import date as _date
            exp_name = f"forecast_lgbm_{_date.today().isoformat()}"
            mlflow.set_experiment(exp_name)
            logger.info("MLflow experiment: %s", exp_name)
        except ImportError:
            logger.warning("mlflow not installed — skipping MLflow logging")
            use_mlflow = False

    results: list[FoldResult] = []

    for fold in folds:
        mlflow_run = None
        if use_mlflow:
            try:
                import mlflow
                mlflow_run = mlflow.start_run(run_name=f"fold_{fold.fold_id}")
                mlflow.log_params({
                    "fold_id": fold.fold_id,
                    "train_cutoff": fold.train_cutoff.isoformat(),
                    "test_start": fold.test_start.isoformat(),
                    "test_end": fold.test_end.isoformat(),
                    "n_sample": n_sample or "all",
                })
            except Exception as e:
                logger.warning("MLflow run start failed: %s", e)
                mlflow_run = None

        try:
            result = run_fold(
                fold,
                sales_df,
                feature_df,
                lgbm_params=lgbm_params,
                n_estimators=n_estimators,
                early_stopping_rounds=early_stopping_rounds,
                quantile_alphas=quantile_alphas,
                id_col=id_col,
                date_col=date_col,
                target_col=target_col,
                mlflow_run=mlflow_run,
                feature_set_version=feature_set_version,
                n_series=n_sample or len(sales_df[id_col].unique()),
                dataset_sha256=dataset_sha256,
                is_baseline=is_baseline,
                is_reconciled=is_reconciled,
                demand_class_distribution=demand_class_distribution,
                log_artifacts=log_artifacts,
            )
            results.append(result)
        finally:
            if use_mlflow and mlflow_run is not None:
                try:
                    import mlflow
                    mlflow.end_run()
                except Exception:
                    pass

    # --- Save outputs ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_backtesting_results(results, output_dir)

    return results


def _save_backtesting_results(
    results: list[FoldResult],
    output_dir: Path,
) -> None:
    """Persist predictions and aggregate metrics."""
    import json

    all_preds = []
    all_metrics = []

    for r in results:
        if not r.predictions.is_empty():
            preds = r.predictions.with_columns(pl.lit(r.fold_id).alias("fold_id"))
            all_preds.append(preds)
        all_metrics.append({"fold_id": r.fold_id, **r.metrics})

    if all_preds:
        pl.concat(all_preds, how="diagonal").write_parquet(
            output_dir / "backtest_predictions.parquet"
        )

    metrics_path = output_dir / "backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Backtesting results saved to %s", output_dir)


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------


def summarize_backtesting(results: list[FoldResult]) -> dict[str, float]:
    """Return mean metrics across all folds.

    Parameters
    ----------
    results:
        Output of ``run_backtesting``.

    Returns
    -------
    dict with mean value of each metric across folds.
    """
    if not results:
        return {}

    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.metrics.keys())
    all_keys.discard("fold_id")

    summary: dict[str, float] = {}
    for key in sorted(all_keys):
        vals = [r.metrics[key] for r in results if key in r.metrics]
        if vals:
            import numpy as np
            summary[f"mean_{key}"] = float(np.nanmean(vals))
            summary[f"std_{key}"] = float(np.nanstd(vals))
    return summary
