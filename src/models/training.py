"""LightGBM global model training for demand forecasting.

Trains a single LightGBM model across all SKU-store series (global model
strategy — ADR-002).  Supports:

- Point forecast  : objective="regression", metric="mae"
- Quantile models : objective="quantile" for α ∈ {0.10, 0.50, 0.90}

Hyperparameters from spec section 9.2:
  n_estimators=2000, learning_rate=0.03, num_leaves=255,
  min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
  reg_alpha=0.1, reg_lambda=0.1, max_bin=255,
  early_stopping_rounds=100.

Categorical features (label-encoded before training):
  store_id, dept_id, cat_id, state_id, day_of_week, month.

LEAKAGE PROTOCOL:
  The feature DataFrame passed to ``train_lgbm`` must have been built with
  a cutoff_date matching the fold's last training day (no future target in
  rolling stats, no future prices — verified by ``leakage_guard``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COL = "sales"
DATE_COL = "date"

# Columns that are identifiers / metadata (never used as features)
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Categorical features that LightGBM handles with special treatment
CAT_FEATURES = ["store_id", "dept_id", "cat_id", "state_id", "day_of_week", "month"]

# Default LightGBM hyperparameters (spec section 9.2)
DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "max_bin": 255,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

QUANTILE_ALPHAS = [0.10, 0.50, 0.90]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return the list of feature columns (excludes metadata, date, target).

    Parameters
    ----------
    df:
        Feature store DataFrame.

    Returns
    -------
    list[str]
        Sorted list of columns to use as model input features.
    """
    exclude = set(ID_COLS) | {DATE_COL, TARGET_COL}
    return [c for c in df.columns if c not in exclude]


def _encode_categoricals(df: "pd.DataFrame", cat_cols: list[str]) -> "pd.DataFrame":
    """Label-encode categorical columns in-place (returns copy).

    LightGBM requires categorical features to be stored as pandas Categorical
    dtype.  Unknown categories in val/test are mapped to -1.
    """
    import pandas as pd

    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def prepare_xy(
    df: pl.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str] | None = None,
) -> tuple["pd.DataFrame", "pd.Series"]:
    """Extract X (features) and y (target) as pandas objects for LightGBM.

    Parameters
    ----------
    df:
        Feature store DataFrame (Polars).
    feature_cols:
        Columns to use as features.
    cat_cols:
        Subset of feature_cols to treat as categoricals.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if cat_cols is None:
        cat_cols = [c for c in CAT_FEATURES if c in feature_cols]

    pdf = df.select(feature_cols + [TARGET_COL]).to_pandas()
    X = pdf[feature_cols]
    y = pdf[TARGET_COL].astype(np.float64)
    X = _encode_categoricals(X, cat_cols)
    return X, y


# ---------------------------------------------------------------------------
# Model dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainedModels:
    """Container for a trained point model + optional quantile models.

    Attributes
    ----------
    point_model:
        LightGBM Booster for point (MAE) regression.
    quantile_models:
        Dict of alpha → LightGBM Booster.
    feature_cols:
        Ordered list of feature column names.
    cat_cols:
        Categorical column names.
    feature_importance:
        Feature importance (gain) from the point model.
    """

    point_model: lgb.Booster
    quantile_models: dict[float, lgb.Booster] = field(default_factory=dict)
    feature_cols: list[str] = field(default_factory=list)
    cat_cols: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def save(self, output_dir: Path) -> dict[str, Path]:
        """Persist models and metadata to ``output_dir``.

        Returns dict of artifact name → path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        point_path = output_dir / "model_point.lgb"
        self.point_model.save_model(str(point_path))
        paths["point_model"] = point_path

        for alpha, model in self.quantile_models.items():
            q_str = str(int(alpha * 100))
            q_path = output_dir / f"model_q{q_str}.lgb"
            model.save_model(str(q_path))
            paths[f"quantile_model_p{q_str}"] = q_path

        # Feature importance JSON
        fi_path = output_dir / "feature_importance.json"
        with open(fi_path, "w") as f:
            json.dump(self.feature_importance, f, indent=2)
        paths["feature_importance"] = fi_path

        # Metadata
        meta_path = output_dir / "model_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "feature_cols": self.feature_cols,
                "cat_cols": self.cat_cols,
                "quantile_alphas": sorted(self.quantile_models.keys()),
            }, f, indent=2)
        paths["meta"] = meta_path

        return paths

    @classmethod
    def load(cls, model_dir: Path) -> "TrainedModels":
        """Load models from a directory previously created by ``save``."""
        meta_path = model_dir / "model_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)

        point_model = lgb.Booster(model_file=str(model_dir / "model_point.lgb"))

        quantile_models: dict[float, lgb.Booster] = {}
        for alpha in meta.get("quantile_alphas", []):
            q_str = str(int(alpha * 100))
            q_path = model_dir / f"model_q{q_str}.lgb"
            if q_path.exists():
                quantile_models[alpha] = lgb.Booster(model_file=str(q_path))

        fi_path = model_dir / "feature_importance.json"
        feature_importance: dict[str, float] = {}
        if fi_path.exists():
            with open(fi_path) as f:
                feature_importance = json.load(f)

        return cls(
            point_model=point_model,
            quantile_models=quantile_models,
            feature_cols=meta["feature_cols"],
            cat_cols=meta["cat_cols"],
            feature_importance=feature_importance,
        )


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------


def _train_single(
    X_train: "pd.DataFrame",
    y_train: "pd.Series",
    X_val: "pd.DataFrame",
    y_val: "pd.Series",
    cat_cols: list[str],
    params: dict[str, Any],
    n_estimators: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    """Train one LightGBM model with early stopping."""
    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        categorical_feature=[c for c in cat_cols if c in X_train.columns] or "auto",
        free_raw_data=False,
    )
    lgb_val = lgb.Dataset(
        X_val,
        y_val,
        reference=lgb_train,
        free_raw_data=False,
    )
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=-1),  # silence per-iteration logging
    ]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=n_estimators,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )
    return model


def train_lgbm(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    params: dict[str, Any] | None = None,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 100,
    quantile_alphas: list[float] | None = None,
    mlflow_run=None,
    feature_set_version: str | None = None,
    fold_id: int | None = None,
    forecast_horizon: int = 28,
    n_series: int | None = None,
    dataset_sha256: str | None = None,
    is_baseline: bool = False,
    is_reconciled: bool = False,
    demand_class_distribution: dict | None = None,
    log_artifacts: bool = False,
) -> TrainedModels:
    """Train the LightGBM global model (point + quantile).

    Parameters
    ----------
    train_df:
        Feature store rows for the training window (Polars DataFrame).
    val_df:
        Feature store rows for early-stopping validation (Polars DataFrame).
    feature_cols:
        Override the auto-detected feature column list.
    cat_cols:
        Override the categorical feature list.
    params:
        Override the default LightGBM hyperparameters.
    n_estimators:
        Maximum number of boosting rounds.
    early_stopping_rounds:
        Patience for early stopping on the validation set.
    quantile_alphas:
        Quantile levels to train. Default: [0.10, 0.50, 0.90].
    mlflow_run:
        Active MLflow run for logging. Pass None to skip MLflow.

    Returns
    -------
    TrainedModels
    """
    if feature_cols is None:
        feature_cols = get_feature_cols(train_df)
    if cat_cols is None:
        cat_cols = [c for c in CAT_FEATURES if c in feature_cols]
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if quantile_alphas is None:
        quantile_alphas = QUANTILE_ALPHAS

    logger.info(
        "Training LightGBM: %d train rows, %d val rows, %d features",
        len(train_df),
        len(val_df),
        len(feature_cols),
    )

    X_train, y_train = prepare_xy(train_df, feature_cols, cat_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols, cat_cols)

    # --- Point model (regression / MAE) ---
    point_params = {**params, "objective": "regression", "metric": "mae"}
    point_model = _train_single(
        X_train, y_train, X_val, y_val,
        cat_cols, point_params, n_estimators, early_stopping_rounds,
    )

    # Feature importance (gain) from point model
    fi = dict(zip(
        point_model.feature_name(),
        point_model.feature_importance(importance_type="gain").tolist(),
    ))

    # --- Quantile models ---
    quantile_models: dict[float, lgb.Booster] = {}
    for alpha in quantile_alphas:
        q_params = {
            **params,
            "objective": "quantile",
            "metric": "quantile",
            "alpha": alpha,
        }
        q_model = _train_single(
            X_train, y_train, X_val, y_val,
            cat_cols, q_params, n_estimators, early_stopping_rounds,
        )
        quantile_models[alpha] = q_model

    trained = TrainedModels(
        point_model=point_model,
        quantile_models=quantile_models,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
        feature_importance=fi,
    )

    # --- MLflow logging ---
    if mlflow_run is not None:
        try:
            import mlflow
            import json as _json
            import tempfile

            # --- Params ---
            _log_params: dict[str, Any] = {
                "model_type": "lightgbm_global",
                "n_estimators": n_estimators,
                "learning_rate": params.get("learning_rate", DEFAULT_PARAMS["learning_rate"]),
                "num_leaves": params.get("num_leaves", DEFAULT_PARAMS["num_leaves"]),
                "n_features": len(feature_cols),
                "quantile_alphas": str(quantile_alphas),
                "forecast_horizon": forecast_horizon,
            }
            if feature_set_version is not None:
                _log_params["feature_set_version"] = feature_set_version
            if fold_id is not None:
                _log_params["fold_id"] = fold_id
            if n_series is not None:
                _log_params["n_series"] = n_series
            mlflow.log_params(_log_params)

            # --- Val metrics ---
            val_pred = point_model.predict(X_val.values)
            from src.evaluation.metrics import (
                mae as _mae, rmse as _rmse, smape as _smape,
                bias as _bias, coverage_80 as _cov80, mean_pinball_loss as _pinball,
            )
            _val_metrics: dict[str, float] = {
                "val_mae": float(_mae(y_val.values, val_pred)),
                "val_rmse": float(_rmse(y_val.values, val_pred)),
                "val_smape": float(_smape(y_val.values, val_pred)),
                "val_bias": float(_bias(y_val.values, val_pred)),
                "best_iteration": float(point_model.best_iteration),
            }
            if 0.10 in trained.quantile_models and 0.90 in trained.quantile_models:
                _vp10 = trained.quantile_models[0.10].predict(X_val.values)
                _vp90 = trained.quantile_models[0.90].predict(X_val.values)
                _vp50 = trained.quantile_models.get(0.50, point_model).predict(X_val.values)
                _val_metrics["val_coverage_80"] = float(_cov80(y_val.values, _vp10, _vp90))
                _val_metrics["val_pinball_loss"] = float(_pinball(y_val.values, _vp10, _vp50, _vp90))
            mlflow.log_metrics(_val_metrics)

            # --- Tags ---
            _tags: dict[str, str] = {
                "is_baseline": str(is_baseline),
                "is_reconciled": str(is_reconciled),
            }
            if dataset_sha256 is not None:
                _tags["dataset_sha256"] = dataset_sha256
            if demand_class_distribution is not None:
                _tags["demand_class_distribution"] = _json.dumps(demand_class_distribution)
            mlflow.set_tags(_tags)

            # --- Artifacts ---
            if log_artifacts:
                with tempfile.TemporaryDirectory() as _tmpdir:
                    _artifact_paths = trained.save(Path(_tmpdir))
                    for _aname, _apath in _artifact_paths.items():
                        mlflow.log_artifact(str(_apath), artifact_path="models")

        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

    return trained


# ---------------------------------------------------------------------------
# MLflow helpers for post-training logging
# ---------------------------------------------------------------------------


def log_conformal_calibration(
    mlflow_run,
    coverage_before: float,
    coverage_after: float,
    adjustment_p10: float,
    adjustment_p90: float,
) -> None:
    """Log conformal calibration results to an active MLflow run.

    Parameters
    ----------
    mlflow_run:
        Active MLflow run context.
    coverage_before:
        Empirical coverage before conformal adjustment.
    coverage_after:
        Empirical coverage after conformal adjustment (target: 0.80).
    adjustment_p10:
        Additive shift applied to the p10 quantile.
    adjustment_p90:
        Additive shift applied to the p90 quantile.
    """
    if mlflow_run is None:
        return
    try:
        import mlflow
        mlflow.log_metrics({
            "conformal_coverage_before": coverage_before,
            "conformal_coverage_after": coverage_after,
            "conformal_adjustment_p10": adjustment_p10,
            "conformal_adjustment_p90": adjustment_p90,
        })
    except Exception as e:
        logger.warning("MLflow conformal logging failed: %s", e)


def log_reconciliation_results(
    mlflow_run,
    method_selected: str,
    mae_pre_reconciliation: float,
    mae_post_reconciliation: float,
    coherence_test_passed: bool,
) -> None:
    """Log hierarchical reconciliation results to an active MLflow run.

    Parameters
    ----------
    mlflow_run:
        Active MLflow run context.
    method_selected:
        Reconciliation method name (e.g. "mint_shrink", "bottom_up").
    mae_pre_reconciliation:
        MAE before reconciliation.
    mae_post_reconciliation:
        MAE after reconciliation (should be ≤ pre for good reconciliation).
    coherence_test_passed:
        Whether the 4 coherence checks all passed post-reconciliation.
    """
    if mlflow_run is None:
        return
    try:
        import mlflow
        mlflow.log_metrics({
            "reconciliation_mae_pre": mae_pre_reconciliation,
            "reconciliation_mae_post": mae_post_reconciliation,
        })
        mlflow.set_tags({
            "reconciliation_method": method_selected,
            "reconciliation_coherence_passed": str(coherence_test_passed),
        })
    except Exception as e:
        logger.warning("MLflow reconciliation logging failed: %s", e)
