"""Integration smoke tests for the LightGBM training pipeline.

Tests train on a small synthetic feature DataFrame to verify:
1. Training completes without error.
2. Output model has the expected structure.
3. Predictions are non-negative.
4. Quantile monotonicity p10 ≤ p50 ≤ p90 holds.
5. Feature importance is returned.
6. Model can be saved and loaded from disk.
7. Backtesting fold definition dates are correct.
8. Segmented report has the expected format.
9. predict() output has the canonical columns.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.evaluation.backtesting import FOLDS, FoldDefinition, get_default_folds, summarize_backtesting
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.segmented_report import generate_segmented_report
from src.models.predict import enforce_monotonicity, predict
from src.models.training import (
    TrainedModels,
    get_feature_cols,
    prepare_xy,
    train_lgbm,
)


# ---------------------------------------------------------------------------
# Synthetic data builder
# ---------------------------------------------------------------------------

N_SERIES = 10
N_DAYS_TRAIN = 60
N_DAYS_VAL = 14


def _make_feature_df(
    n_series: int = N_SERIES,
    n_days_train: int = N_DAYS_TRAIN,
    n_days_val: int = N_DAYS_VAL,
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a minimal feature DataFrame for smoke testing.

    Produces train and val splits with:
    - A numeric target ('sales')
    - All required categorical and numeric feature columns
    """
    rng = np.random.default_rng(seed)
    stores = ["CA_1", "CA_2", "TX_1", "WI_1", "WI_2"]
    depts = ["FOODS_1", "FOODS_2", "HOBBIES_1", "HOUSEHOLD_1"]
    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    states = ["CA", "TX", "WI"]

    rows_train: list[dict] = []
    rows_val: list[dict] = []
    start = date(2015, 1, 1)

    for i in range(n_series):
        store = stores[i % len(stores)]
        dept = depts[i % len(depts)]
        cat = cats[i % len(cats)]
        state = store[:2]
        item_id = f"ITEM_{i}"
        series_id = f"{item_id}_{store}"

        for d_idx in range(n_days_train + n_days_val):
            dt = start + timedelta(days=d_idx)
            sales = max(0, int(rng.poisson(3.0) * (1 + 0.1 * np.sin(d_idx / 7))))
            row = {
                "id": series_id,
                "item_id": item_id,
                "dept_id": dept,
                "cat_id": cat,
                "store_id": store,
                "state_id": state,
                "date": dt,
                "sales": float(sales),
                # Lag features
                "lag_1": float(max(0, sales - 1)),
                "lag_7": float(max(0, sales - 1)),
                "lag_28": float(max(0, sales - 2)),
                # Rolling features
                "rolling_mean_7": float(rng.uniform(1, 5)),
                "rolling_std_7": float(rng.uniform(0, 2)),
                "rolling_mean_28": float(rng.uniform(1, 5)),
                "rolling_zero_pct_28": float(rng.uniform(0, 0.5)),
                "rolling_cv_28": float(rng.uniform(0, 1)),
                "ratio_mean_7_28": float(rng.uniform(0.5, 1.5)),
                "ratio_mean_28_91": float(rng.uniform(0.5, 1.5)),
                # Calendar features
                "day_of_week": dt.weekday(),
                "day_of_month": dt.day,
                "week_of_year": dt.isocalendar()[1],
                "month": dt.month,
                "quarter": (dt.month - 1) // 3 + 1,
                "is_weekend": int(dt.weekday() >= 5),
                "is_month_start": int(dt.day == 1),
                "is_month_end": int(dt.day == 28),
                # Price features
                "sell_price": float(rng.uniform(1.0, 10.0)),
                "price_delta_wow": float(rng.normal(0, 0.1)),
                # Intermittency features
                "pct_zero_last_28d": float(rng.uniform(0, 0.5)),
                "days_since_last_sale": int(rng.integers(0, 10)),
                "streak_zeros": int(rng.integers(0, 5)),
                "demand_intervals_mean": float(rng.uniform(1, 3)),
                "burstiness": float(rng.uniform(-0.5, 0.5)),
            }
            if d_idx < n_days_train:
                rows_train.append(row)
            else:
                rows_val.append(row)

    train_df = pl.DataFrame(rows_train).with_columns(pl.col("date").cast(pl.Date))
    val_df = pl.DataFrame(rows_val).with_columns(pl.col("date").cast(pl.Date))
    return train_df, val_df


# ---------------------------------------------------------------------------
# Training tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_models() -> TrainedModels:
    """Fixture: train a small LightGBM model once for all tests in module."""
    train_df, val_df = _make_feature_df()
    return train_lgbm(
        train_df,
        val_df,
        n_estimators=50,
        early_stopping_rounds=10,
        quantile_alphas=[0.10, 0.50, 0.90],
    )


def test_training_returns_trained_models(trained_models):
    assert isinstance(trained_models, TrainedModels)
    assert trained_models.point_model is not None


def test_training_has_quantile_models(trained_models):
    assert 0.10 in trained_models.quantile_models
    assert 0.50 in trained_models.quantile_models
    assert 0.90 in trained_models.quantile_models


def test_training_feature_cols_non_empty(trained_models):
    assert len(trained_models.feature_cols) > 0


def test_training_feature_importance_populated(trained_models):
    assert len(trained_models.feature_importance) > 0
    # All values should be non-negative
    assert all(v >= 0 for v in trained_models.feature_importance.values())


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predictions(trained_models) -> pl.DataFrame:
    _, val_df = _make_feature_df()
    return predict(trained_models, val_df)


def test_predict_returns_dataframe(predictions):
    assert isinstance(predictions, pl.DataFrame)


def test_predict_has_required_columns(predictions):
    required = {"id", "date", "forecast_p10", "forecast_p50", "forecast_p90"}
    assert required.issubset(set(predictions.columns))


def test_predict_non_negative(predictions):
    assert (predictions["forecast_p10"] >= 0).all()
    assert (predictions["forecast_p50"] >= 0).all()
    assert (predictions["forecast_p90"] >= 0).all()


def test_predict_monotonicity(predictions):
    """p10 ≤ p50 ≤ p90 must hold for all rows."""
    p10 = predictions["forecast_p10"].to_numpy()
    p50 = predictions["forecast_p50"].to_numpy()
    p90 = predictions["forecast_p90"].to_numpy()
    assert np.all(p10 <= p50 + 1e-9), "p10 > p50 violation"
    assert np.all(p50 <= p90 + 1e-9), "p50 > p90 violation"


def test_predict_correct_row_count(trained_models):
    _, val_df = _make_feature_df()
    preds = predict(trained_models, val_df)
    assert len(preds) == len(val_df)


def test_predict_empty_input(trained_models):
    train_df, _ = _make_feature_df()
    empty_df = train_df.filter(pl.lit(False))
    result = predict(trained_models, empty_df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Monotonicity enforcement unit tests
# ---------------------------------------------------------------------------


def test_enforce_monotonicity_no_violation():
    p10 = np.array([1.0, 2.0])
    p50 = np.array([2.0, 3.0])
    p90 = np.array([3.0, 4.0])
    r10, r50, r90 = enforce_monotonicity(p10, p50, p90)
    np.testing.assert_array_almost_equal(r10, [1.0, 2.0])
    np.testing.assert_array_almost_equal(r90, [3.0, 4.0])


def test_enforce_monotonicity_fixes_p10_above_p50():
    p10 = np.array([5.0])
    p50 = np.array([3.0])
    p90 = np.array([4.0])
    r10, r50, r90 = enforce_monotonicity(p10, p50, p90)
    assert r10[0] <= r50[0]


def test_enforce_monotonicity_fixes_p90_below_p50():
    p10 = np.array([1.0])
    p50 = np.array([5.0])
    p90 = np.array([2.0])
    r10, r50, r90 = enforce_monotonicity(p10, p50, p90)
    assert r90[0] >= r50[0]


def test_enforce_monotonicity_non_negative():
    p10 = np.array([-2.0])
    p50 = np.array([-1.0])
    p90 = np.array([0.0])
    r10, r50, r90 = enforce_monotonicity(p10, p50, p90)
    assert r10[0] >= 0.0
    assert r50[0] >= 0.0


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(trained_models, tmp_path):
    paths = trained_models.save(tmp_path / "models")
    assert (tmp_path / "models" / "model_point.lgb").exists()
    assert (tmp_path / "models" / "feature_importance.json").exists()

    loaded = TrainedModels.load(tmp_path / "models")
    assert loaded.point_model is not None
    assert loaded.feature_cols == trained_models.feature_cols
    assert set(loaded.quantile_models.keys()) == set(trained_models.quantile_models.keys())


def test_loaded_model_predicts(trained_models, tmp_path):
    trained_models.save(tmp_path / "models")
    loaded = TrainedModels.load(tmp_path / "models")
    _, val_df = _make_feature_df()
    preds = predict(loaded, val_df)
    assert len(preds) == len(val_df)
    assert (preds["forecast_p50"] >= 0).all()


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------


def test_get_feature_cols_excludes_metadata():
    train_df, _ = _make_feature_df()
    feat_cols = get_feature_cols(train_df)
    for excluded in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "date", "sales"]:
        assert excluded not in feat_cols


def test_get_feature_cols_non_empty():
    train_df, _ = _make_feature_df()
    assert len(get_feature_cols(train_df)) > 0


# ---------------------------------------------------------------------------
# Backtesting fold definitions
# ---------------------------------------------------------------------------


def test_fold_definitions_count():
    folds = get_default_folds()
    assert len(folds) == 5


def test_fold_ids_sequential():
    folds = get_default_folds()
    assert [f.fold_id for f in folds] == [1, 2, 3, 4, 5]


def test_fold_horizons_are_28():
    for fold in get_default_folds():
        assert fold.horizon == 28, f"Fold {fold.fold_id} horizon = {fold.horizon}"


def test_fold_train_cutoffs():
    """Verify exact dates from spec section 9.6."""
    folds = get_default_folds()
    expected_cutoffs = [
        date(2015, 12, 6),
        date(2016, 1, 3),
        date(2016, 1, 31),
        date(2016, 2, 28),
        date(2016, 3, 27),
    ]
    for fold, expected in zip(folds, expected_cutoffs):
        assert fold.train_cutoff == expected, (
            f"Fold {fold.fold_id}: got {fold.train_cutoff}, expected {expected}"
        )


def test_fold_test_starts_are_cutoff_plus_one():
    for fold in get_default_folds():
        assert fold.test_start == fold.train_cutoff + timedelta(days=1), (
            f"Fold {fold.fold_id}: test_start not cutoff+1"
        )


def test_consecutive_folds_are_contiguous():
    """Each fold's test_end + 1 = next fold's test_start."""
    folds = get_default_folds()
    for i in range(len(folds) - 1):
        assert folds[i].test_end + timedelta(days=1) == folds[i + 1].test_start, (
            f"Gap between fold {folds[i].fold_id} and {folds[i+1].fold_id}"
        )


# ---------------------------------------------------------------------------
# Segmented report
# ---------------------------------------------------------------------------


def test_segmented_report_has_overall_row():
    train_df, val_df = _make_feature_df()
    # Build a synthetic predictions frame
    preds = pl.DataFrame({
        "id": val_df["id"].to_list(),
        "date": val_df["date"].to_list(),
        "forecast_p10": [0.5] * len(val_df),
        "forecast_p50": [1.0] * len(val_df),
        "forecast_p90": [2.0] * len(val_df),
        "actual": val_df["sales"].to_list(),
    })
    report = generate_segmented_report(preds)
    assert "overall" in report["segment_col"].to_list()


def test_segmented_report_required_columns():
    _, val_df = _make_feature_df()
    preds = pl.DataFrame({
        "id": val_df["id"].to_list(),
        "date": val_df["date"].to_list(),
        "forecast_p50": val_df["sales"].to_list(),
        "actual": val_df["sales"].to_list(),
    })
    report = generate_segmented_report(preds)
    assert {"segment_col", "segment_value", "mae", "rmse"}.issubset(set(report.columns))


def test_segmented_report_mae_zero_for_perfect():
    _, val_df = _make_feature_df()
    preds = pl.DataFrame({
        "id": val_df["id"].to_list(),
        "date": val_df["date"].to_list(),
        "forecast_p50": val_df["sales"].to_list(),
        "actual": val_df["sales"].to_list(),
    })
    report = generate_segmented_report(preds)
    overall = report.filter(pl.col("segment_col") == "overall")
    assert overall["mae"][0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# summarize_backtesting
# ---------------------------------------------------------------------------


def test_summarize_backtesting_empty():
    assert summarize_backtesting([]) == {}


def test_summarize_backtesting_returns_mean_keys():
    from src.evaluation.backtesting import FoldResult
    results = [
        FoldResult(fold_id=1, metrics={"mae": 1.0, "rmse": 2.0}, predictions=pl.DataFrame()),
        FoldResult(fold_id=2, metrics={"mae": 3.0, "rmse": 4.0}, predictions=pl.DataFrame()),
    ]
    summary = summarize_backtesting(results)
    assert "mean_mae" in summary
    assert summary["mean_mae"] == pytest.approx(2.0)
    assert "mean_rmse" in summary
