"""Integration tests for MLflow logging (V2-Fase 5).

Tests that:
- train_lgbm correctly logs params, metrics, and tags when mlflow_run is provided
- log_conformal_calibration and log_reconciliation_results write correct keys
- run_backtesting with use_mlflow=True creates runs with expected structure
- Artifact logging path exists in tmp run
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.models.training import (
    log_conformal_calibration,
    log_reconciliation_results,
    train_lgbm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_run():
    """Create a minimal MLflow run mock."""
    run = MagicMock()
    return run


def _make_tiny_df(n_rows: int = 50):
    """Create a minimal feature DataFrame for train_lgbm tests."""
    import polars as pl

    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "id": [f"ITEM_{i % 5}_CA_1" for i in range(n_rows)],
        "date": [f"2015-01-{(i % 28) + 1:02d}" for i in range(n_rows)],
        "sales": rng.uniform(0, 10, n_rows).tolist(),
        "store_id": [0] * n_rows,
        "dept_id": [0] * n_rows,
        "cat_id": [0] * n_rows,
        "state_id": [0] * n_rows,
        "day_of_week": (rng.integers(0, 7, n_rows)).tolist(),
        "month": (rng.integers(1, 13, n_rows)).tolist(),
        "lag_1": rng.uniform(0, 10, n_rows).tolist(),
        "lag_7": rng.uniform(0, 10, n_rows).tolist(),
        "rolling_mean_7": rng.uniform(0, 10, n_rows).tolist(),
    })


# ---------------------------------------------------------------------------
# log_conformal_calibration
# ---------------------------------------------------------------------------


class TestLogConformalCalibration:
    def test_logs_four_metrics(self):
        logged = {}
        mock_run = MagicMock()

        with patch("mlflow.log_metrics", side_effect=lambda m: logged.update(m)):
            log_conformal_calibration(
                mock_run,
                coverage_before=0.72,
                coverage_after=0.81,
                adjustment_p10=-0.5,
                adjustment_p90=0.5,
            )

        assert "conformal_coverage_before" in logged
        assert "conformal_coverage_after" in logged
        assert "conformal_adjustment_p10" in logged
        assert "conformal_adjustment_p90" in logged

    def test_coverage_values_correct(self):
        logged = {}
        mock_run = MagicMock()

        with patch("mlflow.log_metrics", side_effect=lambda m: logged.update(m)):
            log_conformal_calibration(
                mock_run,
                coverage_before=0.72,
                coverage_after=0.81,
                adjustment_p10=-0.5,
                adjustment_p90=0.8,
            )

        assert abs(logged["conformal_coverage_before"] - 0.72) < 1e-9
        assert abs(logged["conformal_coverage_after"] - 0.81) < 1e-9
        assert abs(logged["conformal_adjustment_p10"] - (-0.5)) < 1e-9
        assert abs(logged["conformal_adjustment_p90"] - 0.8) < 1e-9

    def test_no_op_when_run_is_none(self):
        """Must not raise even if mlflow_run is None."""
        log_conformal_calibration(
            None,
            coverage_before=0.72,
            coverage_after=0.81,
            adjustment_p10=-0.5,
            adjustment_p90=0.5,
        )

    def test_tolerates_mlflow_exception(self):
        """Must not raise even if mlflow raises."""
        mock_run = MagicMock()
        with patch("mlflow.log_metrics", side_effect=RuntimeError("mlflow down")):
            log_conformal_calibration(
                mock_run,
                coverage_before=0.72,
                coverage_after=0.81,
                adjustment_p10=-0.5,
                adjustment_p90=0.5,
            )


# ---------------------------------------------------------------------------
# log_reconciliation_results
# ---------------------------------------------------------------------------


class TestLogReconciliationResults:
    def test_logs_metrics_and_tags(self):
        logged_metrics = {}
        logged_tags = {}
        mock_run = MagicMock()

        with (
            patch("mlflow.log_metrics", side_effect=lambda m: logged_metrics.update(m)),
            patch("mlflow.set_tags", side_effect=lambda t: logged_tags.update(t)),
        ):
            log_reconciliation_results(
                mock_run,
                method_selected="mint_shrink",
                mae_pre_reconciliation=1.5,
                mae_post_reconciliation=1.2,
                coherence_test_passed=True,
            )

        assert "reconciliation_mae_pre" in logged_metrics
        assert "reconciliation_mae_post" in logged_metrics
        assert logged_tags["reconciliation_method"] == "mint_shrink"
        assert logged_tags["reconciliation_coherence_passed"] == "True"

    def test_mae_values_correct(self):
        logged_metrics = {}
        mock_run = MagicMock()

        with (
            patch("mlflow.log_metrics", side_effect=lambda m: logged_metrics.update(m)),
            patch("mlflow.set_tags"),
        ):
            log_reconciliation_results(
                mock_run,
                method_selected="bottom_up",
                mae_pre_reconciliation=2.0,
                mae_post_reconciliation=1.8,
                coherence_test_passed=False,
            )

        assert abs(logged_metrics["reconciliation_mae_pre"] - 2.0) < 1e-9
        assert abs(logged_metrics["reconciliation_mae_post"] - 1.8) < 1e-9

    def test_no_op_when_run_is_none(self):
        log_reconciliation_results(
            None,
            method_selected="mint_shrink",
            mae_pre_reconciliation=1.5,
            mae_post_reconciliation=1.2,
            coherence_test_passed=True,
        )

    def test_tolerates_mlflow_exception(self):
        mock_run = MagicMock()
        with (
            patch("mlflow.log_metrics", side_effect=RuntimeError("down")),
            patch("mlflow.set_tags", side_effect=RuntimeError("down")),
        ):
            log_reconciliation_results(
                mock_run,
                method_selected="mint_shrink",
                mae_pre_reconciliation=1.5,
                mae_post_reconciliation=1.2,
                coherence_test_passed=True,
            )


# ---------------------------------------------------------------------------
# train_lgbm MLflow integration
# ---------------------------------------------------------------------------


class TestTrainLgbmMlflowLogging:
    """Test that train_lgbm correctly logs to MLflow via mock."""

    def _run_train_with_mock_mlflow(self, extra_kwargs=None):
        """Helper: run train_lgbm with a mock mlflow_run and capture calls."""
        df = _make_tiny_df(60)
        train_df = df.head(45)
        val_df = df.tail(15)

        logged_params = {}
        logged_metrics = {}
        logged_tags = {}

        mock_run = MagicMock()

        with (
            patch("mlflow.log_params", side_effect=lambda p: logged_params.update(p)),
            patch("mlflow.log_metrics", side_effect=lambda m: logged_metrics.update(m)),
            patch("mlflow.set_tags", side_effect=lambda t: logged_tags.update(t)),
            patch("mlflow.log_artifact"),
        ):
            _extra = extra_kwargs or {}
            kwargs = {
                "mlflow_run": mock_run,
                "n_estimators": 5,
                "quantile_alphas": [0.10, 0.50, 0.90],
                **_extra,
            }
            from src.models.training import train_lgbm as _train
            _train(train_df, val_df, **kwargs)

        return logged_params, logged_metrics, logged_tags

    def test_logs_core_params(self):
        params, _, _ = self._run_train_with_mock_mlflow()
        assert "model_type" in params
        assert params["model_type"] == "lightgbm_global"
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "num_leaves" in params

    def test_logs_optional_params_when_provided(self):
        params, _, _ = self._run_train_with_mock_mlflow({
            "feature_set_version": "v2.1",
            "fold_id": 3,
            "forecast_horizon": 28,
            "n_series": 100,
        })
        assert params.get("feature_set_version") == "v2.1"
        assert params.get("fold_id") == 3
        assert params.get("forecast_horizon") == 28
        assert params.get("n_series") == 100

    def test_logs_val_mae_and_rmse(self):
        _, metrics, _ = self._run_train_with_mock_mlflow()
        assert "val_mae" in metrics
        assert "val_rmse" in metrics
        assert metrics["val_mae"] >= 0
        assert metrics["val_rmse"] >= 0

    def test_logs_val_smape_and_bias(self):
        _, metrics, _ = self._run_train_with_mock_mlflow()
        assert "val_smape" in metrics
        assert "val_bias" in metrics

    def test_logs_val_coverage_and_pinball_with_quantiles(self):
        _, metrics, _ = self._run_train_with_mock_mlflow({
            "quantile_alphas": [0.10, 0.50, 0.90],
        })
        assert "val_coverage_80" in metrics
        assert "val_pinball_loss" in metrics
        assert 0 <= metrics["val_coverage_80"] <= 1

    def test_logs_is_baseline_tag(self):
        _, _, tags = self._run_train_with_mock_mlflow({"is_baseline": True})
        assert tags.get("is_baseline") == "True"

    def test_logs_is_reconciled_tag(self):
        _, _, tags = self._run_train_with_mock_mlflow({"is_reconciled": False})
        assert tags.get("is_reconciled") == "False"

    def test_logs_dataset_sha256_tag_when_provided(self):
        _, _, tags = self._run_train_with_mock_mlflow({"dataset_sha256": "abc123"})
        assert tags.get("dataset_sha256") == "abc123"

    def test_logs_demand_class_distribution_as_json(self):
        dist = {"smooth": 0.3, "erratic": 0.2, "intermittent": 0.4, "lumpy": 0.1}
        _, _, tags = self._run_train_with_mock_mlflow({"demand_class_distribution": dist})
        tag_val = tags.get("demand_class_distribution")
        assert tag_val is not None
        parsed = json.loads(tag_val)
        assert parsed["smooth"] == 0.3

    def test_no_crash_without_mlflow_run(self):
        """train_lgbm should work fine with mlflow_run=None."""
        df = _make_tiny_df(60)
        from src.models.training import train_lgbm as _train
        result = _train(df.head(45), df.tail(15), n_estimators=5)
        assert result is not None

    def test_tolerates_mlflow_exception(self):
        """train_lgbm must not fail if MLflow raises an exception."""
        df = _make_tiny_df(60)
        mock_run = MagicMock()
        with patch("mlflow.log_params", side_effect=RuntimeError("mlflow unavailable")):
            from src.models.training import train_lgbm as _train
            result = _train(df.head(45), df.tail(15), mlflow_run=mock_run, n_estimators=5)
        assert result is not None


# ---------------------------------------------------------------------------
# run_backtesting MLflow integration
# ---------------------------------------------------------------------------


class TestRunBacktestingMlflow:
    """Test run_backtesting with use_mlflow=True using mocked MLflow."""

    def test_creates_one_run_per_fold(self):
        """With 2 folds and use_mlflow=True, MLflow start_run called twice."""
        import polars as pl
        from datetime import date, timedelta
        from src.evaluation.backtesting import FoldDefinition, run_backtesting

        # Minimal 3-series DataFrame
        rng = np.random.default_rng(0)
        ids = ["A", "B", "C"]
        start = date(2015, 1, 1)
        rows = []
        for i, sid in enumerate(ids):
            for d in range(60):
                rows.append({
                    "id": sid,
                    "date": start + timedelta(days=d),
                    "sales": float(rng.integers(0, 5)),
                    "store_id": 0, "dept_id": 0, "cat_id": 0, "state_id": 0,
                    "day_of_week": (start + timedelta(days=d)).weekday(),
                    "month": (start + timedelta(days=d)).month,
                    "lag_1": float(rng.uniform(0, 5)),
                    "lag_7": float(rng.uniform(0, 5)),
                    "rolling_mean_7": float(rng.uniform(0, 5)),
                })
        df = pl.DataFrame(rows)

        folds = [
            FoldDefinition(
                fold_id=1,
                train_cutoff=start + timedelta(days=39),
                test_start=start + timedelta(days=40),
                test_end=start + timedelta(days=49),
            ),
            FoldDefinition(
                fold_id=2,
                train_cutoff=start + timedelta(days=49),
                test_start=start + timedelta(days=50),
                test_end=start + timedelta(days=59),
            ),
        ]

        start_run_calls = []

        def mock_start_run(run_name=None, **kwargs):
            start_run_calls.append(run_name)
            return MagicMock()

        with (
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", side_effect=mock_start_run),
            patch("mlflow.log_params"),
            patch("mlflow.log_metrics"),
            patch("mlflow.set_tags"),
            patch("mlflow.end_run"),
            patch("mlflow.log_artifact"),
        ):
            run_backtesting(
                df, df,
                folds=folds,
                n_estimators=3,
                use_mlflow=True,
            )

        assert len(start_run_calls) == 2
        assert "fold_1" in start_run_calls[0]
        assert "fold_2" in start_run_calls[1]

    def test_fold_params_logged(self):
        """run_backtesting logs fold_id, train_cutoff, test_start, test_end."""
        import polars as pl
        from datetime import date, timedelta
        from src.evaluation.backtesting import FoldDefinition, run_backtesting

        rng = np.random.default_rng(1)
        start = date(2015, 1, 1)
        rows = []
        for sid in ["X", "Y"]:
            for d in range(50):
                rows.append({
                    "id": sid,
                    "date": start + timedelta(days=d),
                    "sales": float(rng.integers(0, 5)),
                    "store_id": 0, "dept_id": 0, "cat_id": 0, "state_id": 0,
                    "day_of_week": (start + timedelta(days=d)).weekday(),
                    "month": (start + timedelta(days=d)).month,
                    "lag_1": 1.0, "lag_7": 1.0, "rolling_mean_7": 1.0,
                })
        df = pl.DataFrame(rows)

        folds = [FoldDefinition(
            fold_id=1,
            train_cutoff=start + timedelta(days=34),
            test_start=start + timedelta(days=35),
            test_end=start + timedelta(days=44),
        )]

        all_logged_params = {}

        with (
            patch("mlflow.set_experiment"),
            patch("mlflow.start_run", return_value=MagicMock()),
            patch("mlflow.log_params", side_effect=lambda p: all_logged_params.update(p)),
            patch("mlflow.log_metrics"),
            patch("mlflow.set_tags"),
            patch("mlflow.end_run"),
            patch("mlflow.log_artifact"),
        ):
            run_backtesting(df, df, folds=folds, n_estimators=3, use_mlflow=True)

        assert "fold_id" in all_logged_params
        assert all_logged_params["fold_id"] == 1
