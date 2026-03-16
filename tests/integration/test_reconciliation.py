"""Integration tests for src/reconciliation — V2-Fase 1.

Smoke tests verifying that the full reconciliation pipeline runs end-to-end
with a small synthetic hierarchy (4 SKU-Store series) using real
``hierarchicalforecast`` reconcilers (BottomUp, TopDown, MinTrace).

These tests do NOT require the M5 dataset — they use fully synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.evaluate_reconciliation import run_coherence_tests
from src.reconciliation.hierarchy import build_hierarchy_matrix
from src.reconciliation.reconciler import (
    METHOD_COL,
    aggregate_base_forecasts,
    reconcile_all_methods,
    reconcile_forecasts,
    reconcile_mint_sub_hierarchy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog() -> pd.DataFrame:
    rows = [
        ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
        ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "TX_1", "TX"),
    ]
    return pd.DataFrame(
        rows, columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    )


@pytest.fixture
def hm(catalog):
    return build_hierarchy_matrix(catalog)


def _make_bottom_forecasts(catalog: pd.DataFrame, n_days: int = 5) -> pd.DataFrame:
    """Bottom-level forecasts: each series = 1.0 per day."""
    rows = []
    for _, row in catalog.iterrows():
        uid = f"{row['item_id']}_{row['store_id']}"
        for ds in range(n_days):
            rows.append({"unique_id": uid, "ds": ds, "forecast_p50": 1.0})
    return pd.DataFrame(rows)


def _make_y_df(catalog: pd.DataFrame, hm, n_days: int = 5) -> pd.DataFrame:
    """In-sample actuals + model predictions for MinTrace."""
    bottom_df = _make_bottom_forecasts(catalog, n_days)
    fc_all = aggregate_base_forecasts(
        bottom_df, catalog, hm.tags, model_col="forecast_p50"
    )
    # Y_df = actuals (y = true) + model prediction column
    y_df = fc_all.rename(columns={"forecast_p50": "Forecast"}).copy()
    y_df["y"] = y_df["Forecast"] + np.random.default_rng(0).normal(0, 0.1, len(y_df))
    return y_df


# ---------------------------------------------------------------------------
# aggregate_base_forecasts
# ---------------------------------------------------------------------------


class TestAggregateBaseForecasts:
    def test_produces_all_levels(self, catalog, hm):
        bottom = _make_bottom_forecasts(catalog)
        agg = aggregate_base_forecasts(bottom, catalog, hm.tags)
        # Should contain Total series
        assert "Total" in agg["unique_id"].values

    def test_total_equals_sum_of_bottom(self, catalog, hm):
        n_days = 3
        bottom = _make_bottom_forecasts(catalog, n_days)
        agg = aggregate_base_forecasts(bottom, catalog, hm.tags)
        n_bottom = len(catalog)
        for ds in range(n_days):
            total_val = float(
                agg[(agg["unique_id"] == "Total") & (agg["ds"] == ds)]["forecast_p50"]
            )
            assert total_val == pytest.approx(float(n_bottom), abs=1e-6)

    def test_state_aggregates_correct(self, catalog, hm):
        n_days = 2
        bottom = _make_bottom_forecasts(catalog, n_days)
        agg = aggregate_base_forecasts(bottom, catalog, hm.tags)
        # CA has 2 bottom series → CA total = 2.0
        ca_rows = agg[agg["unique_id"] == "CA"]
        assert not ca_rows.empty
        for _, row in ca_rows.iterrows():
            assert row["forecast_p50"] == pytest.approx(2.0, abs=1e-6)

    def test_no_null_values(self, catalog, hm):
        bottom = _make_bottom_forecasts(catalog)
        agg = aggregate_base_forecasts(bottom, catalog, hm.tags)
        assert not agg["forecast_p50"].isnull().any()


# ---------------------------------------------------------------------------
# reconcile_forecasts — individual methods
# ---------------------------------------------------------------------------


class TestReconcileForecasts:
    def _prepare(self, catalog, hm, n_days=3):
        bottom = _make_bottom_forecasts(catalog, n_days)
        fc_all = aggregate_base_forecasts(
            bottom, catalog, hm.tags, model_col="forecast_p50"
        ).rename(columns={"forecast_p50": "Forecast"})
        return fc_all

    def test_bottom_up_runs(self, catalog, hm):
        fc_all = self._prepare(catalog, hm)
        out = reconcile_forecasts(
            fc_all, hm.S_df, hm.tags, method="bu", model_col="Forecast"
        )
        assert "Forecast/BottomUp" in out.columns

    def test_top_down_runs(self, catalog, hm):
        fc_all = self._prepare(catalog, hm)
        y_df = _make_y_df(catalog, hm)
        out = reconcile_forecasts(
            fc_all, hm.S_df, hm.tags, method="td", Y_df=y_df, model_col="Forecast"
        )
        assert "Forecast/TopDown_method-average_proportions" in out.columns

    def test_mint_shrink_runs(self, catalog, hm):
        fc_all = self._prepare(catalog, hm)
        y_df = _make_y_df(catalog, hm)
        out = reconcile_forecasts(
            fc_all, hm.S_df, hm.tags, method="mint_shrink", Y_df=y_df, model_col="Forecast"
        )
        assert "Forecast/MinTrace_method-mint_shrink" in out.columns

    def test_unknown_method_raises(self, catalog, hm):
        fc_all = self._prepare(catalog, hm)
        with pytest.raises(ValueError, match="Unknown method"):
            reconcile_forecasts(fc_all, hm.S_df, hm.tags, method="bad_method")  # type: ignore

    def test_bu_preserves_bottom_values(self, catalog, hm):
        """BottomUp should NOT change bottom-level forecasts."""
        fc_all = self._prepare(catalog, hm)
        out = reconcile_forecasts(
            fc_all, hm.S_df, hm.tags, method="bu", model_col="Forecast"
        )
        bottom_ids = set(hm.tags["SKU-Store"])
        bottom_out = out[out["unique_id"].isin(bottom_ids)]
        max_dev = float(bottom_out["Forecast/BottomUp"].sub(1.0).abs().max())
        assert max_dev < 1e-6, f"BottomUp changed bottom values (max dev={max_dev})"

    def test_bu_total_equals_sum_bottom(self, catalog, hm):
        """BottomUp total = sum(bottom) = n_bottom × 1.0."""
        fc_all = self._prepare(catalog, hm)
        out = reconcile_forecasts(
            fc_all, hm.S_df, hm.tags, method="bu", model_col="Forecast"
        )
        n_bottom = len(hm.tags["SKU-Store"])
        total_rows = out[out["unique_id"] == "Total"]
        assert not total_rows.empty
        for _, row in total_rows.iterrows():
            assert row["Forecast/BottomUp"] == pytest.approx(float(n_bottom), abs=1e-4)

    def test_output_covers_all_series(self, catalog, hm):
        fc_all = self._prepare(catalog, hm)
        out = reconcile_forecasts(
            fc_all, hm.S_df, hm.tags, method="bu", model_col="Forecast"
        )
        expected_ids = set(hm.S_df["unique_id"])
        actual_ids = set(out["unique_id"])
        assert expected_ids == actual_ids


# ---------------------------------------------------------------------------
# reconcile_all_methods
# ---------------------------------------------------------------------------


class TestReconcileAllMethods:
    def test_all_three_methods_present(self, catalog, hm):
        bottom = _make_bottom_forecasts(catalog)
        fc_all = aggregate_base_forecasts(
            bottom, catalog, hm.tags
        ).rename(columns={"forecast_p50": "Forecast"})
        y_df = _make_y_df(catalog, hm)
        out = reconcile_all_methods(fc_all, hm.S_df, hm.tags, Y_df=y_df, model_col="Forecast")
        assert "Forecast/BottomUp" in out.columns
        assert "Forecast/TopDown_method-average_proportions" in out.columns
        assert "Forecast/MinTrace_method-mint_shrink" in out.columns

    def test_without_y_df_only_bu(self, catalog, hm):
        """Without Y_df, only BottomUp runs (TopDown and MinT require actuals)."""
        bottom = _make_bottom_forecasts(catalog)
        fc_all = aggregate_base_forecasts(
            bottom, catalog, hm.tags
        ).rename(columns={"forecast_p50": "Forecast"})
        out = reconcile_all_methods(fc_all, hm.S_df, hm.tags, Y_df=None, model_col="Forecast")
        assert "Forecast/BottomUp" in out.columns
        assert "Forecast/MinTrace_method-mint_shrink" not in out.columns
        assert "Forecast/TopDown_method-average_proportions" not in out.columns


# ---------------------------------------------------------------------------
# reconcile_mint_sub_hierarchy
# ---------------------------------------------------------------------------


class TestReconcileMintSubHierarchy:
    def test_runs_on_bottom_df(self, catalog, hm):
        bottom = _make_bottom_forecasts(catalog, n_days=4)
        y_df = _make_y_df(catalog, hm, n_days=4)
        # Build Y_df at bottom level
        y_bottom = y_df[y_df["unique_id"].isin(set(hm.tags["SKU-Store"]))].copy()
        y_bottom = y_bottom.rename(columns={"Forecast": "forecast_p50"})
        y_bottom["y"] = y_bottom["forecast_p50"] + 0.1

        out = reconcile_mint_sub_hierarchy(
            bottom.rename(columns={"forecast_p50": "forecast_p50"}),
            catalog,
            Y_df=y_bottom,
            model_col="forecast_p50",
            group_cols=["store_id"],
        )
        assert not out.empty
        rec_col = "forecast_p50/MinTrace_method-mint_shrink"
        assert rec_col in out.columns

    def test_output_non_negative(self, catalog, hm):
        bottom = _make_bottom_forecasts(catalog, n_days=3)
        out = reconcile_mint_sub_hierarchy(
            bottom, catalog, Y_df=None, model_col="forecast_p50",
            group_cols=["store_id"],
        )
        if "forecast_p50/MinTrace_method-mint_shrink" in out.columns:
            assert (out["forecast_p50/MinTrace_method-mint_shrink"] >= 0).all()


# ---------------------------------------------------------------------------
# End-to-end coherence after reconciliation
# ---------------------------------------------------------------------------


class TestEndToEndCoherence:
    def test_bottom_up_passes_all_coherence_checks(self, catalog, hm):
        bottom = _make_bottom_forecasts(catalog, n_days=3)
        fc_all = aggregate_base_forecasts(
            bottom, catalog, hm.tags
        ).rename(columns={"forecast_p50": "Forecast"})

        reconciled = reconcile_forecasts(
            fc_all, hm.S_df, hm.tags, method="bu", model_col="Forecast"
        )

        # Rename reconciled column back to forecast_p50 for coherence test
        reconciled_for_test = reconciled[["unique_id", "ds", "Forecast/BottomUp"]].rename(
            columns={"Forecast/BottomUp": "forecast_p50"}
        )

        results = run_coherence_tests(
            reconciled_for_test, hm.S_df, hm.tags, model_col="forecast_p50"
        )
        assert results["non_negativity"]["passed"] is True
        assert results["daily_total_coherence"]["passed"] is True
        # BottomUp always satisfies bottom-up coherence by definition
        assert results["bottom_up_coherence"]["passed"] is True
