"""Unit tests for all feature engineering modules.

Uses small synthetic DataFrames so tests run in milliseconds without any
real data.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl
import pytest

from src.features.calendar_features import add_calendar_features, _add_event_proximity
from src.features.interaction_features import LeaveOneOutEncoder, add_interaction_features
from src.features.intermittency_features import add_intermittency_features
from src.features.lag_features import add_lag_features
from src.features.price_features import add_price_features
from src.features.rolling_features import add_rolling_features
from src.features.weather_features import add_weather_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sales(
    n: int = 14,
    values: list[int] | None = None,
    item: str = "FOODS_1_001",
    store: str = "CA_1",
    state: str = "CA",
    dept: str = "FOODS_1",
    cat: str = "FOODS",
) -> pl.DataFrame:
    start = date(2011, 1, 29)
    vals = values if values is not None else list(range(1, n + 1))
    n = len(vals)
    return pl.DataFrame(
        {
            "id": [f"{item}_{store}"] * n,
            "item_id": [item] * n,
            "dept_id": [dept] * n,
            "cat_id": [cat] * n,
            "store_id": [store] * n,
            "state_id": [state] * n,
            "date": [start + timedelta(days=i) for i in range(n)],
            "sales": vals,
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def _make_multi_sales(n: int = 14) -> pl.DataFrame:
    """Two series side by side."""
    a = _make_sales(n=n, item="A", store="CA_1", values=list(range(n)))
    b = _make_sales(n=n, item="B", store="CA_1", values=[2] * n)
    return pl.concat([a, b]).sort(["id", "date"])


# ---------------------------------------------------------------------------
# lag_features
# ---------------------------------------------------------------------------


class TestLagFeatures:
    def test_lag_1_values(self):
        df = _make_sales(values=[10, 20, 30, 40])
        result = add_lag_features(df, lags=[1])
        lags = result.sort("date")["lag_1"].to_list()
        assert lags[0] is None  # first row has no lag
        assert lags[1] == pytest.approx(10.0)
        assert lags[2] == pytest.approx(20.0)
        assert lags[3] == pytest.approx(30.0)

    def test_lag_7_first_7_are_null(self):
        df = _make_sales(n=14, values=list(range(14)))
        result = add_lag_features(df, lags=[7])
        first_7 = result.sort("date").head(7)["lag_7"].to_list()
        assert all(v is None for v in first_7)

    def test_lag_7_value_after_7_days(self):
        vals = list(range(14))
        df = _make_sales(n=14, values=vals)
        result = add_lag_features(df, lags=[7]).sort("date")
        assert result["lag_7"][7] == pytest.approx(0.0)  # row 7 → lag7 = sales[0] = 0

    def test_all_lag_columns_created(self):
        df = _make_sales(n=100)
        result = add_lag_features(df)
        from src.features.lag_features import LAG_PERIODS
        for lag in LAG_PERIODS:
            assert f"lag_{lag}" in result.columns

    def test_lag_preserves_row_count(self):
        df = _make_sales(n=30)
        result = add_lag_features(df)
        assert len(result) == len(df)

    def test_lag_within_each_id(self):
        """lag_1 should NOT bleed across different series."""
        df = _make_multi_sales(n=7)
        result = add_lag_features(df, lags=[1]).sort(["id", "date"])
        # First row of each series must be null
        first_rows = result.group_by("id").agg(
            pl.col("lag_1").first().alias("lag_1_first")
        )
        assert first_rows["lag_1_first"].null_count() == 2  # both series

    def test_lag_cutoff_date_makes_future_null(self):
        vals = list(range(10))
        df = _make_sales(n=10, values=vals)
        cutoff = date(2011, 1, 29) + timedelta(days=4)
        result = add_lag_features(df, lags=[1], cutoff_date=cutoff)
        future = result.filter(pl.col("date") > cutoff)
        assert future["lag_1"].null_count() == len(future)


# ---------------------------------------------------------------------------
# rolling_features
# ---------------------------------------------------------------------------


class TestRollingFeatures:
    def test_rolling_mean_7_first_row(self):
        """First row's rolling_mean_7 = first value (min_samples=1)."""
        df = _make_sales(values=[10, 20, 30, 40, 50, 60, 70])
        result = add_rolling_features(df, windows=[7]).sort("date")
        assert result["rolling_mean_7"][0] == pytest.approx(10.0, abs=1e-3)

    def test_rolling_mean_7_after_window(self):
        """After 7 rows: rolling_mean_7 should equal mean of first 7 values."""
        vals = [2] * 7 + [10]
        df = _make_sales(values=vals)
        result = add_rolling_features(df, windows=[7]).sort("date")
        # Row index 6 (7th row, 0-indexed): mean of rows 0-6 = 2.0
        assert result["rolling_mean_7"][6] == pytest.approx(2.0, abs=1e-3)

    def test_ratio_mean_7_28_constant_series(self):
        """For a constant series, ratio_mean_7_28 = 1.0."""
        df = _make_sales(n=60, values=[5] * 60)
        result = add_rolling_features(df, windows=[7, 28])
        ratios = result["ratio_mean_7_28"].drop_nulls().to_list()
        assert all(abs(r - 1.0) < 1e-6 for r in ratios)

    def test_rolling_zero_pct_28_all_zeros(self):
        """All-zero series → rolling_zero_pct_28 = 1.0."""
        df = _make_sales(n=50, values=[0] * 50)
        result = add_rolling_features(df, windows=[7, 28])
        # After 28 rows, zero_pct should be 1.0
        late = result.sort("date").tail(10)["rolling_zero_pct_28"].to_list()
        assert all(abs(v - 1.0) < 1e-6 for v in late)

    def test_rolling_features_preserve_row_count(self):
        df = _make_sales(n=30)
        result = add_rolling_features(df, windows=[7, 14])
        assert len(result) == len(df)

    def test_expected_rolling_columns(self):
        df = _make_sales(n=30)
        result = add_rolling_features(df, windows=[7, 14])
        for w in [7, 14]:
            for stat in ["mean", "std", "median", "min", "max"]:
                assert f"rolling_{stat}_{w}" in result.columns
        assert "rolling_zero_pct_28" in result.columns
        assert "ratio_mean_7_28" in result.columns


# ---------------------------------------------------------------------------
# calendar_features
# ---------------------------------------------------------------------------


class TestCalendarFeatures:
    def _make_calendar(self, n: int = 10) -> pl.DataFrame:
        start = date(2011, 1, 29)
        rows = {
            "date": [start + timedelta(days=i) for i in range(n)],
            "wm_yr_wk": [11101] * n,
            "weekday": ["Saturday"] * n,
            "wday": [1] * n,
            "month": [1] * n,
            "year": [2011] * n,
            "d": [f"d_{i+1}" for i in range(n)],
            "is_weekend": [True] * n,
            "quarter": [1] * n,
            "snap_CA": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0][:n],
            "snap_TX": [0] * n,
            "snap_WI": [0] * n,
            "event_name_1": [None, None, "Christmas", None, None, None, None, None, None, None][:n],
        }
        return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))

    def test_day_of_week_created(self):
        df = _make_sales(n=10)
        cal = self._make_calendar(10)
        result = add_calendar_features(df, cal)
        assert "day_of_week" in result.columns

    def test_snap_active_for_ca(self):
        df = _make_sales(n=10, state="CA")
        cal = self._make_calendar(10)
        result = add_calendar_features(df, cal)
        assert "snap_active" in result.columns
        # First row: snap_CA=1 → snap_active=True
        assert result.sort("date")["snap_active"][0] is True

    def test_is_month_start_january(self):
        df = _make_sales(n=5)
        cal = self._make_calendar(5)
        result = add_calendar_features(df, cal)
        assert "is_month_start" in result.columns

    def test_days_to_next_event_created(self):
        df = _make_sales(n=10)
        cal = self._make_calendar(10)
        result = add_calendar_features(df, cal)
        assert "days_to_next_event" in result.columns
        assert "days_since_last_event" in result.columns

    def test_days_since_event_zero_on_event_day(self):
        df = _make_sales(n=10)
        cal = self._make_calendar(10)
        result = add_calendar_features(df, cal).sort("date")
        # Index 2 has event (Christmas in fixture)
        assert result["days_since_last_event"][2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# price_features
# ---------------------------------------------------------------------------


class TestPriceFeatures:
    def _make_prices(self, n: int = 14, price: float = 2.0) -> pl.DataFrame:
        start = date(2011, 1, 29)
        return pl.DataFrame(
            {
                "store_id": ["CA_1"] * n,
                "item_id": ["FOODS_1_001"] * n,
                "date": [start + timedelta(days=i) for i in range(n)],
                "sell_price": [price] * n,
            }
        ).with_columns(pl.col("date").cast(pl.Date))

    def test_sell_price_joined(self):
        df = _make_sales(n=14)
        prices = self._make_prices(n=14)
        result = add_price_features(df, prices)
        assert "sell_price" in result.columns
        non_null = result["sell_price"].drop_nulls()
        assert len(non_null) > 0

    def test_price_ratio_vs_rolling_28_constant(self):
        """Constant price → ratio = 1.0."""
        df = _make_sales(n=50)
        prices = self._make_prices(n=50, price=3.0)
        result = add_price_features(df, prices)
        ratios = result["price_ratio_vs_rolling_28"].drop_nulls().to_list()
        assert all(abs(r - 1.0) < 1e-6 for r in ratios)

    def test_relative_price_in_dept_same_dept(self):
        """All items in same dept, same price → relative = 1.0."""
        df = _make_sales(n=10)
        prices = self._make_prices(n=10, price=2.0)
        result = add_price_features(df, prices)
        rels = result["relative_price_in_dept"].drop_nulls().to_list()
        assert all(abs(r - 1.0) < 1e-6 for r in rels)

    def test_is_price_drop_gt_10pct(self):
        """No price change → is_price_drop_gt_10pct = False."""
        df = _make_sales(n=20)
        prices = self._make_prices(n=20, price=1.0)
        result = add_price_features(df, prices)
        assert result["is_price_drop_gt_10pct"].sum() == 0


# ---------------------------------------------------------------------------
# intermittency_features
# ---------------------------------------------------------------------------


class TestIntermittencyFeatures:
    def test_pct_zero_last_28d_all_zeros(self):
        df = _make_sales(n=50, values=[0] * 50)
        result = add_intermittency_features(df)
        late = result.sort("date").tail(10)["pct_zero_last_28d"].to_list()
        assert all(abs(v - 1.0) < 1e-6 for v in late)

    def test_pct_zero_last_28d_all_nonzero(self):
        df = _make_sales(n=50, values=[5] * 50)
        result = add_intermittency_features(df)
        late = result.sort("date").tail(10)["pct_zero_last_28d"].to_list()
        assert all(abs(v - 0.0) < 1e-6 for v in late)

    def test_days_since_last_sale_after_nonzero(self):
        """Day immediately after a sale → days_since_last_sale = 1."""
        df = _make_sales(n=3, values=[5, 0, 0])
        result = add_intermittency_features(df).sort("date")
        assert result["days_since_last_sale"][1] == pytest.approx(1.0)
        assert result["days_since_last_sale"][2] == pytest.approx(2.0)

    def test_days_since_last_sale_on_sale_day(self):
        """On a sale day → days_since_last_sale = 0."""
        df = _make_sales(n=3, values=[5, 0, 3])
        result = add_intermittency_features(df).sort("date")
        assert result["days_since_last_sale"][0] == pytest.approx(0.0)
        assert result["days_since_last_sale"][2] == pytest.approx(0.0)

    def test_streak_zeros_resets_on_sale(self):
        df = _make_sales(n=5, values=[0, 0, 5, 0, 0])
        result = add_intermittency_features(df).sort("date")
        streaks = result["streak_zeros"].to_list()
        assert streaks[2] == 0  # sale day
        assert streaks[3] == 1  # first zero after sale
        assert streaks[4] == 2  # second zero after sale

    def test_non_zero_demand_mean_correct(self):
        df = _make_sales(n=5, values=[0, 2, 0, 4, 0])
        result = add_intermittency_features(df)
        # non_zero_demand_mean should be (2+4)/2 = 3.0
        assert result["non_zero_demand_mean"][0] == pytest.approx(3.0, abs=1e-6)

    def test_intermittency_features_preserve_row_count(self):
        df = _make_sales(n=30)
        result = add_intermittency_features(df)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# weather_features
# ---------------------------------------------------------------------------


class TestWeatherFeatures:
    def _make_weather(self, n: int = 10) -> pl.DataFrame:
        start = date(2011, 1, 29)
        return pl.DataFrame(
            {
                "state": ["CA"] * n,
                "date": [start + timedelta(days=i) for i in range(n)],
                "temperature_2m_max": [25.0 + i * 0.1 for i in range(n)],
                "temperature_2m_min": [15.0] * n,
                "temperature_2m_mean": [20.0] * n,
                "precipitation_sum": [0.0] * n,
                "weathercode": [1.0] * n,
            }
        ).with_columns(pl.col("date").cast(pl.Date))

    def test_weather_features_joined(self):
        df = _make_sales(n=10)
        weather = self._make_weather(10)
        result = add_weather_features(df, weather)
        assert "temp_max" in result.columns
        assert "temp_min" in result.columns
        assert "precipitation_sum" in result.columns

    def test_temp_anomaly_vs_30d_avg_created(self):
        df = _make_sales(n=10)
        weather = self._make_weather(10)
        result = add_weather_features(df, weather)
        assert "temp_anomaly_vs_30d_avg" in result.columns

    def test_temp_anomaly_constant_temp_is_zero(self):
        """Constant temp_mean → anomaly = 0."""
        df = _make_sales(n=30)
        weather = self._make_weather(30)
        # All temp_mean = 20.0 → anomaly vs rolling 30d avg = 0
        result = add_weather_features(df, weather)
        anomalies = result["temp_anomaly_vs_30d_avg"].drop_nulls().to_list()
        assert all(abs(a - 0.0) < 1e-6 for a in anomalies)

    def test_weather_row_count_preserved(self):
        df = _make_sales(n=10)
        weather = self._make_weather(10)
        result = add_weather_features(df, weather)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# interaction_features
# ---------------------------------------------------------------------------


class TestInteractionFeatures:
    def _make_df_with_price_and_snap(self, n: int = 5) -> pl.DataFrame:
        start = date(2011, 1, 29)
        return pl.DataFrame(
            {
                "id": ["A_CA_1"] * n,
                "item_id": ["A"] * n,
                "store_id": ["CA_1"] * n,
                "state_id": ["CA"] * n,
                "dept_id": ["FOODS_1"] * n,
                "cat_id": ["FOODS"] * n,
                "date": [start + timedelta(days=i) for i in range(n)],
                "sales": [1] * n,
                "sell_price": [2.0] * n,
                "is_weekend": [True, False, True, False, True],
                "snap_active": [True, True, False, False, True],
            }
        ).with_columns(pl.col("date").cast(pl.Date))

    def test_price_x_is_weekend_created(self):
        df = self._make_df_with_price_and_snap()
        result = add_interaction_features(df)
        assert "price_x_is_weekend" in result.columns

    def test_price_x_is_weekend_values(self):
        """price_x_is_weekend = sell_price × is_weekend (1/0)."""
        df = self._make_df_with_price_and_snap()
        result = add_interaction_features(df).sort("date")
        row0 = result[0]  # is_weekend=True, price=2.0 → 2.0
        row1 = result[1]  # is_weekend=False → 0.0
        assert row0["price_x_is_weekend"][0] == pytest.approx(2.0)
        assert row1["price_x_is_weekend"][0] == pytest.approx(0.0)

    def test_snap_x_dept_created(self):
        df = self._make_df_with_price_and_snap()
        result = add_interaction_features(df)
        assert "snap_x_dept" in result.columns


# ---------------------------------------------------------------------------
# LeaveOneOutEncoder
# ---------------------------------------------------------------------------


class TestLeaveOneOutEncoder:
    def _make_enc_df(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "item_id": ["A", "A", "B", "B", "C"],
                "sales": [10.0, 20.0, 5.0, 5.0, 1.0],
            }
        )

    def test_fit_transform_adds_column(self):
        enc = LeaveOneOutEncoder(noise_std=0.0)
        df = self._make_enc_df()
        result = enc.fit_transform(df, group_cols=["item_id"], target_col="sales")
        assert "target_enc_item_id" in result.columns

    def test_fit_stores_global_mean(self):
        enc = LeaveOneOutEncoder()
        df = self._make_enc_df()
        enc.fit(df, group_cols=["item_id"], target_col="sales")
        expected_global = (10 + 20 + 5 + 5 + 1) / 5
        assert enc._global_mean == pytest.approx(expected_global)

    def test_transform_produces_non_null(self):
        enc = LeaveOneOutEncoder(noise_std=0.0)
        train = self._make_enc_df()
        enc.fit(train, group_cols=["item_id"], target_col="sales")
        test = pl.DataFrame({"item_id": ["A", "B", "D"]})  # D is unseen
        result = enc.transform(test, group_cols=["item_id"])
        assert "target_enc_item_id" in result.columns
        # Unseen category → global mean (no null)
        assert result["target_enc_item_id"].null_count() == 0

    def test_encoder_fit_cutoff_semantics(self):
        """Encoder fitted on earlier data should not see test rows."""
        from src.features.leakage_guard import check_no_target_encoding_on_test
        cutoff = date(2016, 1, 1)
        # Fitted on data up to cutoff → compliant
        assert check_no_target_encoding_on_test(cutoff, cutoff) is True
        # Fitted on data AFTER cutoff → violation
        assert check_no_target_encoding_on_test(
            cutoff + timedelta(days=1), cutoff
        ) is False
