"""Unit tests for all 6 leakage-guard rules (section 8.2 of spec).

Each test exercises both the compliant and violating cases so we can be
confident the guards catch real problems.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.features.leakage_guard import (
    LeakageViolation,
    check_all_rules,
    check_calendar_features_safe,
    check_macro_publication_lag,
    check_no_future_prices,
    check_no_future_target_in_rolling,
    check_no_future_weather,
    check_no_target_encoding_on_test,
)

CUTOFF = date(2016, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_with_price(dates: list[date], prices: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": dates, "sell_price": prices}
    ).with_columns(pl.col("date").cast(pl.Date))


def _df_with_rolling(dates: list[date], rolling: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": dates, "rolling_mean_7": rolling}
    ).with_columns(pl.col("date").cast(pl.Date))


def _df_with_weather(dates: list[date], temps: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {"date": dates, "temp_mean": temps}
    ).with_columns(pl.col("date").cast(pl.Date))


# ---------------------------------------------------------------------------
# Rule 1 — No future prices
# ---------------------------------------------------------------------------


class TestNoFuturePrices:
    def test_passes_when_no_future_rows(self):
        df = _df_with_price(
            [CUTOFF - timedelta(1), CUTOFF],
            [1.99, 2.00],
        )
        assert check_no_future_prices(df, CUTOFF) is True

    def test_passes_when_future_rows_have_null_price(self):
        df = _df_with_price(
            [CUTOFF, CUTOFF + timedelta(1)],
            [1.99, None],
        )
        assert check_no_future_prices(df, CUTOFF) is True

    def test_fails_when_future_row_has_price(self):
        df = _df_with_price(
            [CUTOFF, CUTOFF + timedelta(1)],
            [1.99, 2.50],   # future price present → violation
        )
        assert check_no_future_prices(df, CUTOFF) is False

    def test_passes_no_price_column(self):
        df = pl.DataFrame({"date": [CUTOFF], "sales": [1]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        assert check_no_future_prices(df, CUTOFF) is True


# ---------------------------------------------------------------------------
# Rule 2 — No future calendar binary look-aheads
# ---------------------------------------------------------------------------


class TestCalendarFeaturesSafe:
    def test_passes_with_safe_columns(self):
        df = pl.DataFrame({
            "day_of_week": [1],
            "is_weekend": [True],
            "days_to_next_event": [3],
        })
        assert check_calendar_features_safe(df) is True

    def test_fails_with_is_event_tomorrow(self):
        df = pl.DataFrame({
            "day_of_week": [1],
            "is_event_tomorrow": [True],  # forbidden look-ahead
        })
        assert check_calendar_features_safe(df) is False

    def test_fails_with_future_event_flag(self):
        df = pl.DataFrame({"future_event_flag": [1]})
        assert check_calendar_features_safe(df) is False

    def test_custom_forbidden_cols(self):
        df = pl.DataFrame({"my_lookahead_col": [0]})
        assert check_calendar_features_safe(
            df, forbidden_cols=["my_lookahead_col"]
        ) is False


# ---------------------------------------------------------------------------
# Rule 3 — No future weather
# ---------------------------------------------------------------------------


class TestNoFutureWeather:
    def test_passes_no_future_rows(self):
        df = _df_with_weather(
            [CUTOFF - timedelta(2), CUTOFF - timedelta(1)],
            [20.0, 21.0],
        )
        assert check_no_future_weather(df, CUTOFF) is True

    def test_passes_future_rows_null_weather(self):
        df = _df_with_weather(
            [CUTOFF, CUTOFF + timedelta(1)],
            [20.0, None],
        )
        assert check_no_future_weather(df, CUTOFF) is True

    def test_fails_future_row_has_weather(self):
        df = _df_with_weather(
            [CUTOFF, CUTOFF + timedelta(1)],
            [20.0, 22.0],   # future weather → violation
        )
        assert check_no_future_weather(df, CUTOFF) is False

    def test_passes_no_weather_columns(self):
        df = pl.DataFrame({"date": [CUTOFF]}).with_columns(pl.col("date").cast(pl.Date))
        assert check_no_future_weather(df, CUTOFF) is True


# ---------------------------------------------------------------------------
# Rule 4 — No future target in rolling stats
# ---------------------------------------------------------------------------


class TestNoFutureTargetInRolling:
    def test_passes_no_future_rows(self):
        df = _df_with_rolling(
            [CUTOFF - timedelta(1), CUTOFF],
            [5.0, 6.0],
        )
        assert check_no_future_target_in_rolling(df, CUTOFF) is True

    def test_passes_future_rows_null_rolling(self):
        df = _df_with_rolling(
            [CUTOFF, CUTOFF + timedelta(1)],
            [5.0, None],
        )
        assert check_no_future_target_in_rolling(df, CUTOFF) is True

    def test_fails_future_row_has_rolling(self):
        df = _df_with_rolling(
            [CUTOFF, CUTOFF + timedelta(1)],
            [5.0, 7.0],    # rolling stat in future → violation
        )
        assert check_no_future_target_in_rolling(df, CUTOFF) is False

    def test_passes_no_rolling_columns(self):
        df = pl.DataFrame({"date": [CUTOFF], "sales": [1]}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        assert check_no_future_target_in_rolling(df, CUTOFF) is True

    def test_custom_rolling_cols(self):
        df = pl.DataFrame({
            "date": [CUTOFF, CUTOFF + timedelta(1)],
            "my_rolling_feat": [1.0, 2.0],   # future value
        }).with_columns(pl.col("date").cast(pl.Date))
        assert check_no_future_target_in_rolling(
            df, CUTOFF, rolling_cols=["my_rolling_feat"]
        ) is False


# ---------------------------------------------------------------------------
# Rule 5 — No target encoding on test fold
# ---------------------------------------------------------------------------


class TestNoTargetEncodingOnTest:
    def test_passes_encoder_fitted_on_cutoff(self):
        assert check_no_target_encoding_on_test(CUTOFF, CUTOFF) is True

    def test_passes_encoder_fitted_before_cutoff(self):
        assert check_no_target_encoding_on_test(
            CUTOFF - timedelta(days=30), CUTOFF
        ) is True

    def test_fails_encoder_fitted_after_cutoff(self):
        assert check_no_target_encoding_on_test(
            CUTOFF + timedelta(days=1), CUTOFF
        ) is False

    def test_fails_encoder_far_future(self):
        assert check_no_target_encoding_on_test(
            CUTOFF + timedelta(days=365), CUTOFF
        ) is False


# ---------------------------------------------------------------------------
# Rule 6 — Macro publication lag
# ---------------------------------------------------------------------------


class TestMacroPublicationLag:
    def _make_macro(self, dates: list[date], values: list[float | None]) -> pl.DataFrame:
        return pl.DataFrame(
            {"date": dates, "cpi": values}
        ).with_columns(pl.col("date").cast(pl.Date))

    def test_passes_empty_macro(self):
        df = pl.DataFrame({"date": [], "cpi": []}).with_columns(
            pl.col("date").cast(pl.Date)
        )
        assert check_macro_publication_lag(df, CUTOFF) is True

    def test_passes_macro_outside_lag_window(self):
        """Macro data from 60+ days before cutoff is safe."""
        safe_date = CUTOFF - timedelta(days=60)
        df = self._make_macro([safe_date], [300.0])
        assert check_macro_publication_lag(df, CUTOFF, publication_lag_days=45) is True

    def test_fails_macro_within_lag_window(self):
        """Macro data from 30 days before cutoff violates 45-day lag."""
        recent_date = CUTOFF - timedelta(days=30)
        df = self._make_macro([recent_date], [300.0])
        assert check_macro_publication_lag(df, CUTOFF, publication_lag_days=45) is False

    def test_passes_macro_in_window_but_null(self):
        """Null macro values in lag window → compliant."""
        recent_date = CUTOFF - timedelta(days=30)
        df = self._make_macro([recent_date], [None])
        assert check_macro_publication_lag(df, CUTOFF, publication_lag_days=45) is True

    def test_passes_zero_lag_days(self):
        """With lag=0, no historical data is in the window."""
        today = CUTOFF
        df = self._make_macro([today], [300.0])
        # lag window = [cutoff, cutoff] → today itself is in window
        # (lag_start = cutoff - 0 = cutoff)
        assert check_macro_publication_lag(df, CUTOFF, publication_lag_days=0) is False


# ---------------------------------------------------------------------------
# check_all_rules (aggregate)
# ---------------------------------------------------------------------------


class TestCheckAllRules:
    def _clean_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "date": [CUTOFF - timedelta(1), CUTOFF],
            "sell_price": [1.99, 2.00],
            "rolling_mean_7": [5.0, 6.0],
            "temp_mean": [20.0, 21.0],
            "day_of_week": [1, 2],
        }).with_columns(pl.col("date").cast(pl.Date))

    def test_all_rules_pass_clean_df(self):
        df = self._clean_df()
        all_pass, violations = check_all_rules(df, CUTOFF)
        assert all_pass is True
        assert violations == []

    def test_returns_violations_list(self):
        df = self._clean_df()
        # Inject a future price violation
        df = pl.concat([
            df,
            pl.DataFrame({
                "date": [CUTOFF + timedelta(1)],
                "sell_price": [3.50],    # future price
                "rolling_mean_7": [None],
                "temp_mean": [None],
                "day_of_week": [3],
            }).with_columns(pl.col("date").cast(pl.Date)),
        ])
        all_pass, violations = check_all_rules(df, CUTOFF)
        assert all_pass is False
        assert any(v.rule == 1 for v in violations)

    def test_violation_is_named_tuple(self):
        df = pl.DataFrame({
            "date": [CUTOFF + timedelta(1)],
            "sell_price": [5.0],    # future price
        }).with_columns(pl.col("date").cast(pl.Date))
        _, violations = check_all_rules(df, CUTOFF)
        v = violations[0]
        assert isinstance(v, LeakageViolation)
        assert v.rule == 1
        assert "future" in v.name.lower() or "price" in v.name.lower()
