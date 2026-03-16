"""Leakage control protocol — 6 rules (section 8.2 of spec).

Each rule is implemented as a standalone function that receives a DataFrame
and a ``cutoff_date`` and returns ``True`` if the rule passes (no leakage
detected) or ``False`` if a violation is found.

The rules enforce the contract that features computed for a prediction at
date T must use ONLY information available before T.

Usage
-----
>>> from src.features.leakage_guard import check_all_rules
>>> all_pass, violations = check_all_rules(feature_df, cutoff_date=date(2016,1,1))
>>> assert all_pass, f"Leakage detected: {violations}"
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import NamedTuple

import polars as pl


# ---------------------------------------------------------------------------
# Rule 1 — No future prices
# ---------------------------------------------------------------------------


def check_no_future_prices(
    df: pl.DataFrame,
    cutoff_date: date,
    *,
    price_col: str = "sell_price",
    date_col: str = "date",
) -> bool:
    """Rule 1: sell_price features must use only price data up to t-1.

    Checks that no row where ``date > cutoff_date`` has a non-null price
    originating from after ``cutoff_date``.  In practice this means the
    prices table used to generate features must have been filtered to
    ``price_date <= cutoff_date`` before joining.

    Parameters
    ----------
    df:
        Feature DataFrame that has already been joined with price data.
    cutoff_date:
        The forecast origin date; features for rows with ``date > cutoff_date``
        must not contain future price information.

    Returns
    -------
    bool
        ``True`` if compliant (no future price leakage detected).
    """
    if price_col not in df.columns:
        return True  # price not present — trivially compliant

    # If any future row has a non-null price, that price may be from the future.
    # The guard checks that future rows (date > cutoff) have NULL prices.
    future_rows = df.filter(pl.col(date_col) > cutoff_date)
    if future_rows.is_empty():
        return True

    n_future_with_price = future_rows.filter(pl.col(price_col).is_not_null()).height
    return n_future_with_price == 0


# ---------------------------------------------------------------------------
# Rule 2 — No future calendar info for features
# ---------------------------------------------------------------------------


def check_calendar_features_safe(
    df: pl.DataFrame,
    *,
    forbidden_cols: list[str] | None = None,
) -> bool:
    """Rule 2: Calendar-derived features must be distance-based for events.

    Static temporal fields (day_of_week, month, etc.) are safe.  Binary
    flags that look directly at future event occurrence (e.g. ``is_event_next_7d``
    with a binary True/False) are NOT safe.

    This check verifies that the DataFrame does NOT contain any column
    from a forbidden list of look-ahead binary features.  The spec mandates
    distance-based features (``days_to_next_event``) instead.

    Parameters
    ----------
    df:
        Feature DataFrame.
    forbidden_cols:
        List of column names considered look-ahead binary event features.
        Defaults to the known dangerous patterns.

    Returns
    -------
    bool
        ``True`` if no forbidden look-ahead binary columns are present.
    """
    if forbidden_cols is None:
        forbidden_cols = [
            "is_event_tomorrow",
            "is_event_next_7d",
            "future_event_flag",
        ]
    return not any(col in df.columns for col in forbidden_cols)


# ---------------------------------------------------------------------------
# Rule 3 — No future weather
# ---------------------------------------------------------------------------


def check_no_future_weather(
    df: pl.DataFrame,
    cutoff_date: date,
    *,
    weather_cols: list[str] | None = None,
    date_col: str = "date",
) -> bool:
    """Rule 3: Weather features must use only data up to cutoff_date − 1.

    Same logic as rule 1: rows with ``date > cutoff_date`` must have null
    weather values (they have not been seen yet).

    Parameters
    ----------
    df:
        Feature DataFrame with joined weather columns.
    cutoff_date:
        Forecast origin date.
    weather_cols:
        Names of weather feature columns to check.

    Returns
    -------
    bool
        ``True`` if compliant.
    """
    if weather_cols is None:
        weather_cols = ["temp_max", "temp_min", "temp_mean", "precipitation_sum", "weathercode"]

    present = [c for c in weather_cols if c in df.columns]
    if not present:
        return True

    future_rows = df.filter(pl.col(date_col) > cutoff_date)
    if future_rows.is_empty():
        return True

    for col in present:
        n_violations = future_rows.filter(pl.col(col).is_not_null()).height
        if n_violations > 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Rule 4 — No future target in rolling stats
# ---------------------------------------------------------------------------


def check_no_future_target_in_rolling(
    df: pl.DataFrame,
    cutoff_date: date,
    *,
    rolling_cols: list[str] | None = None,
    date_col: str = "date",
) -> bool:
    """Rule 4: Rolling window statistics must end at or before cutoff_date.

    Rows with ``date > cutoff_date`` should have null rolling statistics
    (these rows represent the forecast horizon — their target is unknown).

    Parameters
    ----------
    df:
        Feature DataFrame with rolling statistics columns.
    cutoff_date:
        The fold's training cutoff.
    rolling_cols:
        Rolling feature columns to inspect.  Defaults to checking any
        column whose name starts with ``"rolling_"``.

    Returns
    -------
    bool
        ``True`` if compliant.
    """
    if rolling_cols is None:
        rolling_cols = [c for c in df.columns if c.startswith("rolling_")]

    if not rolling_cols:
        return True

    future_rows = df.filter(pl.col(date_col) > cutoff_date)
    if future_rows.is_empty():
        return True

    for col in rolling_cols:
        n_violations = future_rows.filter(pl.col(col).is_not_null()).height
        if n_violations > 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Rule 5 — No target encoding on test fold
# ---------------------------------------------------------------------------


def check_no_target_encoding_on_test(
    encoder_fit_cutoff: date,
    cutoff_date: date,
) -> bool:
    """Rule 5: Target encoder must be fitted ONLY on training data.

    The encoder's fit cutoff must be <= the fold's cutoff date.  If the
    encoder was fitted on data that includes test-fold rows, the encoded
    means will have leaked information from the test set.

    Parameters
    ----------
    encoder_fit_cutoff:
        The latest date included in the encoder's training data.
    cutoff_date:
        The current fold's training cutoff.

    Returns
    -------
    bool
        ``True`` if the encoder was fitted on data up to (or before) the fold cutoff.
    """
    return encoder_fit_cutoff <= cutoff_date


# ---------------------------------------------------------------------------
# Rule 6 — Macro publication lag
# ---------------------------------------------------------------------------


def check_macro_publication_lag(
    macro_df: pl.DataFrame,
    cutoff_date: date,
    *,
    publication_lag_days: int = 45,
    date_col: str = "date",
    value_cols: list[str] | None = None,
) -> bool:
    """Rule 6: Macro features must respect publication lag (45 days).

    CPI, unemployment, etc. are published ~1 month after the reference
    period.  A 45-day safety margin ensures we never use macro data that
    was not yet published at ``cutoff_date``.

    Checks that the macro DataFrame contains no non-null values for dates
    within ``publication_lag_days`` of ``cutoff_date``.

    Parameters
    ----------
    macro_df:
        Macro feature DataFrame with a date column.
    cutoff_date:
        The fold's training cutoff.
    publication_lag_days:
        Safety margin in days (default 45).
    date_col:
        Name of the date column.
    value_cols:
        Macro value columns to check.  If None, all non-date columns.

    Returns
    -------
    bool
        ``True`` if compliant (no macro data inside the lag window).
    """
    if macro_df.is_empty():
        return True

    lag_start = cutoff_date - timedelta(days=publication_lag_days)
    # Rows in the lag window: lag_start <= date <= cutoff_date
    lag_window = macro_df.filter(
        (pl.col(date_col) >= lag_start) & (pl.col(date_col) <= cutoff_date)
    )
    if lag_window.is_empty():
        return True

    if value_cols is None:
        value_cols = [c for c in macro_df.columns if c != date_col]

    for col in value_cols:
        if col not in lag_window.columns:
            continue
        n_violations = lag_window.filter(pl.col(col).is_not_null()).height
        if n_violations > 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Aggregate checker
# ---------------------------------------------------------------------------


class LeakageViolation(NamedTuple):
    rule: int
    name: str
    message: str


def check_all_rules(
    feature_df: pl.DataFrame,
    cutoff_date: date,
    *,
    encoder_fit_cutoff: date | None = None,
    macro_df: pl.DataFrame | None = None,
) -> tuple[bool, list[LeakageViolation]]:
    """Run all 6 leakage rules and return a summary.

    Parameters
    ----------
    feature_df:
        Full feature DataFrame (train + optionally test rows).
    cutoff_date:
        The fold's training cutoff.
    encoder_fit_cutoff:
        When the target encoder was fitted (defaults to ``cutoff_date``).
    macro_df:
        Optional macro features DataFrame for rule 6.

    Returns
    -------
    tuple[bool, list[LeakageViolation]]
        ``(all_pass, violations)`` where ``all_pass`` is ``True`` iff no
        violations were found.
    """
    encoder_fit_cutoff = encoder_fit_cutoff or cutoff_date
    violations: list[LeakageViolation] = []

    checks: list[tuple[int, str, bool]] = [
        (1, "No future prices",
         check_no_future_prices(feature_df, cutoff_date)),
        (2, "No future calendar binary look-aheads",
         check_calendar_features_safe(feature_df)),
        (3, "No future weather",
         check_no_future_weather(feature_df, cutoff_date)),
        (4, "No future target in rolling stats",
         check_no_future_target_in_rolling(feature_df, cutoff_date)),
        (5, "No target encoding on test fold",
         check_no_target_encoding_on_test(encoder_fit_cutoff, cutoff_date)),
    ]

    if macro_df is not None:
        checks.append(
            (6, "Macro publication lag",
             check_macro_publication_lag(macro_df, cutoff_date)),
        )

    for rule_num, rule_name, passed in checks:
        if not passed:
            violations.append(
                LeakageViolation(
                    rule=rule_num,
                    name=rule_name,
                    message=f"Rule {rule_num} violated: {rule_name}",
                )
            )

    return len(violations) == 0, violations
