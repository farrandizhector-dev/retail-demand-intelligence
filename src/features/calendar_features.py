"""Calendar and event features for demand forecasting.

Derives temporal and event-proximity features by joining the enriched
silver calendar onto the sales DataFrame.

LEAKAGE PROTOCOL (rule 2 — section 8.2):
  - Static calendar fields (day_of_week, month, etc.) are safe — they carry
    no information about future demand.
  - ``days_to_next_event`` is computed deterministically from known event
    dates and is safe as a distance-based feature (not a binary look-ahead).
  - SNAP flags are state-specific and known in advance (government schedule);
    they are safe to use as-is.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def add_calendar_features(
    df: pl.DataFrame,
    calendar_df: pl.DataFrame,
    *,
    date_col: str = "date",
    state_col: str = "state_id",
    id_col: str = "id",
) -> pl.DataFrame:
    """Join enriched calendar features onto the sales DataFrame.

    Parameters
    ----------
    df:
        Long-format sales DataFrame with columns
        ``(id, date, store_id, item_id, state_id, …)``.
    calendar_df:
        Silver calendar enriched DataFrame (output of
        :func:`src.transform.pipeline.enrich_calendar`).
        Expected columns: ``date, wm_yr_wk, weekday, wday, month, year,
        is_weekend, quarter``, optionally ``snap_CA, snap_TX, snap_WI``,
        ``event_name_1, event_type_1``.
    date_col:
        Name of the date column in both DataFrames.
    state_col:
        Column identifying which state the row belongs to (used for SNAP).
    id_col:
        Series identifier (needed for days_to_next/since_event computation).

    Returns
    -------
    pl.DataFrame
        Input DataFrame with calendar feature columns appended:
        ``day_of_week, day_of_month, week_of_year, month, quarter,
          is_weekend, is_month_start, is_month_end,
          snap_active, days_to_next_event, days_since_last_event``
    """
    df = df.sort([id_col, date_col])

    # --- Basic date-derived features ---
    df = df.with_columns(
        pl.col(date_col).dt.weekday().alias("day_of_week"),       # 1=Mon … 7=Sun (ISO)
        pl.col(date_col).dt.day().alias("day_of_month"),
        pl.col(date_col).dt.week().alias("week_of_year"),
        pl.col(date_col).dt.month().cast(pl.Int32).alias("month"),
        pl.col(date_col).dt.quarter().alias("quarter"),
        (pl.col(date_col).dt.weekday() >= 6).alias("is_weekend"),
        (pl.col(date_col).dt.day() == 1).alias("is_month_start"),
        (
            pl.col(date_col).dt.day()
            == pl.col(date_col).dt.month_end().dt.day()
        ).alias("is_month_end"),
    )

    # --- SNAP active flag (state-specific) ---
    # calendar_df may have snap_CA, snap_TX, snap_WI columns
    snap_cols = {
        "CA": "snap_CA",
        "TX": "snap_TX",
        "WI": "snap_WI",
    }
    cal_snap_cols = [c for c in snap_cols.values() if c in calendar_df.columns]

    cal_join = calendar_df.select(
        [date_col] + cal_snap_cols + (
            ["event_name_1"] if "event_name_1" in calendar_df.columns else []
        )
    )
    df = df.join(cal_join, on=date_col, how="left")

    # Derive snap_active from state-specific column
    if cal_snap_cols:
        df = df.with_columns(
            pl.when(pl.col(state_col) == "CA")
            .then(pl.col("snap_CA").cast(pl.Boolean) if "snap_CA" in df.columns else pl.lit(False))
            .when(pl.col(state_col) == "TX")
            .then(pl.col("snap_TX").cast(pl.Boolean) if "snap_TX" in df.columns else pl.lit(False))
            .when(pl.col(state_col) == "WI")
            .then(pl.col("snap_WI").cast(pl.Boolean) if "snap_WI" in df.columns else pl.lit(False))
            .otherwise(pl.lit(False))
            .alias("snap_active")
        )
        # Drop raw snap columns (they are state-specific; snap_active is the clean version)
        df = df.drop([c for c in cal_snap_cols if c in df.columns])
    else:
        df = df.with_columns(pl.lit(False).alias("snap_active"))

    # --- Event proximity features ---
    df = _add_event_proximity(df, date_col=date_col, id_col=id_col)

    return df


def _add_event_proximity(
    df: pl.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "id",
    event_col: str = "event_name_1",
    max_days: int = 30,
) -> pl.DataFrame:
    """Add ``days_to_next_event`` and ``days_since_last_event`` columns.

    Uses ``event_name_1`` from the joined calendar to identify event days.
    Distance is capped at ``max_days`` to limit feature magnitude.

    These are DISTANCE features (not binary look-aheads), so they are
    leakage-safe even though they reference future dates.

    Parameters
    ----------
    df:
        DataFrame with ``date`` and (optionally) ``event_name_1``.
    date_col:
        Name of the date column.
    id_col:
        Not used in computation; kept for API consistency.
    event_col:
        Column flagging whether the day has a named event (non-null/non-empty).
    max_days:
        Maximum distance (days) to cap the feature at.
    """
    if event_col not in df.columns:
        # No event info — fill with max_days
        return df.with_columns(
            pl.lit(max_days).alias("days_to_next_event"),
            pl.lit(max_days).alias("days_since_last_event"),
        )

    # Build the sorted unique set of event dates from the joined calendar
    is_event_expr = pl.col(event_col).is_not_null() & (pl.col(event_col) != "")

    df = df.with_columns(is_event_expr.alias("_is_event"))

    # days_since_last_event: for each row, how many days since the last event
    # We use a forward-fill trick: record the date on event days, forward-fill, diff
    df = df.with_columns(
        pl.when(pl.col("_is_event"))
        .then(pl.col(date_col))
        .otherwise(None)
        .forward_fill()
        .alias("_last_event_date")
    )
    df = df.with_columns(
        (
            (pl.col(date_col) - pl.col("_last_event_date")).dt.total_days()
            .clip(0, max_days)
        )
        .fill_null(max_days)
        .alias("days_since_last_event")
    )

    # days_to_next_event: backward fill of the next event date, then diff
    df = df.with_columns(
        pl.when(pl.col("_is_event"))
        .then(pl.col(date_col))
        .otherwise(None)
        .backward_fill()
        .alias("_next_event_date")
    )
    df = df.with_columns(
        (
            (pl.col("_next_event_date") - pl.col(date_col)).dt.total_days()
            .clip(0, max_days)
        )
        .fill_null(max_days)
        .alias("days_to_next_event")
    )

    return df.drop(["_is_event", "_last_event_date", "_next_event_date"])
