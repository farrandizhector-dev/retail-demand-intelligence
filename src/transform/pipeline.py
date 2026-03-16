"""Bronze → Silver transformation pipeline.

Three independent transformation steps:

1. :func:`sales_wide_to_long`
   Melts the wide-format M5 sales table (d_1 … d_N columns) into a long
   daily-grain table, joins with the calendar for real dates, and derives
   ``id`` and ``is_zero_sale``.  Output is partitioned by ``state_id`` and
   ``year`` under ``data/silver/silver_sales_long/``.

2. :func:`prices_weekly_to_daily`
   Expands weekly sell-prices to daily grain by joining with the calendar
   date-to-``wm_yr_wk`` mapping, then forward-fills within each
   ``(store_id, item_id)`` series.

3. :func:`enrich_calendar`
   Adds derived columns (``is_weekend``, ``quarter``, typed ``date``) to the
   bronze calendar table and writes ``silver_calendar_enriched.parquet``.

4. :func:`build_silver_weather`
   Casts the bronze weather date column to ``pl.Date`` and writes
   ``silver_weather_daily.parquet``.

5. :func:`run_bronze_to_silver`
   Orchestrates all four steps in the correct dependency order.

Processing engine: Polars (ADR-001).
Leakage control: transformations are purely structural / temporal — no
future information is used to fill or derive values.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _day_columns(df: pl.DataFrame) -> list[str]:
    """Return column names that match the M5 pattern ``d_<integer>``."""
    return [c for c in df.columns if c.startswith("d_") and c[2:].isdigit()]


def _write_partition(df: pl.DataFrame, silver_sales_dir: Path) -> None:
    """Write a long-format sales DataFrame partitioned by ``state_id`` / ``year``.

    Output path pattern:
        ``<silver_sales_dir>/state=<STATE>/year=<YEAR>.parquet``

    Parameters
    ----------
    df:
        Long-format sales DataFrame with ``state_id``, ``date`` (pl.Date),
        and all silver schema columns.
    silver_sales_dir:
        Root directory for the partitioned output.
    """
    states = df["state_id"].unique().sort().to_list()
    for state in states:
        state_df = df.filter(pl.col("state_id") == state)
        years = state_df["date"].dt.year().unique().sort().to_list()
        for year in years:
            year_df = state_df.filter(pl.col("date").dt.year() == year)
            partition_dir = silver_sales_dir / f"state={state}" / f"year={year}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            year_df.write_parquet(partition_dir / "data.parquet", compression="snappy")


# ---------------------------------------------------------------------------
# 1. Sales: wide → long
# ---------------------------------------------------------------------------


def sales_wide_to_long(
    bronze_sales_path: Path,
    bronze_calendar_path: Path,
    silver_sales_dir: Path,
    *,
    force: bool = False,
) -> None:
    """Melt M5 wide sales table to long format and write partitioned Parquet.

    The output is partitioned by ``state_id`` and ``year`` so downstream
    readers can push-down filters without scanning the full dataset.

    Schema of each partition file (silver grain: ``(date, store_id, item_id)``):

    .. code-block:: text

        id            str   — "{item_id}_{store_id}"
        date          Date
        store_id      str
        item_id       str
        state_id      str   — one of CA, TX, WI
        dept_id       str
        cat_id        str
        sales         i64   — ≥ 0
        is_zero_sale  bool

    Parameters
    ----------
    bronze_sales_path:
        Path to ``bronze_sales.parquet`` (wide format).
    bronze_calendar_path:
        Path to ``bronze_calendar.parquet``.
    silver_sales_dir:
        Root of the partitioned silver output (e.g. ``data/silver/silver_sales_long``).
    force:
        Re-process even when output partitions already exist.
    """
    # Idempotency: check if at least one partition exists
    if silver_sales_dir.exists() and any(silver_sales_dir.rglob("*.parquet")) and not force:
        return

    # --- Load bronze ---
    sales = pl.read_parquet(bronze_sales_path)
    calendar = pl.read_parquet(bronze_calendar_path)

    # --- Identify day columns ---
    day_cols = _day_columns(sales)
    id_vars = [c for c in sales.columns if c not in day_cols]

    # --- Unpivot wide → long (melt is deprecated in Polars ≥ 1.0) ---
    long = sales.unpivot(
        index=id_vars,
        on=day_cols,
        variable_name="d",
        value_name="sales",
    )

    # Cast sales to Int64 (may come as float after melt if NaN present)
    long = long.with_columns(pl.col("sales").cast(pl.Int64))

    # --- Join with calendar to get real dates ---
    # Calendar has columns: date (str), d (str like "d_1"), wm_yr_wk, ...
    cal_slim = calendar.select(["d", "date"]).with_columns(
        pl.col("date").str.to_date("%Y-%m-%d").alias("date")
    )
    long = long.join(cal_slim, on="d", how="left")

    # --- Derive silver columns ---
    long = long.with_columns(
        # Clean id: "{item_id}_{store_id}" (drop competition suffix)
        (pl.col("item_id") + "_" + pl.col("store_id")).alias("id"),
        (pl.col("sales") == 0).alias("is_zero_sale"),
    )

    # --- Select and reorder to silver schema ---
    silver_cols = ["id", "date", "store_id", "item_id", "state_id", "dept_id", "cat_id",
                   "sales", "is_zero_sale"]
    long = long.select(silver_cols)

    # Drop rows where date is null (calendar mismatch — should not happen)
    long = long.filter(pl.col("date").is_not_null())

    # --- Write partitions ---
    _write_partition(long, silver_sales_dir)


def read_silver_sales(silver_sales_dir: Path) -> pl.LazyFrame:
    """Read all silver sales partitions as a single LazyFrame.

    Parameters
    ----------
    silver_sales_dir:
        Root of the partitioned silver output.

    Returns
    -------
    pl.LazyFrame
        Concatenated LazyFrame over all partitions.
    """
    parquet_files = sorted(silver_sales_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No silver sales partitions found in {silver_sales_dir}")
    return pl.scan_parquet([str(p) for p in parquet_files])


# ---------------------------------------------------------------------------
# 2. Prices: weekly → daily
# ---------------------------------------------------------------------------


def prices_weekly_to_daily(
    bronze_prices_path: Path,
    bronze_calendar_path: Path,
    silver_prices_path: Path,
    *,
    force: bool = False,
) -> None:
    """Expand weekly sell-prices to daily grain with forward-fill.

    The M5 price table has grain ``(store_id, item_id, wm_yr_wk)``.  We join
    with the calendar to get one row per ``(store_id, item_id, date)`` and
    forward-fill within each series to cover every calendar day.

    Leakage note: forward-fill only propagates *past* known prices forward,
    never backward — this is safe for backtesting.

    Parameters
    ----------
    bronze_prices_path:
        Path to ``bronze_prices.parquet``.
    bronze_calendar_path:
        Path to ``bronze_calendar.parquet``.
    silver_prices_path:
        Output path for ``silver_prices_daily.parquet``.
    force:
        Re-process even when the output already exists.
    """
    if silver_prices_path.exists() and not force:
        return

    prices = pl.read_parquet(bronze_prices_path)
    calendar = pl.read_parquet(bronze_calendar_path)

    # Calendar: one row per day, with wm_yr_wk for joining
    cal_dates = (
        calendar
        .select(["date", "wm_yr_wk"])
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("date"))
        .unique()
    )

    # Expand prices to daily: each (store, item, wm_yr_wk) → N days in that week
    prices = prices.with_columns(pl.col("wm_yr_wk").cast(pl.Int64))
    cal_dates = cal_dates.with_columns(pl.col("wm_yr_wk").cast(pl.Int64))

    daily = prices.join(cal_dates, on="wm_yr_wk", how="left")

    # Sort for correct forward-fill order
    daily = daily.sort(["store_id", "item_id", "date"])

    # Forward-fill sell_price within (store_id, item_id)
    daily = daily.with_columns(
        pl.col("sell_price")
        .forward_fill()
        .over(["store_id", "item_id"])
    )

    # Final schema: (date, store_id, item_id, sell_price)
    daily = daily.select(["date", "store_id", "item_id", "sell_price"])

    silver_prices_path.parent.mkdir(parents=True, exist_ok=True)
    daily.write_parquet(silver_prices_path, compression="snappy")


# ---------------------------------------------------------------------------
# 3. Calendar enrichment
# ---------------------------------------------------------------------------


def enrich_calendar(
    bronze_calendar_path: Path,
    silver_calendar_path: Path,
    *,
    force: bool = False,
) -> None:
    """Add derived columns to the calendar and write silver_calendar_enriched.parquet.

    Derived columns added:
    - ``date``       : pl.Date (parsed from string)
    - ``is_weekend`` : bool — True when ``wday`` ∈ {1, 2} (Sat/Sun in M5 encoding)
    - ``quarter``    : int  — calendar quarter [1, 4]

    M5 ``wday`` encoding: 1=Saturday, 2=Sunday, 3=Monday, …, 7=Friday.

    Parameters
    ----------
    bronze_calendar_path:
        Path to ``bronze_calendar.parquet``.
    silver_calendar_path:
        Output path for ``silver_calendar_enriched.parquet``.
    force:
        Re-process even when the output already exists.
    """
    if silver_calendar_path.exists() and not force:
        return

    cal = pl.read_parquet(bronze_calendar_path)

    cal = cal.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d").alias("date"),
    )

    cal = cal.with_columns(
        pl.col("wday").cast(pl.Int64),
        pl.col("month").cast(pl.Int64),
        pl.col("year").cast(pl.Int64),
        pl.col("wm_yr_wk").cast(pl.Int64),
    )

    # is_weekend: wday 1 (Saturday) or 2 (Sunday) in M5 encoding
    cal = cal.with_columns(
        pl.col("wday").is_in([1, 2]).alias("is_weekend"),
        ((pl.col("month") - 1) // 3 + 1).alias("quarter"),
    )

    silver_calendar_path.parent.mkdir(parents=True, exist_ok=True)
    cal.write_parquet(silver_calendar_path, compression="snappy")


# ---------------------------------------------------------------------------
# 4. Weather: cast to typed silver
# ---------------------------------------------------------------------------


def build_silver_weather(
    bronze_weather_path: Path,
    silver_weather_path: Path,
    *,
    force: bool = False,
) -> None:
    """Cast the bronze weather table's date column to pl.Date.

    Parameters
    ----------
    bronze_weather_path:
        Path to ``bronze_weather.parquet``.
    silver_weather_path:
        Output path for ``silver_weather_daily.parquet``.
    force:
        Re-process even when the output already exists.
    """
    if silver_weather_path.exists() and not force:
        return

    weather = pl.read_parquet(bronze_weather_path)

    weather = weather.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d").alias("date"),
    )

    # Coerce numeric columns
    numeric_cols = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "weathercode",
    ]
    existing_numeric = [c for c in numeric_cols if c in weather.columns]
    for col in existing_numeric:
        weather = weather.with_columns(pl.col(col).cast(pl.Float64))

    silver_weather_path.parent.mkdir(parents=True, exist_ok=True)
    weather.write_parquet(silver_weather_path, compression="snappy")


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------


def run_bronze_to_silver(
    bronze_dir: Path,
    silver_dir: Path,
    *,
    force: bool = False,
) -> None:
    """Run the end-to-end bronze → silver transformation pipeline.

    Executes all four transformations in dependency order:
    1. Calendar enrichment  (no upstream silver dependency)
    2. Prices weekly → daily (depends on calendar)
    3. Sales wide → long    (depends on calendar)
    4. Weather → typed      (independent)

    Parameters
    ----------
    bronze_dir:
        Directory containing bronze Parquet files.
    silver_dir:
        Root directory where silver artefacts will be written.
    force:
        Re-run all steps even if outputs already exist.
    """
    silver_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Calendar
    enrich_calendar(
        bronze_dir / "bronze_calendar.parquet",
        silver_dir / "silver_calendar_enriched.parquet",
        force=force,
    )

    # Step 2: Prices (depends on calendar)
    prices_weekly_to_daily(
        bronze_dir / "bronze_prices.parquet",
        bronze_dir / "bronze_calendar.parquet",
        silver_dir / "silver_prices_daily.parquet",
        force=force,
    )

    # Step 3: Sales (depends on calendar)
    sales_wide_to_long(
        bronze_dir / "bronze_sales.parquet",
        bronze_dir / "bronze_calendar.parquet",
        silver_dir / "silver_sales_long",
        force=force,
    )

    # Step 4: Weather
    bronze_weather = bronze_dir / "bronze_weather.parquet"
    if bronze_weather.exists():
        build_silver_weather(
            bronze_weather,
            silver_dir / "silver_weather_daily.parquet",
            force=force,
        )
