"""Unit tests for src/transform/pipeline.py.

All tests use small synthetic DataFrames written to tmp_path, so no real
M5 or weather data is required.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from src.transform.pipeline import (
    _day_columns,
    build_silver_weather,
    enrich_calendar,
    prices_weekly_to_daily,
    read_silver_sales,
    run_bronze_to_silver,
    sales_wide_to_long,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal bronze DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_bronze_sales(tmp_path: Path) -> Path:
    """Write a minimal bronze_sales.parquet with 2 items × 3 days."""
    df = pl.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_evaluation", "FOODS_1_002_CA_1_evaluation"],
            "item_id": ["FOODS_1_001", "FOODS_1_002"],
            "dept_id": ["FOODS_1", "FOODS_1"],
            "cat_id": ["FOODS", "FOODS"],
            "store_id": ["CA_1", "CA_1"],
            "state_id": ["CA", "CA"],
            "d_1": [0, 1],
            "d_2": [2, 0],
            "d_3": [1, 3],
        }
    )
    p = tmp_path / "bronze_sales.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture
def minimal_bronze_calendar(tmp_path: Path) -> Path:
    """Write a minimal bronze_calendar.parquet aligned with 3 days."""
    df = pl.DataFrame(
        {
            "date": ["2011-01-29", "2011-01-30", "2011-01-31"],
            "wm_yr_wk": [11101, 11101, 11101],
            "weekday": ["Saturday", "Sunday", "Monday"],
            "wday": [1, 2, 3],
            "month": [1, 1, 1],
            "year": [2011, 2011, 2011],
            "d": ["d_1", "d_2", "d_3"],
            "event_name_1": [None, None, None],
            "event_type_1": [None, None, None],
            "snap_CA": [0, 0, 1],
            "snap_TX": [0, 0, 0],
            "snap_WI": [0, 0, 0],
        }
    )
    p = tmp_path / "bronze_calendar.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture
def minimal_bronze_prices(tmp_path: Path) -> Path:
    """Write a minimal bronze_prices.parquet."""
    df = pl.DataFrame(
        {
            "store_id": ["CA_1", "CA_1"],
            "item_id": ["FOODS_1_001", "FOODS_1_002"],
            "wm_yr_wk": [11101, 11101],
            "sell_price": [1.99, 2.49],
        }
    )
    p = tmp_path / "bronze_prices.parquet"
    df.write_parquet(p)
    return p


@pytest.fixture
def minimal_bronze_weather(tmp_path: Path) -> Path:
    """Write a minimal bronze_weather.parquet."""
    df = pl.DataFrame(
        {
            "state": ["CA", "CA", "TX"],
            "date": ["2011-01-29", "2011-01-30", "2011-01-29"],
            "temperature_2m_max": [20.1, 21.0, 18.5],
            "temperature_2m_min": [10.2, 11.0, 9.0],
            "temperature_2m_mean": [15.3, 16.0, 13.8],
            "precipitation_sum": [0.0, 0.5, 0.0],
            "weathercode": [1.0, 2.0, 0.0],
        }
    )
    p = tmp_path / "bronze_weather.parquet"
    df.write_parquet(p)
    return p


# ---------------------------------------------------------------------------
# _day_columns
# ---------------------------------------------------------------------------


def test_day_columns_identifies_d_n_columns() -> None:
    df = pl.DataFrame({"id": ["x"], "item_id": ["y"], "d_1": [0], "d_2": [1], "d_10": [2]})
    assert _day_columns(df) == ["d_1", "d_2", "d_10"]


def test_day_columns_excludes_non_day_columns() -> None:
    df = pl.DataFrame({"id": ["x"], "dept_id": ["y"], "d_abc": ["z"], "d_1": [0]})
    # "d_abc" should NOT be included (not integer suffix)
    result = _day_columns(df)
    assert "d_abc" not in result
    assert "d_1" in result


# ---------------------------------------------------------------------------
# sales_wide_to_long
# ---------------------------------------------------------------------------


def test_sales_wide_to_long_creates_partitions(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    partitions = list(out_dir.rglob("*.parquet"))
    assert len(partitions) >= 1


def test_sales_wide_to_long_row_count(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    lf = read_silver_sales(out_dir)
    result = lf.collect()
    # 2 items × 3 days = 6 rows
    assert len(result) == 6


def test_sales_wide_to_long_has_correct_columns(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    df = read_silver_sales(out_dir).collect()
    expected = {"id", "date", "store_id", "item_id", "state_id", "dept_id", "cat_id",
                "sales", "is_zero_sale"}
    assert expected.issubset(set(df.columns))


def test_sales_wide_to_long_id_format(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    df = read_silver_sales(out_dir).collect()
    ids = df["id"].unique().sort().to_list()
    # id should be "{item_id}_{store_id}"
    assert "FOODS_1_001_CA_1" in ids
    assert "FOODS_1_002_CA_1" in ids


def test_sales_wide_to_long_is_zero_sale(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    df = read_silver_sales(out_dir).collect()
    # Verify is_zero_sale == (sales == 0)
    mismatch = df.filter(pl.col("is_zero_sale") != (pl.col("sales") == 0))
    assert len(mismatch) == 0


def test_sales_wide_to_long_date_type(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    df = read_silver_sales(out_dir).collect()
    assert df["date"].dtype == pl.Date


def test_sales_wide_to_long_nonnegative_sales(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    df = read_silver_sales(out_dir).collect()
    assert df["sales"].min() >= 0


def test_sales_wide_to_long_is_idempotent(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)
    mtime = list(out_dir.rglob("*.parquet"))[0].stat().st_mtime

    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)
    mtime2 = list(out_dir.rglob("*.parquet"))[0].stat().st_mtime
    # Should not re-write
    assert mtime == mtime2


def test_sales_wide_to_long_partition_structure(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out_dir = tmp_path / "silver_sales_long"
    sales_wide_to_long(minimal_bronze_sales, minimal_bronze_calendar, out_dir)

    # Must have state=CA/year=2011/data.parquet
    expected = out_dir / "state=CA" / "year=2011" / "data.parquet"
    assert expected.exists(), f"Expected partition file {expected}"


# ---------------------------------------------------------------------------
# prices_weekly_to_daily
# ---------------------------------------------------------------------------


def test_prices_weekly_to_daily_creates_output(
    tmp_path: Path,
    minimal_bronze_prices: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_prices_daily.parquet"
    prices_weekly_to_daily(minimal_bronze_prices, minimal_bronze_calendar, out)
    assert out.exists()


def test_prices_weekly_to_daily_row_count(
    tmp_path: Path,
    minimal_bronze_prices: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_prices_daily.parquet"
    prices_weekly_to_daily(minimal_bronze_prices, minimal_bronze_calendar, out)

    df = pl.read_parquet(out)
    # 2 items × 3 days = 6 rows (calendar has 3 days all in wm_yr_wk=11101)
    assert len(df) == 6


def test_prices_weekly_to_daily_schema(
    tmp_path: Path,
    minimal_bronze_prices: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_prices_daily.parquet"
    prices_weekly_to_daily(minimal_bronze_prices, minimal_bronze_calendar, out)

    df = pl.read_parquet(out)
    assert set(df.columns) == {"date", "store_id", "item_id", "sell_price"}
    assert df["date"].dtype == pl.Date


def test_prices_weekly_to_daily_no_null_prices_after_ffill(
    tmp_path: Path,
    minimal_bronze_prices: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_prices_daily.parquet"
    prices_weekly_to_daily(minimal_bronze_prices, minimal_bronze_calendar, out)

    df = pl.read_parquet(out)
    # All prices should be non-null (single week in fixture, prices available)
    assert df["sell_price"].null_count() == 0


def test_prices_weekly_to_daily_is_idempotent(
    tmp_path: Path,
    minimal_bronze_prices: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_prices_daily.parquet"
    prices_weekly_to_daily(minimal_bronze_prices, minimal_bronze_calendar, out)
    mtime = out.stat().st_mtime

    prices_weekly_to_daily(minimal_bronze_prices, minimal_bronze_calendar, out)
    assert out.stat().st_mtime == mtime


# ---------------------------------------------------------------------------
# enrich_calendar
# ---------------------------------------------------------------------------


def test_enrich_calendar_creates_output(
    tmp_path: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_calendar_enriched.parquet"
    enrich_calendar(minimal_bronze_calendar, out)
    assert out.exists()


def test_enrich_calendar_has_derived_columns(
    tmp_path: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_calendar_enriched.parquet"
    enrich_calendar(minimal_bronze_calendar, out)

    df = pl.read_parquet(out)
    assert "is_weekend" in df.columns
    assert "quarter" in df.columns
    assert df["date"].dtype == pl.Date


def test_enrich_calendar_is_weekend_logic(
    tmp_path: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_calendar_enriched.parquet"
    enrich_calendar(minimal_bronze_calendar, out)

    df = pl.read_parquet(out)
    # wday=1 (Sat) and wday=2 (Sun) → is_weekend=True; wday=3 (Mon) → False
    by_wday = df.select(["wday", "is_weekend"]).to_dicts()
    wday_map = {row["wday"]: row["is_weekend"] for row in by_wday}
    assert wday_map[1] is True   # Saturday
    assert wday_map[2] is True   # Sunday
    assert wday_map[3] is False  # Monday


def test_enrich_calendar_quarter_for_january(
    tmp_path: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_calendar_enriched.parquet"
    enrich_calendar(minimal_bronze_calendar, out)

    df = pl.read_parquet(out)
    quarters = df["quarter"].unique().to_list()
    assert all(q == 1 for q in quarters)  # All rows are in January


def test_enrich_calendar_is_idempotent(
    tmp_path: Path,
    minimal_bronze_calendar: Path,
) -> None:
    out = tmp_path / "silver_calendar_enriched.parquet"
    enrich_calendar(minimal_bronze_calendar, out)
    mtime = out.stat().st_mtime

    enrich_calendar(minimal_bronze_calendar, out)
    assert out.stat().st_mtime == mtime


# ---------------------------------------------------------------------------
# build_silver_weather
# ---------------------------------------------------------------------------


def test_build_silver_weather_creates_output(
    tmp_path: Path,
    minimal_bronze_weather: Path,
) -> None:
    out = tmp_path / "silver_weather_daily.parquet"
    build_silver_weather(minimal_bronze_weather, out)
    assert out.exists()


def test_build_silver_weather_date_is_polars_date(
    tmp_path: Path,
    minimal_bronze_weather: Path,
) -> None:
    out = tmp_path / "silver_weather_daily.parquet"
    build_silver_weather(minimal_bronze_weather, out)

    df = pl.read_parquet(out)
    assert df["date"].dtype == pl.Date


def test_build_silver_weather_preserves_row_count(
    tmp_path: Path,
    minimal_bronze_weather: Path,
) -> None:
    out = tmp_path / "silver_weather_daily.parquet"
    build_silver_weather(minimal_bronze_weather, out)

    df = pl.read_parquet(out)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# run_bronze_to_silver (orchestrator)
# ---------------------------------------------------------------------------


def test_run_bronze_to_silver_creates_all_outputs(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
    minimal_bronze_prices: Path,
    minimal_bronze_weather: Path,
) -> None:
    # Copy all bronze files to a single bronze_dir
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()

    import shutil
    shutil.copy(minimal_bronze_sales, bronze_dir / "bronze_sales.parquet")
    shutil.copy(minimal_bronze_calendar, bronze_dir / "bronze_calendar.parquet")
    shutil.copy(minimal_bronze_prices, bronze_dir / "bronze_prices.parquet")
    shutil.copy(minimal_bronze_weather, bronze_dir / "bronze_weather.parquet")

    silver_dir = tmp_path / "silver"
    run_bronze_to_silver(bronze_dir, silver_dir)

    assert (silver_dir / "silver_calendar_enriched.parquet").exists()
    assert (silver_dir / "silver_prices_daily.parquet").exists()
    assert any((silver_dir / "silver_sales_long").rglob("*.parquet"))
    assert (silver_dir / "silver_weather_daily.parquet").exists()


def test_run_bronze_to_silver_is_idempotent(
    tmp_path: Path,
    minimal_bronze_sales: Path,
    minimal_bronze_calendar: Path,
    minimal_bronze_prices: Path,
) -> None:
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()

    import shutil
    shutil.copy(minimal_bronze_sales, bronze_dir / "bronze_sales.parquet")
    shutil.copy(minimal_bronze_calendar, bronze_dir / "bronze_calendar.parquet")
    shutil.copy(minimal_bronze_prices, bronze_dir / "bronze_prices.parquet")

    silver_dir = tmp_path / "silver"
    run_bronze_to_silver(bronze_dir, silver_dir)
    mtime = (silver_dir / "silver_calendar_enriched.parquet").stat().st_mtime

    run_bronze_to_silver(bronze_dir, silver_dir)
    assert (silver_dir / "silver_calendar_enriched.parquet").stat().st_mtime == mtime
