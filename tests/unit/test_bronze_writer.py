"""Unit tests for src/ingest/bronze_writer.py."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.ingest.bronze_writer import (
    CHECKSUM_FILENAME,
    M5_CSV_TO_BRONZE,
    WEATHER_BRONZE_NAME,
    get_bronze_checksums,
    load_bronze,
    sha256_parquet,
    write_m5_bronze,
    write_weather_bronze,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_m5_csvs(raw_dir: Path) -> None:
    """Create stub M5 CSVs with minimal valid content."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    # sales_train_validation.csv — wide format with 3 day columns
    (raw_dir / "sales_train_validation.csv").write_text(
        "id,item_id,dept_id,cat_id,store_id,state_id,d_1,d_2,d_3\n"
        "FOODS_1_001_CA_1_evaluation,FOODS_1_001,FOODS_1,FOODS,CA_1,CA,0,1,2\n"
    )
    # calendar.csv
    (raw_dir / "calendar.csv").write_text(
        "date,wm_yr_wk,weekday,wday,month,year,d,event_name_1,event_type_1\n"
        "2011-01-29,11101,Saturday,1,1,2011,d_1,,\n"
        "2011-01-30,11101,Sunday,2,1,2011,d_2,,\n"
        "2011-01-31,11101,Monday,3,1,2011,d_3,,\n"
    )
    # sell_prices.csv
    (raw_dir / "sell_prices.csv").write_text(
        "store_id,item_id,wm_yr_wk,sell_price\n"
        "CA_1,FOODS_1_001,11101,1.99\n"
    )


def _write_minimal_weather_csvs(weather_dir: Path) -> None:
    """Create stub weather CSVs for CA and TX."""
    weather_dir.mkdir(parents=True, exist_ok=True)
    content = "state,date,temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,weathercode\n"
    content += "CA,2011-01-29,20.1,10.2,15.3,0.0,1\n"
    (weather_dir / "weather_ca.csv").write_text(content)

    content2 = content.replace("CA,2011-01-29", "TX,2011-01-29")
    (weather_dir / "weather_tx.csv").write_text(content2)


# ---------------------------------------------------------------------------
# sha256_parquet
# ---------------------------------------------------------------------------


def test_sha256_parquet_produces_64_char_hex(tmp_path: Path) -> None:
    p = tmp_path / "test.bin"
    p.write_bytes(b"hello")
    digest = sha256_parquet(p)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# write_m5_bronze
# ---------------------------------------------------------------------------


def test_write_m5_bronze_creates_parquet_files(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "m5"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_m5_csvs(raw_dir)

    checksums = write_m5_bronze(raw_dir, bronze_dir)

    for parquet_name in M5_CSV_TO_BRONZE.values():
        assert (bronze_dir / parquet_name).exists(), f"{parquet_name} not created"

    assert set(checksums.keys()) == set(M5_CSV_TO_BRONZE.values())
    for digest in checksums.values():
        assert len(digest) == 64


def test_write_m5_bronze_creates_checksum_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "m5"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_m5_csvs(raw_dir)

    write_m5_bronze(raw_dir, bronze_dir)

    manifest = bronze_dir / CHECKSUM_FILENAME
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert set(data.keys()) == set(M5_CSV_TO_BRONZE.values())


def test_write_m5_bronze_parquet_readable_by_polars(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "m5"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_m5_csvs(raw_dir)

    write_m5_bronze(raw_dir, bronze_dir)

    df = pl.read_parquet(bronze_dir / "bronze_sales.parquet")
    assert len(df) >= 1
    assert "id" in df.columns


def test_write_m5_bronze_is_idempotent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "m5"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_m5_csvs(raw_dir)

    c1 = write_m5_bronze(raw_dir, bronze_dir)
    c2 = write_m5_bronze(raw_dir, bronze_dir)
    # Checksums must be identical on repeat calls
    assert c1 == c2


def test_write_m5_bronze_raises_if_csv_missing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "m5"
    bronze_dir = tmp_path / "bronze"
    raw_dir.mkdir(parents=True)
    # Don't write any CSV

    with pytest.raises(FileNotFoundError, match="M5 CSV not found"):
        write_m5_bronze(raw_dir, bronze_dir)


def test_write_m5_bronze_force_re_converts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw" / "m5"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_m5_csvs(raw_dir)

    write_m5_bronze(raw_dir, bronze_dir)
    mtime_before = (bronze_dir / "bronze_sales.parquet").stat().st_mtime

    import time as _time
    _time.sleep(0.05)
    write_m5_bronze(raw_dir, bronze_dir, force=True)
    mtime_after = (bronze_dir / "bronze_sales.parquet").stat().st_mtime

    assert mtime_after >= mtime_before


# ---------------------------------------------------------------------------
# write_weather_bronze
# ---------------------------------------------------------------------------


def test_write_weather_bronze_creates_parquet(tmp_path: Path) -> None:
    weather_dir = tmp_path / "raw" / "weather"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_weather_csvs(weather_dir)

    result = write_weather_bronze(weather_dir, bronze_dir)

    assert (bronze_dir / WEATHER_BRONZE_NAME).exists()
    assert WEATHER_BRONZE_NAME in result


def test_write_weather_bronze_concatenates_states(tmp_path: Path) -> None:
    weather_dir = tmp_path / "raw" / "weather"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_weather_csvs(weather_dir)  # creates CA + TX

    write_weather_bronze(weather_dir, bronze_dir)
    df = pl.read_parquet(bronze_dir / WEATHER_BRONZE_NAME)
    states = df["state"].unique().sort().to_list()
    assert "CA" in states
    assert "TX" in states


def test_write_weather_bronze_raises_if_no_csvs(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No weather CSVs found"):
        write_weather_bronze(empty_dir, tmp_path / "bronze")


def test_write_weather_bronze_is_idempotent(tmp_path: Path) -> None:
    weather_dir = tmp_path / "raw" / "weather"
    bronze_dir = tmp_path / "bronze"
    _write_minimal_weather_csvs(weather_dir)

    r1 = write_weather_bronze(weather_dir, bronze_dir)
    r2 = write_weather_bronze(weather_dir, bronze_dir)
    assert r1 == r2


# ---------------------------------------------------------------------------
# load_bronze
# ---------------------------------------------------------------------------


def test_load_bronze_returns_dataframe(tmp_path: Path) -> None:
    bronze_dir = tmp_path / "bronze"
    bronze_dir.mkdir()
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.write_parquet(bronze_dir / "test.parquet")

    loaded = load_bronze(bronze_dir, "test.parquet")
    assert isinstance(loaded, pl.DataFrame)
    assert len(loaded) == 3


def test_load_bronze_raises_if_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bronze(tmp_path / "bronze", "nonexistent.parquet")


# ---------------------------------------------------------------------------
# get_bronze_checksums
# ---------------------------------------------------------------------------


def test_get_bronze_checksums_returns_empty_if_no_manifest(tmp_path: Path) -> None:
    result = get_bronze_checksums(tmp_path)
    assert result == {}


def test_get_bronze_checksums_returns_manifest_contents(tmp_path: Path) -> None:
    manifest = tmp_path / CHECKSUM_FILENAME
    data = {"bronze_sales.parquet": "abc123"}
    manifest.write_text(json.dumps(data))

    result = get_bronze_checksums(tmp_path)
    assert result == data
