"""Unit tests for src/validation/contracts.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandera.errors
import polars as pl
import pytest
import yaml

from src.validation.contracts import (
    BronzeCalendarSchema,
    BronzePricesSchema,
    BronzeSalesSchema,
    BronzeWeatherSchema,
    SilverCalendarSchema,
    SilverPricesDailySchema,
    SilverSalesSchema,
    SilverWeatherSchema,
    load_contract,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# load_contract
# ---------------------------------------------------------------------------


def test_load_contract_returns_dict_for_silver_sales() -> None:
    contract = load_contract("silver_sales")
    assert isinstance(contract, dict)
    assert "grain" in contract


def test_load_contract_with_yaml_extension() -> None:
    contract = load_contract("silver_sales.yaml")
    assert "grain" in contract


def test_load_contract_raises_for_unknown_name() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_contract("nonexistent_contract")


def test_load_contract_custom_base_path(tmp_path: Path) -> None:
    (tmp_path / "my_schema.yaml").write_text("grain: [date]\n")
    result = load_contract("my_schema", base_path=tmp_path)
    assert result == {"grain": ["date"]}


# ---------------------------------------------------------------------------
# BronzeSalesSchema
# ---------------------------------------------------------------------------


def _minimal_bronze_sales() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1_evaluation"],
            "item_id": ["FOODS_1_001"],
            "dept_id": ["FOODS_1"],
            "cat_id": ["FOODS"],
            "store_id": ["CA_1"],
            "state_id": ["CA"],
            "d_1": [0],
            "d_2": [1],
        }
    )


def test_bronze_sales_schema_valid() -> None:
    df = _minimal_bronze_sales()
    validated = BronzeSalesSchema.validate(df)
    assert len(validated) == 1


def test_bronze_sales_schema_rejects_null_id() -> None:
    df = _minimal_bronze_sales().with_columns(pl.lit(None).cast(pl.Utf8).alias("id"))
    with pytest.raises(Exception):
        BronzeSalesSchema.validate(df)


# ---------------------------------------------------------------------------
# BronzeCalendarSchema
# ---------------------------------------------------------------------------


def _minimal_bronze_calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": ["2011-01-29"],
            "wm_yr_wk": [11101],
            "weekday": ["Saturday"],
            "wday": [1],
            "month": [1],
            "year": [2011],
            "d": ["d_1"],
        }
    )


def test_bronze_calendar_schema_valid() -> None:
    df = _minimal_bronze_calendar()
    validated = BronzeCalendarSchema.validate(df)
    assert len(validated) == 1


def test_bronze_calendar_schema_rejects_invalid_month() -> None:
    df = _minimal_bronze_calendar().with_columns(pl.lit(13).alias("month"))
    with pytest.raises(Exception):
        BronzeCalendarSchema.validate(df)


# ---------------------------------------------------------------------------
# BronzePricesSchema
# ---------------------------------------------------------------------------


def _minimal_bronze_prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "store_id": ["CA_1"],
            "item_id": ["FOODS_1_001"],
            "wm_yr_wk": [11101],
            "sell_price": [1.99],
        }
    )


def test_bronze_prices_schema_valid() -> None:
    df = _minimal_bronze_prices()
    validated = BronzePricesSchema.validate(df)
    assert len(validated) == 1


def test_bronze_prices_schema_rejects_negative_price() -> None:
    df = _minimal_bronze_prices().with_columns(pl.lit(-1.0).alias("sell_price"))
    with pytest.raises(Exception):
        BronzePricesSchema.validate(df)


def test_bronze_prices_schema_allows_null_price() -> None:
    df = _minimal_bronze_prices().with_columns(
        pl.lit(None).cast(pl.Float64).alias("sell_price")
    )
    validated = BronzePricesSchema.validate(df)
    assert len(validated) == 1


# ---------------------------------------------------------------------------
# BronzeWeatherSchema
# ---------------------------------------------------------------------------


def _minimal_bronze_weather() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "state": ["CA"],
            "date": ["2011-01-29"],
            "temperature_2m_max": [20.1],
            "temperature_2m_min": [10.2],
            "temperature_2m_mean": [15.3],
            "precipitation_sum": [0.0],
            "weathercode": [1.0],
        }
    )


def test_bronze_weather_schema_valid() -> None:
    df = _minimal_bronze_weather()
    validated = BronzeWeatherSchema.validate(df)
    assert len(validated) == 1


def test_bronze_weather_schema_rejects_invalid_state() -> None:
    df = _minimal_bronze_weather().with_columns(pl.lit("NY").alias("state"))
    with pytest.raises(Exception):
        BronzeWeatherSchema.validate(df)


# ---------------------------------------------------------------------------
# SilverSalesSchema
# ---------------------------------------------------------------------------


def _minimal_silver_sales() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["FOODS_1_001_CA_1"],
            "date": [date(2011, 1, 29)],
            "store_id": ["CA_1"],
            "item_id": ["FOODS_1_001"],
            "state_id": ["CA"],
            "dept_id": ["FOODS_1"],
            "cat_id": ["FOODS"],
            "sales": [0],
            "is_zero_sale": [True],
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def test_silver_sales_schema_valid() -> None:
    df = _minimal_silver_sales()
    validated = SilverSalesSchema.validate(df)
    assert len(validated) == 1


def test_silver_sales_schema_rejects_negative_sales() -> None:
    df = _minimal_silver_sales().with_columns(pl.lit(-1).alias("sales"))
    with pytest.raises(Exception):
        SilverSalesSchema.validate(df)


def test_silver_sales_schema_rejects_invalid_state() -> None:
    df = _minimal_silver_sales().with_columns(pl.lit("FL").alias("state_id"))
    with pytest.raises(Exception):
        SilverSalesSchema.validate(df)


def test_silver_sales_schema_rejects_invalid_cat() -> None:
    df = _minimal_silver_sales().with_columns(pl.lit("ELECTRONICS").alias("cat_id"))
    with pytest.raises(Exception):
        SilverSalesSchema.validate(df)


# ---------------------------------------------------------------------------
# SilverCalendarSchema
# ---------------------------------------------------------------------------


def _minimal_silver_calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2011, 1, 29)],
            "wm_yr_wk": [11101],
            "weekday": ["Saturday"],
            "wday": [1],
            "month": [1],
            "year": [2011],
            "d": ["d_1"],
            "is_weekend": [True],
            "quarter": [1],
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def test_silver_calendar_schema_valid() -> None:
    validated = SilverCalendarSchema.validate(_minimal_silver_calendar())
    assert len(validated) == 1


def test_silver_calendar_schema_rejects_invalid_quarter() -> None:
    df = _minimal_silver_calendar().with_columns(pl.lit(5).alias("quarter"))
    with pytest.raises(Exception):
        SilverCalendarSchema.validate(df)


# ---------------------------------------------------------------------------
# SilverWeatherSchema
# ---------------------------------------------------------------------------


def _minimal_silver_weather() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [date(2011, 1, 29)],
            "state": ["CA"],
            "temperature_2m_max": [20.1],
            "temperature_2m_min": [10.2],
            "temperature_2m_mean": [15.3],
            "precipitation_sum": [0.0],
            "weathercode": [1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Date))


def test_silver_weather_schema_valid() -> None:
    validated = SilverWeatherSchema.validate(_minimal_silver_weather())
    assert len(validated) == 1


# ---------------------------------------------------------------------------
# validate_dataframe (generic entry-point)
# ---------------------------------------------------------------------------


def test_validate_dataframe_dispatches_to_correct_schema() -> None:
    df = _minimal_silver_sales()
    result = validate_dataframe(df, "silver_sales")
    assert len(result) == len(df)


def test_validate_dataframe_raises_for_unknown_contract() -> None:
    df = pl.DataFrame({"x": [1]})
    with pytest.raises(KeyError, match="Unknown contract"):
        validate_dataframe(df, "made_up_contract")


def test_validate_dataframe_bronze_sales() -> None:
    df = _minimal_bronze_sales()
    result = validate_dataframe(df, "bronze_sales")
    assert len(result) == len(df)
