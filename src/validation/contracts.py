"""Pandera data-contract schemas for bronze and silver layers.

Each schema is defined as a ``pandera.polars.DataFrameModel`` subclass so
validation operates directly on Polars DataFrames (no Pandas conversion).

Schemas correspond to the YAML contracts in ``contracts/``:
  - ``contracts/silver_sales.yaml``  →  :class:`SilverSalesSchema`

Bronze schemas are lighter (raw data, minimal constraints) while silver
schemas enforce full business invariants.

Usage
-----
>>> from src.validation.contracts import SilverSalesSchema
>>> SilverSalesSchema.validate(df)   # raises SchemaError if invalid
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Optional

import pandera.polars as pa
import polars as pl
import yaml


# ---------------------------------------------------------------------------
# Utility: YAML contract loader
# ---------------------------------------------------------------------------

_CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "contracts"


def load_contract(name: str, base_path: Path | None = None) -> Mapping[str, Any]:
    """Load a YAML contract definition by name.

    Parameters
    ----------
    name:
        Logical contract name, e.g. ``"silver_sales"`` or
        ``"silver_sales.yaml"``.
    base_path:
        Override the default ``contracts/`` directory.

    Returns
    -------
    Mapping[str, Any]
        Parsed YAML content as a plain dict.

    Raises
    ------
    FileNotFoundError
        If the contract YAML does not exist.
    """
    base = base_path or _CONTRACTS_DIR
    stem = name.removesuffix(".yaml")
    candidates = [base / f"{stem}.yaml", base / name]
    for c in candidates:
        if c.exists():
            return yaml.safe_load(c.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"Contract {name!r} not found in {base}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Bronze schemas — lightweight, raw-layer checks
# ---------------------------------------------------------------------------


class BronzeSalesSchema(pa.DataFrameModel):
    """Minimal schema for ``bronze_sales.parquet``.

    Reflects the wide-format M5 sales file before melting.  Only structural
    invariants are enforced here; value constraints live in the silver schema.
    """

    id: str = pa.Field(nullable=False)
    item_id: str = pa.Field(nullable=False)
    dept_id: str = pa.Field(nullable=False)
    cat_id: str = pa.Field(nullable=False)
    store_id: str = pa.Field(nullable=False)
    state_id: str = pa.Field(nullable=False)

    class Config:
        # Additional d_N columns are allowed
        add_missing_columns = False
        coerce = False


class BronzeCalendarSchema(pa.DataFrameModel):
    """Schema for ``bronze_calendar.parquet``."""

    date: str = pa.Field(nullable=False)
    wm_yr_wk: int = pa.Field(nullable=False, ge=11100)
    weekday: str = pa.Field(nullable=False)
    wday: int = pa.Field(nullable=False, ge=1, le=7)
    month: int = pa.Field(nullable=False, ge=1, le=12)
    year: int = pa.Field(nullable=False, ge=2011, le=2016)
    d: str = pa.Field(nullable=False)

    class Config:
        add_missing_columns = False
        coerce = True


class BronzePricesSchema(pa.DataFrameModel):
    """Schema for ``bronze_prices.parquet``."""

    store_id: str = pa.Field(nullable=False)
    item_id: str = pa.Field(nullable=False)
    wm_yr_wk: int = pa.Field(nullable=False, ge=11100)
    sell_price: float = pa.Field(nullable=True, gt=0)

    class Config:
        coerce = True


class BronzeWeatherSchema(pa.DataFrameModel):
    """Schema for ``bronze_weather.parquet``."""

    state: str = pa.Field(nullable=False, isin=["CA", "TX", "WI"])
    date: str = pa.Field(nullable=False)
    temperature_2m_max: Optional[float] = pa.Field(nullable=True)
    temperature_2m_min: Optional[float] = pa.Field(nullable=True)
    temperature_2m_mean: Optional[float] = pa.Field(nullable=True)
    precipitation_sum: Optional[float] = pa.Field(nullable=True, ge=0)
    weathercode: Optional[float] = pa.Field(nullable=True, ge=0)

    class Config:
        coerce = True


# ---------------------------------------------------------------------------
# Silver schemas — full business invariants
# ---------------------------------------------------------------------------


class SilverSalesSchema(pa.DataFrameModel):
    """Full schema for ``silver_sales_long`` (long-format daily sales).

    Corresponds to ``contracts/silver_sales.yaml``.

    Grain: ``(date, store_id, item_id)``
    Primary key: ``(id, date)``
    """

    id: str = pa.Field(nullable=False)
    date: pl.Date = pa.Field(nullable=False)
    store_id: str = pa.Field(nullable=False)
    item_id: str = pa.Field(nullable=False)
    state_id: str = pa.Field(nullable=False, isin=["CA", "TX", "WI"])
    dept_id: str = pa.Field(nullable=False)
    cat_id: str = pa.Field(nullable=False, isin=["FOODS", "HOUSEHOLD", "HOBBIES"])
    sales: int = pa.Field(nullable=False, ge=0)
    is_zero_sale: bool = pa.Field(nullable=False)

    class Config:
        coerce = False


class SilverPricesDailySchema(pa.DataFrameModel):
    """Schema for ``silver_prices_daily.parquet``.

    Weekly sell-prices forward-filled to daily grain.
    """

    date: pl.Date = pa.Field(nullable=False)
    store_id: str = pa.Field(nullable=False)
    item_id: str = pa.Field(nullable=False)
    sell_price: float = pa.Field(nullable=True, gt=0)

    class Config:
        coerce = False


class SilverCalendarSchema(pa.DataFrameModel):
    """Schema for ``silver_calendar_enriched.parquet``."""

    date: pl.Date = pa.Field(nullable=False)
    wm_yr_wk: int = pa.Field(nullable=False)
    weekday: str = pa.Field(nullable=False)
    wday: int = pa.Field(nullable=False, ge=1, le=7)
    month: int = pa.Field(nullable=False, ge=1, le=12)
    year: int = pa.Field(nullable=False)
    d: str = pa.Field(nullable=False)
    is_weekend: bool = pa.Field(nullable=False)
    quarter: int = pa.Field(nullable=False, ge=1, le=4)

    class Config:
        coerce = False


class SilverWeatherSchema(pa.DataFrameModel):
    """Schema for ``silver_weather_daily.parquet``."""

    date: pl.Date = pa.Field(nullable=False)
    state: str = pa.Field(nullable=False, isin=["CA", "TX", "WI"])
    temperature_2m_max: Optional[float] = pa.Field(nullable=True)
    temperature_2m_min: Optional[float] = pa.Field(nullable=True)
    temperature_2m_mean: Optional[float] = pa.Field(nullable=True)
    precipitation_sum: Optional[float] = pa.Field(nullable=True, ge=0)
    weathercode: Optional[float] = pa.Field(nullable=True, ge=0)

    class Config:
        coerce = False


# ---------------------------------------------------------------------------
# Generic validation entry-point
# ---------------------------------------------------------------------------

_SCHEMA_REGISTRY: dict[str, type[pa.DataFrameModel]] = {
    # bronze
    "bronze_sales": BronzeSalesSchema,
    "bronze_calendar": BronzeCalendarSchema,
    "bronze_prices": BronzePricesSchema,
    "bronze_weather": BronzeWeatherSchema,
    # silver
    "silver_sales": SilverSalesSchema,
    "silver_prices_daily": SilverPricesDailySchema,
    "silver_calendar": SilverCalendarSchema,
    "silver_weather": SilverWeatherSchema,
}


def validate_dataframe(df: pl.DataFrame, contract_name: str) -> pl.DataFrame:
    """Validate a Polars DataFrame against a named contract schema.

    Parameters
    ----------
    df:
        DataFrame to validate.
    contract_name:
        Key into the internal schema registry (e.g. ``"silver_sales"``).

    Returns
    -------
    pl.DataFrame
        The validated DataFrame (passthrough if valid).

    Raises
    ------
    KeyError
        If ``contract_name`` is not registered.
    pandera.errors.SchemaError
        If the DataFrame fails validation.
    """
    schema_cls = _SCHEMA_REGISTRY.get(contract_name)
    if schema_cls is None:
        available = sorted(_SCHEMA_REGISTRY.keys())
        raise KeyError(
            f"Unknown contract {contract_name!r}. Available: {available}"
        )
    return schema_cls.validate(df)
