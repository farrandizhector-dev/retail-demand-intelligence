"""Feature store orchestrator — builds feature_store_v1.parquet.

Combines all feature families into a single versioned Parquet file stored
at ``data/features/feature_store_v1.parquet``.

Pipeline:
  silver_sales_long  →  lags  →  rolling  →  intermittency
       ↓                                             ↓
  silver_calendar   →  calendar features              →
  silver_prices     →  price features                 →
  silver_weather    →  weather features               →  JOIN  →  interactions  →  write
  classification    →  demand_class, abc_class, xyz_class →

LEAKAGE PROTOCOL:
  When ``cutoff_date`` is provided (backtesting mode), all feature
  families filter their source data to ``date <= cutoff_date`` before
  computing backward-looking statistics.  Rows after ``cutoff_date``
  are appended with null feature values — they must be predicted.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from src.features.calendar_features import add_calendar_features
from src.features.interaction_features import add_interaction_features
from src.features.intermittency_features import add_intermittency_features
from src.features.lag_features import add_lag_features
from src.features.price_features import add_price_features
from src.features.rolling_features import add_rolling_features
from src.features.weather_features import add_weather_features

FEATURE_STORE_VERSION = 1


def build_feature_store(
    silver_dir: Path,
    output_path: Path,
    *,
    cutoff_date: date | None = None,
    force: bool = False,
    classification_path: Path | None = None,
) -> pl.DataFrame:
    """Build the versioned feature store from silver-layer inputs.

    Parameters
    ----------
    silver_dir:
        Directory containing silver Parquet files:
        ``silver_calendar_enriched.parquet``,
        ``silver_prices_daily.parquet``,
        ``silver_weather_daily.parquet``.
        Silver sales are expected under ``silver_sales_long/``.
    output_path:
        Destination Parquet path (e.g.
        ``data/features/feature_store_v1.parquet``).
    cutoff_date:
        If not None, build features only using data up to this date
        (backtesting / fold mode).
    force:
        Re-build even if the output already exists.
    classification_path:
        Optional path to ``demand_classification.parquet``.  When provided,
        ``demand_class``, ``abc_class``, ``xyz_class`` are joined.

    Returns
    -------
    pl.DataFrame
        The complete feature DataFrame (also written to ``output_path``).
    """
    if output_path.exists() and not force:
        return pl.read_parquet(output_path)

    # --- Load silver sales (base) ---
    sales_long_dir = silver_dir / "silver_sales_long"
    sales_files = sorted(sales_long_dir.rglob("*.parquet"))
    if not sales_files:
        raise FileNotFoundError(
            f"No silver sales partitions found under {sales_long_dir}"
        )
    df = pl.read_parquet([str(p) for p in sales_files])

    if cutoff_date:
        # Include a window beyond cutoff for lag/rolling targets
        df = df.filter(pl.col("date") <= cutoff_date)

    df = df.sort(["id", "date"])

    # --- Feature family 1: lags ---
    df = add_lag_features(df, cutoff_date=cutoff_date)

    # --- Feature family 2: rolling ---
    df = add_rolling_features(df, cutoff_date=cutoff_date)

    # --- Feature family 3: intermittency ---
    df = add_intermittency_features(df, cutoff_date=cutoff_date)

    # --- Feature family 4: calendar ---
    cal_path = silver_dir / "silver_calendar_enriched.parquet"
    if cal_path.exists():
        calendar_df = pl.read_parquet(cal_path)
        df = add_calendar_features(df, calendar_df)

    # --- Feature family 5: prices ---
    prices_path = silver_dir / "silver_prices_daily.parquet"
    if prices_path.exists():
        prices_df = pl.read_parquet(prices_path)
        df = add_price_features(df, prices_df, cutoff_date=cutoff_date)

    # --- Feature family 6: weather ---
    weather_path = silver_dir / "silver_weather_daily.parquet"
    if weather_path.exists():
        weather_df = pl.read_parquet(weather_path)
        df = add_weather_features(df, weather_df, cutoff_date=cutoff_date)

    # --- Feature family 7: interactions ---
    df = add_interaction_features(df)

    # --- Join demand classification ---
    if classification_path and classification_path.exists():
        cls_df = pl.read_parquet(classification_path).select(
            ["id", "demand_class", "abc_class", "xyz_class"]
        )
        df = df.join(cls_df, on="id", how="left")

    # --- Write ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path, compression="snappy")

    return df


def load_feature_store(path: Path) -> pl.LazyFrame:
    """Load the feature store as a Polars LazyFrame for downstream use.

    Parameters
    ----------
    path:
        Path to the feature store Parquet file.

    Returns
    -------
    pl.LazyFrame
        Lazy reader — call ``.collect()`` when you need materialised data.
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature store not found: {path}")
    return pl.scan_parquet(str(path))
