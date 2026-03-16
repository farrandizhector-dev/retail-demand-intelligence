"""Baseline forecasting models for demand benchmarking.

Implements 4 classical baselines via statsforecast 2.x:

- SeasonalNaive  : ŷ(t+h) = y(t+h-7)  (same day of week, prior week)
- MovingAverage28: ŷ = mean(last 28 days)
- Croston        : classic Croston for intermittent demand
- TSB            : Teunter-Syntetos-Babai for lumpy demand

All models return a DataFrame with columns:
    date (pl.Date), store_id (str), item_id (str),
    forecast_p50 (float), model_name (str)

Input DataFrame must contain: id, item_id, store_id, date (pl.Date), sales.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

import polars as pl

# statsforecast models
from statsforecast import StatsForecast
from statsforecast.models import CrostonClassic, SeasonalNaive, TSB, WindowAverage


ModelName = Literal["SeasonalNaive", "MovingAverage28", "Croston", "TSB"]

HORIZON: int = 28
SEASON_LENGTH: int = 7
MA_WINDOW: int = 28


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_sf_frame(df: pl.DataFrame, cutoff_date: date, id_col: str = "id") -> "pd.DataFrame":
    """Return pandas DataFrame in statsforecast format (unique_id, ds, y)."""
    filtered = df.filter(pl.col("date") <= cutoff_date)
    return (
        filtered
        .select([
            pl.col(id_col).alias("unique_id"),
            pl.col("date").alias("ds"),
            pl.col("sales").cast(pl.Float64).alias("y"),
        ])
        .to_pandas()
    )


def _sf_to_polars(
    forecast_df: "pd.DataFrame",
    sf_col: str,
    model_name: str,
    id_df: pl.DataFrame,
    id_col: str = "id",
) -> pl.DataFrame:
    """Convert statsforecast output (pandas) to the canonical output format."""
    result = pl.from_pandas(
        forecast_df[["unique_id", "ds", sf_col]].rename(
            columns={"unique_id": id_col, "ds": "date", sf_col: "forecast_p50"}
        )
    )
    result = result.with_columns([
        pl.col("date").cast(pl.Date),
        pl.col("forecast_p50").cast(pl.Float64).clip(0.0, None),
        pl.lit(model_name).alias("model_name"),
    ])

    # Attach item_id / store_id from original DataFrame
    id_map = id_df.select([id_col, "item_id", "store_id"]).unique(id_col)
    result = result.join(id_map, on=id_col, how="left")
    return result.select(["date", "store_id", "item_id", "forecast_p50", "model_name"])


def _run_sf(
    df: pl.DataFrame,
    cutoff_date: date,
    sf_model,
    sf_col: str,
    model_name: str,
    horizon: int,
    id_col: str = "id",
) -> pl.DataFrame:
    """Generic runner: fit a statsforecast model and return formatted output."""
    sf_df = _to_sf_frame(df, cutoff_date, id_col=id_col)
    sf = StatsForecast(models=[sf_model], freq="D", n_jobs=1)
    sf.fit(sf_df)
    forecast = sf.predict(h=horizon)
    return _sf_to_polars(forecast, sf_col, model_name, df, id_col=id_col)


# ---------------------------------------------------------------------------
# Public per-model functions
# ---------------------------------------------------------------------------


def forecast_seasonal_naive(
    df: pl.DataFrame,
    cutoff_date: date,
    horizon: int = HORIZON,
    id_col: str = "id",
) -> pl.DataFrame:
    """Seasonal Naive: ŷ(t+h) = y(t+h-7)."""
    return _run_sf(
        df,
        cutoff_date,
        sf_model=SeasonalNaive(season_length=SEASON_LENGTH),
        sf_col="SeasonalNaive",
        model_name="SeasonalNaive",
        horizon=horizon,
        id_col=id_col,
    )


def forecast_moving_average(
    df: pl.DataFrame,
    cutoff_date: date,
    horizon: int = HORIZON,
    window: int = MA_WINDOW,
    id_col: str = "id",
) -> pl.DataFrame:
    """Moving Average (window=28): ŷ = mean of last 28 days."""
    return _run_sf(
        df,
        cutoff_date,
        sf_model=WindowAverage(window_size=window),
        sf_col="WindowAverage",
        model_name="MovingAverage28",
        horizon=horizon,
        id_col=id_col,
    )


def forecast_croston(
    df: pl.DataFrame,
    cutoff_date: date,
    horizon: int = HORIZON,
    id_col: str = "id",
) -> pl.DataFrame:
    """Croston Classic: for intermittent demand series."""
    return _run_sf(
        df,
        cutoff_date,
        sf_model=CrostonClassic(),
        sf_col="CrostonClassic",
        model_name="Croston",
        horizon=horizon,
        id_col=id_col,
    )


def forecast_tsb(
    df: pl.DataFrame,
    cutoff_date: date,
    horizon: int = HORIZON,
    alpha_d: float = 0.2,
    alpha_p: float = 0.2,
    id_col: str = "id",
) -> pl.DataFrame:
    """TSB (Teunter-Syntetos-Babai): for lumpy demand series."""
    return _run_sf(
        df,
        cutoff_date,
        sf_model=TSB(alpha_d=alpha_d, alpha_p=alpha_p),
        sf_col="TSB",
        model_name="TSB",
        horizon=horizon,
        id_col=id_col,
    )


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------


def run_baselines(
    df: pl.DataFrame,
    cutoff_date: date,
    horizon: int = HORIZON,
    models: list[ModelName] | None = None,
    id_col: str = "id",
) -> pl.DataFrame:
    """Run all (or selected) baseline models and return stacked forecasts.

    Parameters
    ----------
    df:
        Long-format sales DataFrame with columns:
        id, item_id, store_id, date (pl.Date), sales.
    cutoff_date:
        Last date of training data. Forecast starts cutoff_date + 1 day.
    horizon:
        Forecast horizon in days (default 28).
    models:
        Subset of model names to run. Default: all 4 baselines.
    id_col:
        Column that uniquely identifies each time series.

    Returns
    -------
    pl.DataFrame
        Columns: date, store_id, item_id, forecast_p50, model_name.
        Stacked (one block of rows per model).
    """
    if models is None:
        models = ["SeasonalNaive", "MovingAverage28", "Croston", "TSB"]

    dispatch: dict[str, object] = {
        "SeasonalNaive": lambda: forecast_seasonal_naive(df, cutoff_date, horizon, id_col),
        "MovingAverage28": lambda: forecast_moving_average(df, cutoff_date, horizon, id_col=id_col),
        "Croston": lambda: forecast_croston(df, cutoff_date, horizon, id_col),
        "TSB": lambda: forecast_tsb(df, cutoff_date, horizon, id_col=id_col),
    }

    parts: list[pl.DataFrame] = []
    for name in models:
        if name not in dispatch:
            raise ValueError(f"Unknown model: {name!r}. Choose from {list(dispatch)}")
        parts.append(dispatch[name]())  # type: ignore[operator]

    return pl.concat(parts, how="vertical")
