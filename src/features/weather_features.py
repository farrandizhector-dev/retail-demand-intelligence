"""Weather features for demand forecasting.

Joins the silver weather table (daily, state-level) with the sales
DataFrame and computes a temperature anomaly feature.

LEAKAGE PROTOCOL (rule 3 — section 8.2):
  Weather features use only data up to ``cutoff_date``.  The join is on
  ``(date, state_id)``, so each row receives the weather observation for
  its date and state — no future data is used.
"""

from __future__ import annotations

from datetime import date

import polars as pl


# Weather variable names as they appear in bronze/silver weather
WEATHER_VARS: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "weathercode",
]

# Renamed feature names in the feature store
WEATHER_FEATURE_NAMES: dict[str, str] = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "precipitation_sum",
    "weathercode": "weathercode",
}


def add_weather_features(
    df: pl.DataFrame,
    weather_df: pl.DataFrame,
    *,
    date_col: str = "date",
    state_col: str = "state_id",
    id_col: str = "id",
    cutoff_date: date | None = None,
    anomaly_window: int = 30,
) -> pl.DataFrame:
    """Join weather features and derive temperature anomaly.

    Features added:
    - ``temp_max``, ``temp_min``, ``temp_mean``: daily temperatures.
    - ``precipitation_sum``: daily precipitation (mm).
    - ``weathercode``: WMO weather interpretation code.
    - ``temp_anomaly_vs_30d_avg``: deviation of ``temp_mean`` from its
        30-day backward rolling average (per state, per day).

    Parameters
    ----------
    df:
        Long-format sales DataFrame sorted by ``(id, date)``.
    weather_df:
        Silver weather DataFrame with columns
        ``(date, state, temperature_2m_max, temperature_2m_min,
           temperature_2m_mean, precipitation_sum, weathercode)``.
    date_col:
        Name of the date column in ``df``.
    state_col:
        Column in ``df`` holding the state identifier (``CA``, ``TX``, ``WI``).
    id_col:
        Series identifier (used for sorting).
    cutoff_date:
        Only weather data on or before this date is used.
    anomaly_window:
        Rolling window (days) for temperature anomaly baseline.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with weather feature columns appended.
    """
    if cutoff_date:
        weather_df = weather_df.filter(pl.col(date_col) <= cutoff_date)

    # Rename columns + align state column name
    weather_col = "state" if "state" in weather_df.columns else state_col
    weather_slim = weather_df.rename(
        {weather_col: state_col} if weather_col != state_col else {}
    )

    # Keep only needed columns and rename to feature names
    rename_map = {k: v for k, v in WEATHER_FEATURE_NAMES.items() if k in weather_slim.columns}
    weather_slim = weather_slim.select(
        [date_col, state_col] + list(rename_map.keys())
    ).rename(rename_map)

    # Compute temperature anomaly per state
    weather_slim = weather_slim.sort([state_col, date_col])
    if "temp_mean" in weather_slim.columns:
        weather_slim = weather_slim.with_columns(
            (
                pl.col("temp_mean")
                - pl.col("temp_mean")
                .rolling_mean(window_size=anomaly_window, min_samples=1)
                .over(state_col)
            ).alias("temp_anomaly_vs_30d_avg")
        )

    # Join onto sales DataFrame on (date, state_id)
    df = df.join(weather_slim, on=[date_col, state_col], how="left")

    return df
