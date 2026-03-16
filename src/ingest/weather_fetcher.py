"""Fetch historical daily weather data from the Open-Meteo Archive API.

Covers the 3 US states present in the M5 dataset (CA, TX, WI) for the full
M5 temporal range 2011-01-29 → 2016-06-19.

Data is saved as per-state CSV files under ``data/raw/weather/``.

API docs: https://open-meteo.com/en/docs/historical-weather-api
Rate limit: no API key required; ~10 000 requests / day.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARIABLES: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "weathercode",
]

# M5 temporal coverage
M5_START = date(2011, 1, 29)
M5_END = date(2016, 6, 19)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class WeatherLocation:
    """Geographic coordinates and metadata for a weather location."""

    lat: float
    lon: float
    reference: str = ""


# Default state-level locations matching the M5 spec
DEFAULT_LOCATIONS: dict[str, WeatherLocation] = {
    "CA": WeatherLocation(lat=34.05, lon=-118.24, reference="Los Angeles metro area"),
    "TX": WeatherLocation(lat=29.76, lon=-95.37, reference="Houston metro area"),
    "WI": WeatherLocation(lat=43.04, lon=-87.91, reference="Milwaukee metro area"),
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _build_request_params(
    location: WeatherLocation,
    start: date,
    end: date,
    variables: list[str],
) -> dict[str, Any]:
    """Build the query-parameter dict for the Open-Meteo archive endpoint."""
    return {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(variables),
        "timezone": "UTC",
    }


def _fetch_location(
    state: str,
    location: WeatherLocation,
    start: date,
    end: date,
    variables: list[str],
    client: httpx.Client,
    timeout: float = 60.0,
) -> list[dict[str, Any]]:
    """Call the Open-Meteo API for one location and return a list of daily records.

    Each record is a flat dict with keys: ``state``, ``date``, and one key per
    weather variable.

    Raises
    ------
    httpx.HTTPStatusError
        On non-2xx HTTP responses.
    """
    params = _build_request_params(location, start, end, variables)
    response = client.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    daily = payload.get("daily", {})

    dates: list[str] = daily.get("time", [])
    rows: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        row: dict[str, Any] = {"state": state, "date": d}
        for var in variables:
            values = daily.get(var, [])
            row[var] = values[i] if i < len(values) else None
        rows.append(row)

    return rows


def fetch_weather_data(
    output_dir: Path,
    *,
    locations: dict[str, WeatherLocation] | None = None,
    start: date = M5_START,
    end: date = M5_END,
    variables: list[str] | None = None,
    retry_delay: float = 5.0,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Path]:
    """Fetch historical weather from Open-Meteo and persist as CSV files.

    Idempotent: skips states whose output file already exists unless
    ``force=True``.

    Parameters
    ----------
    output_dir:
        Destination directory; created if needed.
    locations:
        Mapping ``{state_code: WeatherLocation}``; defaults to the 3 M5 states.
    start:
        Start date (inclusive). Defaults to the first M5 day.
    end:
        End date (inclusive). Defaults to the last M5 day.
    variables:
        List of Open-Meteo daily variable names. Defaults to ``WEATHER_VARIABLES``.
    retry_delay:
        Seconds to wait before a single automatic retry on request failure.
    force:
        Re-fetch even when the output CSV already exists.
    client:
        Optional pre-configured ``httpx.Client`` (useful for testing / mocking).

    Returns
    -------
    dict[str, Path]
        ``{state_code: path_to_csv}`` for every fetched location.
    """
    locations = locations or DEFAULT_LOCATIONS
    variables = variables or WEATHER_VARIABLES
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["state", "date"] + variables
    output_paths: dict[str, Path] = {}

    _owned_client = client is None
    if _owned_client:
        client = httpx.Client()

    try:
        for state, loc in locations.items():
            out_path = output_dir / f"weather_{state.lower()}.csv"
            output_paths[state] = out_path

            if out_path.exists() and not force:
                continue

            try:
                rows = _fetch_location(state, loc, start, end, variables, client)
            except (httpx.HTTPStatusError, httpx.RequestError):
                # Single automatic retry after a brief wait
                time.sleep(retry_delay)
                rows = _fetch_location(state, loc, start, end, variables, client)

            _write_csv(out_path, fieldnames, rows)

    finally:
        if _owned_client:
            client.close()

    return output_paths


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to a CSV file at ``path``."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_weather_csv(path: Path) -> list[dict[str, Any]]:
    """Read a weather CSV file into a list of row dicts.

    Parameters
    ----------
    path:
        Path to the CSV file produced by :func:`fetch_weather_data`.

    Returns
    -------
    list[dict[str, Any]]
        One dict per day with keys matching the CSV fieldnames.
    """
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)
