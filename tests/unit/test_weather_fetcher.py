"""Unit tests for src/ingest/weather_fetcher.py.

All HTTP calls are intercepted via a custom httpx Transport so no real
network requests are made during testing.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.ingest.weather_fetcher import (
    DEFAULT_LOCATIONS,
    M5_END,
    M5_START,
    WEATHER_VARIABLES,
    WeatherLocation,
    _build_request_params,
    _fetch_location,
    fetch_weather_data,
    load_weather_csv,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_daily_payload(
    start: date,
    end: date,
    variables: list[str],
) -> dict[str, Any]:
    """Build a minimal Open-Meteo-style JSON payload for a date range."""
    days = (end - start).days + 1
    daily: dict[str, Any] = {
        "time": [
            (date.fromordinal(start.toordinal() + i)).isoformat() for i in range(days)
        ]
    }
    for var in variables:
        daily[var] = [float(i) for i in range(days)]
    return {"latitude": 34.05, "longitude": -118.24, "daily": daily}


class _FakeTransport(httpx.BaseTransport):
    """Return a canned Open-Meteo JSON response for any request."""

    def __init__(self, start: date, end: date, variables: list[str]) -> None:
        self.start = start
        self.end = end
        self.variables = variables
        self.call_count = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        payload = _make_daily_payload(self.start, self.end, self.variables)
        return httpx.Response(200, json=payload)


@pytest.fixture
def small_date_range() -> tuple[date, date]:
    return date(2011, 1, 29), date(2011, 2, 4)  # 7 days


@pytest.fixture
def fake_transport(small_date_range: tuple[date, date]) -> _FakeTransport:
    start, end = small_date_range
    return _FakeTransport(start, end, WEATHER_VARIABLES)


# ---------------------------------------------------------------------------
# _build_request_params
# ---------------------------------------------------------------------------


def test_build_request_params_keys() -> None:
    loc = WeatherLocation(lat=34.05, lon=-118.24, reference="LA")
    params = _build_request_params(loc, date(2011, 1, 29), date(2011, 2, 4), WEATHER_VARIABLES)
    assert params["latitude"] == 34.05
    assert params["longitude"] == -118.24
    assert params["start_date"] == "2011-01-29"
    assert params["end_date"] == "2011-02-04"
    assert params["timezone"] == "UTC"
    assert all(v in params["daily"] for v in WEATHER_VARIABLES)


# ---------------------------------------------------------------------------
# _fetch_location
# ---------------------------------------------------------------------------


def test_fetch_location_returns_correct_row_count(
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)
    rows = _fetch_location("CA", DEFAULT_LOCATIONS["CA"], start, end, WEATHER_VARIABLES, client)
    expected_days = (end - start).days + 1
    assert len(rows) == expected_days


def test_fetch_location_row_has_state_and_date(
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)
    rows = _fetch_location("TX", DEFAULT_LOCATIONS["TX"], start, end, WEATHER_VARIABLES, client)
    assert rows[0]["state"] == "TX"
    assert rows[0]["date"] == start.isoformat()


def test_fetch_location_row_has_all_variables(
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)
    rows = _fetch_location("WI", DEFAULT_LOCATIONS["WI"], start, end, WEATHER_VARIABLES, client)
    for var in WEATHER_VARIABLES:
        assert var in rows[0], f"Variable {var!r} missing from row"


# ---------------------------------------------------------------------------
# fetch_weather_data
# ---------------------------------------------------------------------------


def test_fetch_weather_data_creates_one_csv_per_state(
    tmp_path: Path,
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)

    result = fetch_weather_data(
        tmp_path,
        start=start,
        end=end,
        client=client,
    )

    assert set(result.keys()) == {"CA", "TX", "WI"}
    for path in result.values():
        assert path.exists(), f"{path} not created"


def test_fetch_weather_data_csv_has_correct_rows(
    tmp_path: Path,
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)

    result = fetch_weather_data(tmp_path, start=start, end=end, client=client)
    rows = load_weather_csv(result["CA"])
    expected_days = (end - start).days + 1
    assert len(rows) == expected_days


def test_fetch_weather_data_is_idempotent(
    tmp_path: Path,
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)

    fetch_weather_data(tmp_path, start=start, end=end, client=client)
    calls_after_first = fake_transport.call_count

    fetch_weather_data(tmp_path, start=start, end=end, client=client)
    # No additional HTTP calls should be made
    assert fake_transport.call_count == calls_after_first


def test_fetch_weather_data_force_refetches(
    tmp_path: Path,
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)

    fetch_weather_data(tmp_path, start=start, end=end, client=client)
    calls_before = fake_transport.call_count

    fetch_weather_data(tmp_path, start=start, end=end, client=client, force=True)
    assert fake_transport.call_count > calls_before


def test_fetch_weather_data_custom_locations(
    tmp_path: Path,
    fake_transport: _FakeTransport,
    small_date_range: tuple[date, date],
) -> None:
    start, end = small_date_range
    client = httpx.Client(transport=fake_transport)
    custom = {"NY": WeatherLocation(lat=40.71, lon=-74.01, reference="NYC")}

    result = fetch_weather_data(
        tmp_path, locations=custom, start=start, end=end, client=client
    )
    assert "NY" in result
    assert result["NY"].exists()


# ---------------------------------------------------------------------------
# load_weather_csv
# ---------------------------------------------------------------------------


def test_load_weather_csv_returns_list_of_dicts(tmp_path: Path) -> None:
    p = tmp_path / "weather_ca.csv"
    fieldnames = ["state", "date", "temperature_2m_max"]
    rows = [{"state": "CA", "date": "2011-01-29", "temperature_2m_max": "20.5"}]
    with open(p, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_weather_csv(p)
    assert len(loaded) == 1
    assert loaded[0]["state"] == "CA"
    assert loaded[0]["date"] == "2011-01-29"
