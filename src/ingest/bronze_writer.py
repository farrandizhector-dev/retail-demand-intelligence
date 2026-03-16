"""Convert raw M5 CSVs (and weather CSVs) to Parquet in the bronze layer.

Each Parquet file is accompanied by a SHA-256 checksum stored in a
``data/bronze/checksums.json`` manifest so downstream stages can verify
lineage integrity.

Polars is used as the primary processing engine (ADR-001).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKSUM_FILENAME = "checksums.json"

# M5 CSV files → bronze Parquet names
M5_CSV_TO_BRONZE: dict[str, str] = {
    "sales_train_validation.csv": "bronze_sales.parquet",
    "calendar.csv": "bronze_calendar.parquet",
    "sell_prices.csv": "bronze_prices.parquet",
}

# Weather per-state CSVs → single bronze Parquet
WEATHER_BRONZE_NAME = "bronze_weather.parquet"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def sha256_parquet(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Parameters
    ----------
    path:
        File to hash.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_checksums(manifest: Path) -> dict[str, str]:
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    return {}


def _save_checksums(manifest: Path, checksums: dict[str, str]) -> None:
    manifest.write_text(
        json.dumps(checksums, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Individual conversion helpers
# ---------------------------------------------------------------------------


def _csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    *,
    infer_schema_length: int = 10_000,
) -> str:
    """Read a CSV with Polars and write it as Parquet.

    Parameters
    ----------
    csv_path:
        Input CSV file.
    parquet_path:
        Output Parquet file (created or overwritten).
    infer_schema_length:
        Number of rows to sample for schema inference.

    Returns
    -------
    str
        SHA-256 digest of the written Parquet file.
    """
    df = pl.read_csv(csv_path, infer_schema_length=infer_schema_length)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path, compression="snappy")
    return sha256_parquet(parquet_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_m5_bronze(
    raw_m5_dir: Path,
    bronze_dir: Path,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Convert M5 raw CSVs to bronze Parquet files.

    Converts three M5 files:
    - ``sales_train_validation.csv`` → ``bronze_sales.parquet``
    - ``calendar.csv``              → ``bronze_calendar.parquet``
    - ``sell_prices.csv``           → ``bronze_prices.parquet``

    Idempotent: skips files already present in ``bronze_dir`` unless
    ``force=True``.

    Parameters
    ----------
    raw_m5_dir:
        Directory containing the extracted M5 CSVs.
    bronze_dir:
        Output directory for bronze Parquet files and the checksum manifest.
    force:
        Re-convert even if the Parquet file already exists.

    Returns
    -------
    dict[str, str]
        Updated ``{parquet_filename: sha256}`` mapping (all bronze files in manifest).
    """
    bronze_dir.mkdir(parents=True, exist_ok=True)
    manifest = bronze_dir / CHECKSUM_FILENAME
    checksums = _load_checksums(manifest)

    for csv_name, parquet_name in M5_CSV_TO_BRONZE.items():
        csv_path = raw_m5_dir / csv_name
        parquet_path = bronze_dir / parquet_name

        if not csv_path.exists():
            raise FileNotFoundError(f"M5 CSV not found: {csv_path}")

        if parquet_path.exists() and not force:
            if parquet_name not in checksums:
                checksums[parquet_name] = sha256_parquet(parquet_path)
            continue

        checksums[parquet_name] = _csv_to_parquet(csv_path, parquet_path)

    _save_checksums(manifest, checksums)
    return checksums


def write_weather_bronze(
    raw_weather_dir: Path,
    bronze_dir: Path,
    *,
    force: bool = False,
) -> dict[str, str]:
    """Concatenate per-state weather CSVs into a single bronze Parquet.

    Reads all ``weather_*.csv`` files from ``raw_weather_dir`` and writes
    ``bronze_weather.parquet`` to ``bronze_dir``.

    Parameters
    ----------
    raw_weather_dir:
        Directory containing ``weather_ca.csv``, ``weather_tx.csv``, etc.
    bronze_dir:
        Output directory for the bronze Parquet file.
    force:
        Re-convert even if ``bronze_weather.parquet`` already exists.

    Returns
    -------
    dict[str, str]
        Updated checksum manifest (just ``bronze_weather.parquet`` entry).
    """
    bronze_dir.mkdir(parents=True, exist_ok=True)
    manifest = bronze_dir / CHECKSUM_FILENAME
    checksums = _load_checksums(manifest)

    parquet_path = bronze_dir / WEATHER_BRONZE_NAME

    if parquet_path.exists() and not force:
        if WEATHER_BRONZE_NAME not in checksums:
            checksums[WEATHER_BRONZE_NAME] = sha256_parquet(parquet_path)
        _save_checksums(manifest, checksums)
        return {WEATHER_BRONZE_NAME: checksums[WEATHER_BRONZE_NAME]}

    weather_csvs = sorted(raw_weather_dir.glob("weather_*.csv"))
    if not weather_csvs:
        raise FileNotFoundError(
            f"No weather CSVs found in {raw_weather_dir}. "
            "Run weather_fetcher.fetch_weather_data() first."
        )

    frames = [pl.read_csv(p) for p in weather_csvs]
    df = pl.concat(frames, how="vertical")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(parquet_path, compression="snappy")

    checksums[WEATHER_BRONZE_NAME] = sha256_parquet(parquet_path)
    _save_checksums(manifest, checksums)
    return {WEATHER_BRONZE_NAME: checksums[WEATHER_BRONZE_NAME]}


def load_bronze(bronze_dir: Path, name: str) -> pl.DataFrame:
    """Load a named bronze Parquet file.

    Parameters
    ----------
    bronze_dir:
        Directory containing bronze Parquet files.
    name:
        File name, e.g. ``"bronze_sales.parquet"``.

    Returns
    -------
    pl.DataFrame
        Loaded Parquet as a Polars DataFrame.

    Raises
    ------
    FileNotFoundError
        If the requested Parquet is absent from ``bronze_dir``.
    """
    path = bronze_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Bronze file not found: {path}")
    return pl.read_parquet(path)


def get_bronze_checksums(bronze_dir: Path) -> dict[str, str]:
    """Read the current checksum manifest from ``bronze_dir``.

    Returns an empty dict if the manifest does not yet exist.
    """
    manifest = bronze_dir / CHECKSUM_FILENAME
    return _load_checksums(manifest)
