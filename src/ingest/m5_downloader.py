"""Extract M5 competition CSVs from a local zip archive into the raw layer.

The M5 zip is expected at ``data/raw/m5/m5-forecasting-accuracy.zip``.
No Kaggle API interaction is required — the zip must already be present locally.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

# All five CSVs shipped inside the M5 competition zip
M5_EXPECTED_FILES: list[str] = [
    "sales_train_validation.csv",
    "sales_train_evaluation.csv",
    "calendar.csv",
    "sell_prices.csv",
    "sample_submission.csv",
]


def extract_m5_zip(
    zip_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
) -> list[Path]:
    """Extract M5 CSVs from ``zip_path`` into ``output_dir``.

    Idempotent: skips extraction if all expected files already exist,
    unless ``force=True``.

    Parameters
    ----------
    zip_path:
        Path to ``m5-forecasting-accuracy.zip``.
    output_dir:
        Destination directory; created if it does not exist.
    force:
        Re-extract even when output files are already present.

    Returns
    -------
    list[Path]
        Paths of the extracted CSV files, in the canonical order defined by
        ``M5_EXPECTED_FILES``.

    Raises
    ------
    FileNotFoundError
        If ``zip_path`` does not exist or an expected CSV is absent from
        the zip.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"M5 zip not found: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    already_extracted = all((output_dir / f).exists() for f in M5_EXPECTED_FILES)
    if already_extracted and not force:
        return [output_dir / f for f in M5_EXPECTED_FILES]

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for fname in M5_EXPECTED_FILES:
            # Files may be at the zip root or inside a subdirectory
            match = next((m for m in members if m.endswith(fname)), None)
            if match is None:
                raise FileNotFoundError(
                    f"Expected file {fname!r} not found inside {zip_path}"
                )
            with zf.open(match) as src, open(output_dir / fname, "wb") as dst:
                dst.write(src.read())

    return [output_dir / f for f in M5_EXPECTED_FILES]


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Parameters
    ----------
    path:
        Path to the file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_m5_files(output_dir: Path) -> dict[str, str]:
    """Return a mapping of filename → SHA-256 for all M5 CSVs in ``output_dir``.

    Parameters
    ----------
    output_dir:
        Directory where M5 CSVs were extracted.

    Returns
    -------
    dict[str, str]
        ``{filename: sha256_hex}`` for every expected M5 CSV file.

    Raises
    ------
    FileNotFoundError
        If any expected CSV is missing.
    """
    checksums: dict[str, str] = {}
    for fname in M5_EXPECTED_FILES:
        p = output_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"M5 file not found: {p}")
        checksums[fname] = sha256_file(p)
    return checksums
