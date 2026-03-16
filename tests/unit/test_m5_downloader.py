"""Unit tests for src/ingest/m5_downloader.py."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from src.ingest.m5_downloader import (
    M5_EXPECTED_FILES,
    extract_m5_zip,
    sha256_file,
    verify_m5_files,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_zip(zip_path: Path, files: list[str] | None = None) -> Path:
    """Create a minimal zip containing stub CSVs for testing.

    Parameters
    ----------
    zip_path:
        Destination path for the zip file.
    files:
        List of filenames to include; defaults to ``M5_EXPECTED_FILES``.
    """
    files = files if files is not None else M5_EXPECTED_FILES
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname in files:
            # Minimal CSV content: just headers
            content = "id,value\n1,100\n"
            zf.writestr(fname, content)
    return zip_path


# ---------------------------------------------------------------------------
# extract_m5_zip
# ---------------------------------------------------------------------------


def test_extract_m5_zip_creates_all_expected_files(tmp_path: Path) -> None:
    zip_path = _make_fake_zip(tmp_path / "m5.zip")
    output_dir = tmp_path / "raw"

    result = extract_m5_zip(zip_path, output_dir)

    assert len(result) == len(M5_EXPECTED_FILES)
    for fname in M5_EXPECTED_FILES:
        assert (output_dir / fname).exists(), f"{fname} missing after extraction"


def test_extract_m5_zip_is_idempotent(tmp_path: Path) -> None:
    zip_path = _make_fake_zip(tmp_path / "m5.zip")
    output_dir = tmp_path / "raw"

    first = extract_m5_zip(zip_path, output_dir)
    second = extract_m5_zip(zip_path, output_dir)  # should not re-extract

    assert [str(p) for p in first] == [str(p) for p in second]


def test_extract_m5_zip_force_re_extracts(tmp_path: Path) -> None:
    zip_path = _make_fake_zip(tmp_path / "m5.zip")
    output_dir = tmp_path / "raw"

    extract_m5_zip(zip_path, output_dir)
    # Corrupt one file to confirm force re-extracts
    (output_dir / M5_EXPECTED_FILES[0]).write_bytes(b"corrupted")

    extract_m5_zip(zip_path, output_dir, force=True)
    content = (output_dir / M5_EXPECTED_FILES[0]).read_text()
    assert "id,value" in content  # original content restored


def test_extract_m5_zip_raises_if_zip_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="M5 zip not found"):
        extract_m5_zip(tmp_path / "nonexistent.zip", tmp_path / "out")


def test_extract_m5_zip_raises_if_file_missing_from_zip(tmp_path: Path) -> None:
    # Create zip with only 3 of the 5 expected files
    partial_zip = _make_fake_zip(tmp_path / "partial.zip", files=M5_EXPECTED_FILES[:3])
    with pytest.raises(FileNotFoundError, match="not found inside"):
        extract_m5_zip(partial_zip, tmp_path / "out")


def test_extract_m5_zip_handles_subdir_in_zip(tmp_path: Path) -> None:
    """Files inside a zip subdirectory should still extract to the flat output_dir."""
    zip_path = tmp_path / "m5_nested.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname in M5_EXPECTED_FILES:
            zf.writestr(f"m5-data/{fname}", "id,value\n1,1\n")

    output_dir = tmp_path / "out"
    result = extract_m5_zip(zip_path, output_dir)

    for p in result:
        assert p.parent == output_dir
        assert p.exists()


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_returns_64_char_hex(tmp_path: Path) -> None:
    p = tmp_path / "test.txt"
    p.write_text("hello world")
    digest = sha256_file(p)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "data.csv"
    p.write_bytes(b"col1,col2\n1,2\n")
    assert sha256_file(p) == sha256_file(p)


def test_sha256_file_differs_for_different_contents(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("content A")
    b.write_text("content B")
    assert sha256_file(a) != sha256_file(b)


# ---------------------------------------------------------------------------
# verify_m5_files
# ---------------------------------------------------------------------------


def test_verify_m5_files_returns_correct_keys(tmp_path: Path) -> None:
    for fname in M5_EXPECTED_FILES:
        (tmp_path / fname).write_text("id\n1\n")

    result = verify_m5_files(tmp_path)
    assert set(result.keys()) == set(M5_EXPECTED_FILES)
    for digest in result.values():
        assert len(digest) == 64


def test_verify_m5_files_raises_if_file_missing(tmp_path: Path) -> None:
    # Write only the first 4 files
    for fname in M5_EXPECTED_FILES[:-1]:
        (tmp_path / fname).write_text("id\n1\n")
    with pytest.raises(FileNotFoundError):
        verify_m5_files(tmp_path)
