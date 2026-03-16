"""Frontend smoke tests (V3-Fase 3, spec §17).

Tests:
  1. serving assets exist in app/public/data/
  2. serving assets are valid JSON
  3. serving budget < 5 MB total
  4. asset_manifest.json lists all expected files
  5. (conditional on npm) npm run build succeeds
  6. (conditional on prior build) app/dist/index.html exists
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Project root (tests/e2e/ → ../.. = project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
APP_DIR = PROJECT_ROOT / "app"
PUBLIC_DATA_DIR = APP_DIR / "public" / "data"
DIST_DIR = APP_DIR / "dist"
BUDGET_BYTES = 5 * 1024 * 1024  # 5 MB

EXPECTED_ASSETS = [
    "asset_manifest.json",
    "executive_summary.json",
    "inventory_risk_matrix.json",
    "model_metrics.json",
]

# Resolve the npm executable: on Windows it may be npm.cmd or npm.CMD
_npm_path = shutil.which("npm") or shutil.which("npm.cmd") or shutil.which("npm.CMD")
NPM_AVAILABLE = _npm_path is not None
# Use shell=True on Windows to avoid FileNotFoundError with .cmd scripts
_NPM_SHELL = os.name == "nt"
_NPM_CMD = "npm" if _NPM_SHELL else (_npm_path or "npm")


class TestServingAssetsExist:
    def test_public_data_dir_exists(self):
        assert PUBLIC_DATA_DIR.exists(), f"Missing: {PUBLIC_DATA_DIR}"

    def test_asset_manifest_exists(self):
        assert (PUBLIC_DATA_DIR / "asset_manifest.json").exists()

    def test_executive_summary_exists(self):
        assert (PUBLIC_DATA_DIR / "executive_summary.json").exists()

    def test_inventory_risk_matrix_exists(self):
        assert (PUBLIC_DATA_DIR / "inventory_risk_matrix.json").exists()

    def test_model_metrics_exists(self):
        assert (PUBLIC_DATA_DIR / "model_metrics.json").exists()

    def test_forecast_series_files_exist(self):
        """At least one forecast_series_*.json file should exist."""
        forecast_files = list(PUBLIC_DATA_DIR.glob("forecast_series_*.json"))
        assert len(forecast_files) > 0, "No forecast_series_*.json files found"


class TestServingAssetsValidJson:
    def _load_json(self, path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_asset_manifest_valid_json(self):
        data = self._load_json(PUBLIC_DATA_DIR / "asset_manifest.json")
        assert isinstance(data, dict)

    def test_executive_summary_valid_json(self):
        data = self._load_json(PUBLIC_DATA_DIR / "executive_summary.json")
        assert isinstance(data, dict)

    def test_inventory_risk_matrix_valid_json(self):
        data = self._load_json(PUBLIC_DATA_DIR / "inventory_risk_matrix.json")
        assert isinstance(data, dict)

    def test_model_metrics_valid_json(self):
        data = self._load_json(PUBLIC_DATA_DIR / "model_metrics.json")
        assert isinstance(data, dict)

    def test_forecast_series_valid_json(self):
        for path in PUBLIC_DATA_DIR.glob("forecast_series_*.json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert isinstance(data, dict), f"Invalid JSON in {path.name}"


class TestServingBudget:
    def test_total_size_under_5mb(self):
        total = sum(p.stat().st_size for p in PUBLIC_DATA_DIR.iterdir() if p.is_file())
        assert total <= BUDGET_BYTES, (
            f"Serving assets total {total / 1024 / 1024:.2f} MB exceeds 5 MB budget"
        )

    def test_individual_files_reasonable(self):
        """No single file should exceed 3 MB."""
        for path in PUBLIC_DATA_DIR.iterdir():
            if path.is_file():
                size = path.stat().st_size
                assert size <= 3 * 1024 * 1024, f"{path.name} is {size / 1024 / 1024:.2f} MB (> 3 MB)"


class TestAssetManifestComplete:
    def test_manifest_lists_expected_files(self):
        manifest_path = PUBLIC_DATA_DIR / "asset_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        # Manifest should reference at least the core assets
        manifest_str = json.dumps(manifest)
        for asset in EXPECTED_ASSETS:
            assert asset in manifest_str or asset.replace(".json", "") in manifest_str, (
                f"Expected asset '{asset}' not referenced in manifest"
            )

    def test_manifest_has_generated_at(self):
        manifest_path = PUBLIC_DATA_DIR / "asset_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        # Check for any timestamp-like key
        manifest_str = json.dumps(manifest).lower()
        assert any(k in manifest_str for k in ["generated_at", "generated", "timestamp", "created"]), (
            "Manifest should contain a generation timestamp"
        )


@pytest.mark.skipif(not NPM_AVAILABLE, reason="npm not available in this environment")
class TestBuildSucceeds:
    def test_build_succeeds(self):
        """npm run build should exit 0."""
        result = subprocess.run(
            f"{_NPM_CMD} run build",
            cwd=APP_DIR,
            capture_output=True,
            text=True,
            timeout=120,
            shell=_NPM_SHELL,
        )
        assert result.returncode == 0, (
            f"npm run build failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_index_html_exists_after_build(self):
        """app/dist/index.html should exist after build."""
        # Run build first
        subprocess.run(
            f"{_NPM_CMD} run build",
            cwd=APP_DIR,
            capture_output=True,
            timeout=120,
            shell=_NPM_SHELL,
        )
        assert (DIST_DIR / "index.html").exists(), "app/dist/index.html not found after build"
