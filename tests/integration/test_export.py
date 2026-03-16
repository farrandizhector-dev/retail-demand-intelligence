"""Integration tests for the serving assets export pipeline.

Verifies:
1. All required JSON files are generated.
2. Each file is valid JSON.
3. Each file is within its size budget.
4. Total serving directory is < 5 MB.
5. Asset manifest contains SHA-256 and correct sizes.
6. Executive summary has required keys.
7. Forecast series files are present for expected state×category combos.
8. Inventory risk matrix has the expected structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.export.serving_exporter import (
    _make_minimal_synthetic_sales,
    build_executive_summary,
    build_forecast_series,
    build_inventory_risk_matrix,
    export_serving_assets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_sales():
    """Small synthetic sales DataFrame for all export tests."""
    return _make_minimal_synthetic_sales(n_days=90, seed=0)


@pytest.fixture(scope="module")
def export_paths(tmp_path_factory, synthetic_sales):
    """Run the full export once and return the written paths dict."""
    output_dir = tmp_path_factory.mktemp("serving")
    return export_serving_assets(
        output_dir=output_dir,
        sales_df=synthetic_sales,
        is_synthetic=True,
    )


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


def test_executive_summary_exists(export_paths):
    assert export_paths["executive_summary"].exists()


def test_inventory_risk_matrix_exists(export_paths):
    assert export_paths["inventory_risk_matrix"].exists()


def test_model_metrics_exists(export_paths):
    assert export_paths["model_metrics"].exists()


def test_asset_manifest_exists(export_paths):
    assert export_paths["asset_manifest"].exists()


def test_forecast_series_files_exist(export_paths):
    """At least one forecast_series file should be present."""
    fc_files = [k for k in export_paths if k.startswith("forecast_series_")]
    assert len(fc_files) > 0


# ---------------------------------------------------------------------------
# Valid JSON
# ---------------------------------------------------------------------------


def test_all_files_are_valid_json(export_paths):
    for name, path in export_paths.items():
        with open(path, encoding="utf-8") as f:
            try:
                json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"{name} ({path.name}) is not valid JSON: {e}")


# ---------------------------------------------------------------------------
# Size budgets
# ---------------------------------------------------------------------------


def test_executive_summary_under_50kb(export_paths):
    size_kb = export_paths["executive_summary"].stat().st_size / 1024
    assert size_kb < 50, f"executive_summary is {size_kb:.1f} KB (limit: 50 KB)"


def test_inventory_risk_matrix_under_100kb(export_paths):
    size_kb = export_paths["inventory_risk_matrix"].stat().st_size / 1024
    assert size_kb < 100, f"inventory_risk_matrix is {size_kb:.1f} KB (limit: 100 KB)"


def test_model_metrics_under_100kb(export_paths):
    size_kb = export_paths["model_metrics"].stat().st_size / 1024
    assert size_kb < 100, f"model_metrics is {size_kb:.1f} KB (limit: 100 KB)"


def test_forecast_series_files_under_200kb(export_paths):
    for name, path in export_paths.items():
        if name.startswith("forecast_series_"):
            size_kb = path.stat().st_size / 1024
            assert size_kb < 200, (
                f"{path.name} is {size_kb:.1f} KB (limit: 200 KB)"
            )


def test_total_serving_dir_under_5mb(export_paths):
    """Total of all serving assets must be < 5 MB."""
    total_bytes = sum(p.stat().st_size for p in export_paths.values())
    total_mb = total_bytes / (1024 ** 2)
    assert total_mb < 5.0, f"Total serving assets: {total_mb:.2f} MB (limit: 5 MB)"


# ---------------------------------------------------------------------------
# Asset manifest
# ---------------------------------------------------------------------------


def test_manifest_has_assets_list(export_paths):
    with open(export_paths["asset_manifest"]) as f:
        manifest = json.load(f)
    assert "assets" in manifest
    assert isinstance(manifest["assets"], list)


def test_manifest_has_total_size(export_paths):
    with open(export_paths["asset_manifest"]) as f:
        manifest = json.load(f)
    assert "total_size_bytes" in manifest
    assert manifest["total_size_bytes"] > 0


def test_manifest_assets_have_sha256(export_paths):
    with open(export_paths["asset_manifest"]) as f:
        manifest = json.load(f)
    for asset in manifest["assets"]:
        assert "sha256" in asset, f"Asset {asset.get('name')} missing sha256"
        assert len(asset["sha256"]) == 64, f"SHA-256 not 64 chars: {asset['sha256']}"


def test_manifest_asset_sizes_match_disk(export_paths):
    """Sizes in manifest must match actual file sizes on disk."""
    with open(export_paths["asset_manifest"]) as f:
        manifest = json.load(f)
    for asset in manifest["assets"]:
        path = export_paths["asset_manifest"].parent / asset["name"]
        if path.exists():
            assert path.stat().st_size == asset["size_bytes"], (
                f"{asset['name']}: manifest says {asset['size_bytes']} bytes, "
                f"actual {path.stat().st_size}"
            )


# ---------------------------------------------------------------------------
# Executive summary content
# ---------------------------------------------------------------------------


def test_executive_summary_required_keys(export_paths):
    with open(export_paths["executive_summary"]) as f:
        summary = json.load(f)
    required = {
        "revenue_proxy_total",
        "revenue_by_state",
        "revenue_by_category",
        "fill_rate_avg",
        "stockout_rate",
        "inventory_value_total",
        "days_of_supply_avg",
        "monthly_trend",
        "n_skus",
    }
    assert required.issubset(set(summary.keys()))


def test_executive_summary_revenue_positive(export_paths):
    with open(export_paths["executive_summary"]) as f:
        summary = json.load(f)
    assert summary["revenue_proxy_total"] > 0


def test_executive_summary_monthly_trend_is_list(export_paths):
    with open(export_paths["executive_summary"]) as f:
        summary = json.load(f)
    assert isinstance(summary["monthly_trend"], list)
    if summary["monthly_trend"]:
        item = summary["monthly_trend"][0]
        assert "month" in item and "sales" in item


# ---------------------------------------------------------------------------
# Forecast series content
# ---------------------------------------------------------------------------


def test_forecast_series_has_history_and_forecast_keys(export_paths):
    fc_files = {k: v for k, v in export_paths.items() if k.startswith("forecast_series_")}
    for name, path in fc_files.items():
        with open(path) as f:
            data = json.load(f)
        assert "history" in data, f"{name} missing 'history'"
        assert "forecast" in data, f"{name} missing 'forecast'"
        assert "state_id" in data and "cat_id" in data


def test_forecast_series_state_category_coverage(export_paths, synthetic_sales):
    """A file should exist for each state×category in the sales data."""
    states = synthetic_sales["state_id"].unique().to_list()
    cats = synthetic_sales["cat_id"].unique().to_list()
    for state in states:
        for cat in cats:
            key = f"forecast_series_{state}_{cat}"
            assert key in export_paths, (
                f"Missing forecast series file for {state}×{cat}"
            )


# ---------------------------------------------------------------------------
# Inventory risk matrix
# ---------------------------------------------------------------------------


def test_risk_matrix_has_stores_key(export_paths):
    with open(export_paths["inventory_risk_matrix"]) as f:
        data = json.load(f)
    assert "stores" in data


# ---------------------------------------------------------------------------
# build_executive_summary (unit)
# ---------------------------------------------------------------------------


def test_build_executive_summary_fills_rate_zero_without_inventory():
    sales = _make_minimal_synthetic_sales(n_days=30, seed=1)
    summary = build_executive_summary(sales)
    assert summary["fill_rate_avg"] == 0.0


def test_build_executive_summary_n_skus_positive():
    sales = _make_minimal_synthetic_sales(n_days=30, seed=1)
    summary = build_executive_summary(sales)
    assert summary["n_skus"] > 0


# ---------------------------------------------------------------------------
# build_forecast_series (unit)
# ---------------------------------------------------------------------------


def test_build_forecast_series_returns_dict():
    sales = _make_minimal_synthetic_sales(n_days=30, seed=2)
    result = build_forecast_series(sales)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_build_forecast_series_history_non_empty():
    sales = _make_minimal_synthetic_sales(n_days=30, seed=2)
    result = build_forecast_series(sales)
    for key, payload in result.items():
        assert len(payload["history"]) > 0, f"Empty history for {key}"


# ---------------------------------------------------------------------------
# build_inventory_risk_matrix (unit)
# ---------------------------------------------------------------------------


def test_build_risk_matrix_empty_df():
    import polars as pl
    result = build_inventory_risk_matrix(pl.DataFrame())
    assert result["stores"] == []


def test_build_risk_matrix_with_data():
    import polars as pl
    inv_df = pl.DataFrame({
        "id": ["A_CA_1", "B_CA_1"],
        "store_id": ["CA_1", "CA_1"],
        "dept_id": ["FOODS_1", "FOODS_1"],
        "fill_rate": [0.95, 0.85],
        "stockout_days": [2, 8],
        "days_of_supply": [30.0, 10.0],
    })
    result = build_inventory_risk_matrix(inv_df)
    assert len(result["stores"]) == 1  # one store
    store = result["stores"][0]
    assert store["store_id"] == "CA_1"
    assert "FOODS_1" in store["departments"]
