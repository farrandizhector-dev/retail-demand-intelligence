"""Export pipeline: GOLD layer → compact JSON serving assets for the frontend.

Generates the following files under ``output_dir`` (default: data/gold/serving/):

  executive_summary.json        < 50 KB  — KPIs, trends, top-line metrics
  forecast_series_{st}_{cat}.json < 200 KB each  — state×category aggregated
  inventory_risk_matrix.json    < 100 KB — store×department risk table
  model_metrics.json            < 100 KB — metrics by fold/model/segment
  asset_manifest.json           < 5 KB   — name, size, sha256 of each asset

Total bundle target: < 5 MB (spec section 13.4).

All functions accept DataFrames directly and return Path objects so that tests
can work with in-memory synthetic data without touching disk beyond a tmp_path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

SYNTHETIC_TAG = "SYNTHETIC"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(data: Any, path: Path) -> Path:
    """Write Python object as UTF-8 JSON; return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), default=str)
    return path


def _size_kb(path: Path) -> float:
    return path.stat().st_size / 1024


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------


def build_executive_summary(
    sales_df: pl.DataFrame,
    inventory_df: pl.DataFrame | None = None,
    metrics: dict | None = None,
    n_trend_months: int = 12,
    is_synthetic: bool = False,
) -> dict:
    """Build the executive_summary payload.

    Parameters
    ----------
    sales_df:
        Long-format silver sales (id, item_id, cat_id, store_id, state_id,
        dept_id, date, sales).
    inventory_df:
        Simulation output (id, fill_rate, stockout_days, avg_inventory, …).
    metrics:
        Optional dict of aggregate model metrics (e.g. from backtesting).
    n_trend_months:
        Number of trailing months to include in the monthly trend.
    is_synthetic:
        Tag summary as synthetic if True.

    Returns
    -------
    dict — the executive_summary payload.
    """
    # Revenue proxy (units sold — no price for V1)
    total_revenue = float(sales_df["sales"].sum())

    # By state
    rev_by_state = (
        sales_df.group_by("state_id").agg(pl.col("sales").sum().alias("rev"))
        .sort("state_id")
    )
    revenue_by_state = {
        row["state_id"]: int(row["rev"])
        for row in rev_by_state.iter_rows(named=True)
    }

    # By category
    rev_by_cat = (
        sales_df.group_by("cat_id").agg(pl.col("sales").sum().alias("rev"))
        .sort("cat_id")
    )
    revenue_by_category = {
        row["cat_id"]: int(row["rev"])
        for row in rev_by_cat.iter_rows(named=True)
    }

    # Monthly trend (last n_trend_months)
    monthly = (
        sales_df
        .with_columns(
            pl.col("date").dt.truncate("1mo").alias("month")
        )
        .group_by("month")
        .agg(pl.col("sales").sum().alias("monthly_sales"))
        .sort("month")
        .tail(n_trend_months)
    )
    monthly_trend = [
        {"month": str(row["month"]), "sales": int(row["monthly_sales"])}
        for row in monthly.iter_rows(named=True)
    ]

    # Inventory KPIs
    fill_rate_avg = 0.0
    stockout_rate = 0.0
    inventory_value_total = 0.0
    days_of_supply_avg = 0.0

    if inventory_df is not None and not inventory_df.is_empty():
        fill_rate_avg = float(inventory_df["fill_rate"].mean())
        stockout_rate = float(
            (inventory_df["stockout_days"] > 0).sum() / len(inventory_df)
        )
        days_of_supply_avg = float(inventory_df["days_of_supply"].mean())
        if "avg_inventory" in inventory_df.columns:
            inventory_value_total = float(inventory_df["avg_inventory"].sum())

    # Forecast accuracy
    forecast_mae_avg = float(metrics.get("mean_mae", 0.0)) if metrics else 0.0

    return {
        "data_label": SYNTHETIC_TAG if is_synthetic else "M5_REAL",
        "revenue_proxy_total": int(total_revenue),
        "revenue_by_state": revenue_by_state,
        "revenue_by_category": revenue_by_category,
        "fill_rate_avg": round(fill_rate_avg, 4),
        "stockout_rate": round(stockout_rate, 4),
        "forecast_mae_avg": round(forecast_mae_avg, 4),
        "inventory_value_total": round(inventory_value_total, 2),
        "days_of_supply_avg": round(days_of_supply_avg, 2),
        "monthly_trend": monthly_trend,
        "n_skus": int(sales_df["id"].n_unique()),
        "n_stores": int(sales_df["store_id"].n_unique()),
    }


# ---------------------------------------------------------------------------
# Forecast series (state × category aggregated)
# ---------------------------------------------------------------------------


def build_forecast_series(
    sales_df: pl.DataFrame,
    forecast_df: pl.DataFrame | None = None,
    n_history_days: int = 180,
) -> dict[tuple[str, str], dict]:
    """Build one payload per (state, category) combination.

    Parameters
    ----------
    sales_df:
        Long-format sales with date, state_id, cat_id, sales.
    forecast_df:
        Optional DataFrame with date, state_id, cat_id,
        forecast_p10, forecast_p50, forecast_p90 (already aggregated
        at state×category level, or will be aggregated here if at id level).
    n_history_days:
        Number of trailing days of actual sales to include.

    Returns
    -------
    dict mapping (state_id, cat_id) → payload dict.
    """
    payloads: dict[tuple[str, str], dict] = {}

    # Aggregate actual sales to state×category×date
    actual_agg = (
        sales_df
        .group_by(["state_id", "cat_id", "date"])
        .agg(pl.col("sales").sum().alias("actual"))
        .sort(["state_id", "cat_id", "date"])
    )

    # Collect unique (state, cat) combinations
    combos = (
        sales_df.select(["state_id", "cat_id"])
        .unique()
        .sort(["state_id", "cat_id"])
    )

    for row in combos.iter_rows(named=True):
        state = row["state_id"]
        cat = row["cat_id"]

        # Historical actuals (last n_history_days)
        hist = (
            actual_agg
            .filter(
                (pl.col("state_id") == state) & (pl.col("cat_id") == cat)
            )
            .tail(n_history_days)
        )

        history_records = [
            {"date": str(r["date"]), "actual": int(r["actual"])}
            for r in hist.iter_rows(named=True)
        ]

        # Forecast records
        forecast_records: list[dict] = []
        if forecast_df is not None and not forecast_df.is_empty():
            # Aggregate forecast to state×cat level if needed
            agg_cols = [c for c in ["state_id", "cat_id"] if c in forecast_df.columns]
            if agg_cols and "forecast_p50" in forecast_df.columns:
                fc_agg = (
                    forecast_df
                    .filter(
                        (pl.col("state_id") == state) & (pl.col("cat_id") == cat)
                    )
                    .group_by("date")
                    .agg([
                        pl.col("forecast_p10").sum(),
                        pl.col("forecast_p50").sum(),
                        pl.col("forecast_p90").sum(),
                    ])
                    .sort("date")
                    if all(c in forecast_df.columns for c in ["state_id", "cat_id"])
                    else pl.DataFrame()
                )
                for r in fc_agg.iter_rows(named=True):
                    forecast_records.append({
                        "date": str(r["date"]),
                        "forecast_p10": round(float(r["forecast_p10"]), 2),
                        "forecast_p50": round(float(r["forecast_p50"]), 2),
                        "forecast_p90": round(float(r["forecast_p90"]), 2),
                    })

        payloads[(state, cat)] = {
            "state_id": state,
            "cat_id": cat,
            "history": history_records,
            "forecast": forecast_records,
        }

    return payloads


# ---------------------------------------------------------------------------
# Inventory risk matrix
# ---------------------------------------------------------------------------


def build_inventory_risk_matrix(
    inventory_df: pl.DataFrame,
    sales_df: pl.DataFrame | None = None,
) -> dict:
    """Build the inventory risk matrix (store × department).

    Parameters
    ----------
    inventory_df:
        Simulation output (id, fill_rate, stockout_days, days_of_supply,
        avg_inventory, …) with item_id / store_id / dept metadata.
    sales_df:
        Optional raw sales for avg_demand reference.

    Returns
    -------
    dict — the risk matrix payload.
    """
    if inventory_df.is_empty():
        return {"stores": [], "synthetic_tag": SYNTHETIC_TAG}

    # Join metadata if not already present
    df = inventory_df.clone()
    if sales_df is not None and "store_id" not in df.columns:
        meta = (
            sales_df.select(["id", "store_id", "dept_id"])
            .unique("id")
        )
        df = df.join(meta, on="id", how="left")

    required_cols = ["store_id", "dept_id", "fill_rate", "stockout_days", "days_of_supply"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("Inventory risk matrix missing columns: %s", missing)
        return {"stores": [], "missing_columns": missing, "synthetic_tag": SYNTHETIC_TAG}

    # Aggregate to store × department
    agg = (
        df
        .group_by(["store_id", "dept_id"])
        .agg([
            pl.col("fill_rate").mean().alias("fill_rate"),
            pl.col("stockout_days").mean().alias("avg_stockout_days"),
            pl.col("days_of_supply").mean().alias("days_of_supply"),
            pl.len().alias("n_items"),
            # Items with fill_rate < 0.90 are "at risk"
            (pl.col("fill_rate") < 0.90).sum().alias("n_items_at_risk"),
        ])
        .sort(["store_id", "dept_id"])
    )

    # Build nested structure: {store_id: {dept_id: {...}}}
    stores_dict: dict[str, dict] = {}
    for row in agg.iter_rows(named=True):
        store = row["store_id"]
        dept = row["dept_id"]
        if store not in stores_dict:
            stores_dict[store] = {}
        fill = round(float(row["fill_rate"]), 4)
        dos = round(float(row["days_of_supply"]), 2)
        stores_dict[store][dept] = {
            "fill_rate": fill,
            "avg_stockout_days": round(float(row["avg_stockout_days"]), 2),
            "days_of_supply": dos,
            "n_items": int(row["n_items"]),
            "n_items_at_risk": int(row["n_items_at_risk"]),
            "overstock_flag": dos > 60,  # > 2 months of supply = overstock
            "stockout_probability": round(1.0 - fill, 4),
        }

    return {
        "stores": [
            {"store_id": store, "departments": dept_data}
            for store, dept_data in sorted(stores_dict.items())
        ],
        "synthetic_tag": SYNTHETIC_TAG,
    }


# ---------------------------------------------------------------------------
# Main export orchestrator
# ---------------------------------------------------------------------------


def export_serving_assets(
    output_dir: Path,
    sales_df: pl.DataFrame | None = None,
    forecast_df: pl.DataFrame | None = None,
    inventory_df: pl.DataFrame | None = None,
    metrics: dict | None = None,
    metrics_report_df: pl.DataFrame | None = None,
    n_history_days: int = 180,
    is_synthetic: bool = False,
    shap_summary: dict | None = None,
    coverage_report: dict | None = None,
    policy_comparison_summary: dict | None = None,
    scenario_results: dict | None = None,
) -> dict[str, Path]:
    """Generate all serving assets and write them to ``output_dir``.

    Parameters
    ----------
    output_dir:
        Destination directory.  Created if it does not exist.
    sales_df:
        Silver sales DataFrame.  If None, minimal synthetic data is used.
    forecast_df:
        Forecast DataFrame (id, date, forecast_p50/p10/p90).
    inventory_df:
        Inventory simulation results.
    metrics:
        Scalar metrics dict (from ``summarize_backtesting``).
    metrics_report_df:
        Segmented metrics DataFrame (from ``generate_segmented_report``).
    n_history_days:
        Days of history to include in forecast_series files.
    is_synthetic:
        Tag all outputs as SYNTHETIC.
    shap_summary:
        SHAP summary dict from ``run_shap_analysis`` (top-30 features).
        Written to ``shap_summary.json`` if provided.
    coverage_report:
        Conformal coverage report dict from ``run_conformal_calibration``.
        Written to ``coverage_report.json`` if provided.
    policy_comparison_summary:
        Policy comparison summary dict from ``build_policy_comparison_summary``.
        Written to ``policy_comparison_summary.json`` if provided.
    scenario_results:
        Scenario results dict from ``build_scenario_results_json``.
        Written to ``scenario_results.json`` if provided.

    Returns
    -------
    dict mapping asset name → Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fallback to minimal synthetic sales if none provided
    if sales_df is None:
        sales_df = _make_minimal_synthetic_sales()
        is_synthetic = True

    written: dict[str, Path] = {}

    # ── executive_summary.json ──────────────────────────────────────────────
    summary = build_executive_summary(
        sales_df,
        inventory_df=inventory_df,
        metrics=metrics,
        is_synthetic=is_synthetic,
    )
    written["executive_summary"] = _write_json(
        summary, output_dir / "executive_summary.json"
    )

    # ── forecast_series_{state}_{cat}.json ─────────────────────────────────
    fc_payloads = build_forecast_series(sales_df, forecast_df, n_history_days)
    for (state, cat), payload in fc_payloads.items():
        fname = f"forecast_series_{state}_{cat}.json"
        written[f"forecast_series_{state}_{cat}"] = _write_json(
            payload, output_dir / fname
        )

    # ── inventory_risk_matrix.json ─────────────────────────────────────────
    if inventory_df is not None:
        risk_matrix = build_inventory_risk_matrix(inventory_df, sales_df)
    else:
        risk_matrix = {"stores": [], "synthetic_tag": SYNTHETIC_TAG,
                       "note": "No inventory simulation data available"}
    written["inventory_risk_matrix"] = _write_json(
        risk_matrix, output_dir / "inventory_risk_matrix.json"
    )

    # ── model_metrics.json ─────────────────────────────────────────────────
    model_metrics_payload: dict = {
        "synthetic_tag": SYNTHETIC_TAG if is_synthetic else "M5_REAL",
    }
    if metrics:
        model_metrics_payload["aggregate"] = {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in metrics.items()
        }
    if metrics_report_df is not None and not metrics_report_df.is_empty():
        model_metrics_payload["segments"] = metrics_report_df.to_dicts()
    written["model_metrics"] = _write_json(
        model_metrics_payload, output_dir / "model_metrics.json"
    )

    # ── shap_summary.json ──────────────────────────────────────────────────
    if shap_summary is not None:
        shap_payload = {
            "top_features": shap_summary.get("top_features", []),
            "n_samples": shap_summary.get("n_samples", 0),
            "n_features": shap_summary.get("n_features", 0),
        }
        written["shap_summary"] = _write_json(
            shap_payload, output_dir / "shap_summary.json"
        )

    # ── coverage_report.json ───────────────────────────────────────────────
    if coverage_report is not None:
        written["coverage_report"] = _write_json(
            coverage_report, output_dir / "coverage_report.json"
        )

    # ── policy_comparison_summary.json ─────────────────────────────────────
    if policy_comparison_summary is not None:
        written["policy_comparison_summary"] = _write_json(
            policy_comparison_summary, output_dir / "policy_comparison_summary.json"
        )

    # ── scenario_results.json ──────────────────────────────────────────────
    if scenario_results is not None:
        written["scenario_results"] = _write_json(
            scenario_results, output_dir / "scenario_results.json"
        )

    # ── asset_manifest.json ────────────────────────────────────────────────
    manifest = _build_manifest(written)
    written["asset_manifest"] = _write_json(
        manifest, output_dir / "asset_manifest.json"
    )

    # Size report
    total_kb = sum(_size_kb(p) for p in written.values())
    logger.info(
        "Serving assets written: %d files, total %.1f KB",
        len(written),
        total_kb,
    )
    if total_kb > 5 * 1024:
        logger.warning("Total serving assets exceed 5 MB budget: %.1f KB", total_kb)

    return written


def _build_manifest(paths: dict[str, Path]) -> dict:
    """Build asset manifest with name, size_bytes, sha256 for each file."""
    from datetime import datetime as _dt

    manifest: dict = {
        "manifest_file": "asset_manifest.json",
        "generated_at": _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assets": [],
        "total_size_bytes": 0,
    }
    total = 0

    for name, path in sorted(paths.items()):
        if name == "asset_manifest":
            continue  # don't include manifest itself (not written yet)
        if not path.exists():
            continue
        size = path.stat().st_size
        total += size
        manifest["assets"].append({
            "name": path.name,
            "size_bytes": size,
            "size_kb": round(size / 1024, 2),
            "sha256": _sha256_file(path),
        })

    manifest["total_size_bytes"] = total
    manifest["total_size_kb"] = round(total / 1024, 2)
    return manifest


# ---------------------------------------------------------------------------
# Minimal synthetic sales fallback
# ---------------------------------------------------------------------------


def _make_minimal_synthetic_sales(
    n_days: int = 365,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a minimal synthetic sales DataFrame for export tests.

    Creates 3 states × 3 categories × 2 stores × 3 items = 54 series.
    """
    rng = np.random.default_rng(seed)
    states = ["CA", "TX", "WI"]
    cats = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    stores_per_state = {"CA": ["CA_1", "CA_2"], "TX": ["TX_1", "TX_2"], "WI": ["WI_1", "WI_2"]}
    depts = {"FOODS": "FOODS_1", "HOBBIES": "HOBBIES_1", "HOUSEHOLD": "HOUSEHOLD_1"}

    start = date(2015, 1, 1)
    rows: list[dict] = []

    for state in states:
        for store in stores_per_state[state]:
            for cat in cats:
                dept = depts[cat]
                for item_idx in range(3):
                    item_id = f"{cat[:3]}_{item_idx:03d}"
                    series_id = f"{item_id}_{store}"
                    avg_d = rng.uniform(1, 10)
                    for d in range(n_days):
                        sales_val = max(0, int(rng.poisson(avg_d)))
                        rows.append({
                            "id": series_id,
                            "item_id": item_id,
                            "store_id": store,
                            "state_id": state,
                            "cat_id": cat,
                            "dept_id": dept,
                            "date": start + timedelta(days=d),
                            "sales": sales_val,
                        })

    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
