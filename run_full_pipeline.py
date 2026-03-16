#!/usr/bin/env python
"""
run_full_pipeline.py — V1 Pipeline Orchestrator (Full 30,490 series)

Executes all pipeline stages in dependency order with timing, logging,
chunked feature processing (OOM-safe), and a final metrics report.

Usage:
    python run_full_pipeline.py [--force] [--start-from STEP]

Steps:
    1  extract        M5 zip → CSVs
    2  bronze         CSVs → Parquet
    3  silver         Bronze → Silver (long format, daily prices, calendar)
    4  classify       ADI/CV² demand classes + ABC/XYZ
    5  features       Feature store (85 features, chunked by state)
    6  train          LightGBM + backtesting (5 folds)
    7  inventory      Monte Carlo simulation + scenarios
    8  export         Generate serving JSONs
    9  copy           Copy JSONs to app/public/data/
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows terminals (cp1252 rejects box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"
RAW_M5 = DATA / "raw" / "m5"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
SERVING = GOLD / "serving"
FEATURES_DIR = DATA / "features"
MODELS_DIR = GOLD / "models"
BACKTESTING_DIR = GOLD / "backtesting"
INVENTORY_DIR = GOLD / "inventory_snapshot"
METRICS_DIR = GOLD / "metrics"
APP_DATA = ROOT / "app" / "public" / "data"

M5_ZIP = RAW_M5 / "m5-forecasting-accuracy.zip"
SILVER_SALES_DIR = SILVER / "silver_sales_long"
FEATURE_STORE_PATH = FEATURES_DIR / "feature_store_v1.parquet"
CLASSIFICATION_PATH = SILVER / "demand_classification.parquet"
SYNTHETIC_PARAMS = GOLD / "synthetic_params.parquet"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE = ROOT / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")


# ── Utilities ─────────────────────────────────────────────────────────────────

class PipelineError(RuntimeError):
    """Raised when a pipeline step fails unrecoverably."""


@contextmanager
def timed(label: str):
    """Context manager that logs elapsed time for a labelled block."""
    log.info(">> %s ...", label)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        log.info("OK %s -- %.1f s", label, elapsed)


# ── Step implementations ───────────────────────────────────────────────────────

def step1_extract(force: bool = False) -> dict[str, Any]:
    """Extract M5 zip to CSV files."""
    from src.ingest.m5_downloader import extract_m5_zip, verify_m5_files, M5_EXPECTED_FILES

    if not M5_ZIP.exists():
        raise PipelineError(
            f"M5 zip not found at {M5_ZIP}.\n"
            "Download from https://www.kaggle.com/competitions/m5-forecasting-accuracy/data "
            "and place it at data/raw/m5/m5-forecasting-accuracy.zip"
        )

    existing = [f for f in M5_EXPECTED_FILES if (RAW_M5 / f).exists()]
    if len(existing) == len(M5_EXPECTED_FILES) and not force:
        log.info("  CSVs already extracted (%d files). Skipping.", len(existing))
        return {"files_extracted": len(existing), "skipped": True}

    with timed("Extracting M5 zip"):
        paths = extract_m5_zip(M5_ZIP, RAW_M5, force=force)

    checksums = verify_m5_files(RAW_M5)
    log.info("  Extracted %d files. SHA-256 verified.", len(paths))
    return {"files_extracted": len(paths), "checksums": checksums}


def step2_bronze(force: bool = False) -> dict[str, Any]:
    """Convert M5 CSVs to Bronze Parquet."""
    from src.ingest.bronze_writer import write_m5_bronze, get_bronze_checksums

    existing = get_bronze_checksums(BRONZE)
    if existing and not force:
        log.info("  Bronze already exists (%d files). Skipping.", len(existing))
        return {"files_written": len(existing), "skipped": True}

    BRONZE.mkdir(parents=True, exist_ok=True)
    with timed("Writing Bronze Parquet"):
        checksums = write_m5_bronze(RAW_M5, BRONZE, force=force)

    log.info("  Written %d bronze files.", len(checksums))
    return {"files_written": len(checksums)}


def step3_silver(force: bool = False) -> dict[str, Any]:
    """Transform Bronze → Silver (long format + daily prices + calendar)."""
    from src.transform.pipeline import run_bronze_to_silver

    silver_sales_done = (SILVER / "silver_sales_long").exists()
    silver_prices_done = (SILVER / "silver_prices_daily.parquet").exists()
    if silver_sales_done and silver_prices_done and not force:
        log.info("  Silver already exists. Skipping.")
        return {"skipped": True}

    SILVER.mkdir(parents=True, exist_ok=True)
    with timed("Bronze → Silver transform"):
        run_bronze_to_silver(BRONZE, SILVER, force=force)

    # Count partitions written
    parts = list((SILVER / "silver_sales_long").rglob("*.parquet"))
    log.info("  Silver sales: %d partition files.", len(parts))
    return {"partition_files": len(parts)}


def step4_classify(force: bool = False) -> dict[str, Any]:
    """Compute ADI/CV² demand classes + ABC/XYZ classification."""
    import polars as pl
    from src.transform.pipeline import read_silver_sales
    from src.classification.demand_classifier import classify_all_series, save_demand_classification
    from src.classification.abc_xyz import enrich_with_abc_xyz, save_full_classification

    if CLASSIFICATION_PATH.exists() and not force:
        df = pl.read_parquet(CLASSIFICATION_PATH)
        log.info("  Classification already exists (%d series). Skipping.", len(df))
        _log_class_distribution(df)
        return {"n_series": len(df), "skipped": True}

    with timed("Loading silver sales (lazy scan)"):
        sales_lf = read_silver_sales(SILVER_SALES_DIR)
        sales_df = sales_lf.collect()
        log.info("  Sales rows: %s  cols: %s", f"{len(sales_df):,}", sales_df.columns)

    with timed("ADI / CV² demand classification"):
        classification_df = classify_all_series(sales_df)
        log.info("  Series classified: %s", f"{len(classification_df):,}")

    prices_path = SILVER / "silver_prices_daily.parquet"
    prices_df = pl.read_parquet(prices_path) if prices_path.exists() else None

    with timed("ABC / XYZ classification"):
        classification_df = enrich_with_abc_xyz(classification_df, sales_df, prices_df)

    CLASSIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_full_classification(classification_df, CLASSIFICATION_PATH)
    log.info("  Saved to %s", CLASSIFICATION_PATH)

    _log_class_distribution(classification_df)
    dist = _class_distribution(classification_df)
    del sales_df
    gc.collect()
    return {"n_series": len(classification_df), "distribution": dist}


def _log_class_distribution(df) -> None:
    import polars as pl
    if "demand_class" in df.columns:
        counts = (
            df.group_by("demand_class")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
        )
        for row in counts.iter_rows(named=True):
            pct = row["n"] / len(df) * 100
            log.info("    %-15s %5d  (%.1f%%)", row["demand_class"], row["n"], pct)


def _class_distribution(df) -> dict[str, float]:
    import polars as pl
    if "demand_class" not in df.columns:
        return {}
    counts = df.group_by("demand_class").agg(pl.len().alias("n")).sort("demand_class")
    total = len(df)
    return {row["demand_class"]: round(row["n"] / total * 100, 2) for row in counts.iter_rows(named=True)}


def step5_features(force: bool = False) -> dict[str, Any]:
    """Build feature store chunked by store_id (10 stores x ~3K series each)."""
    import polars as pl

    if FEATURE_STORE_PATH.exists() and not force:
        # Scan metadata only — no full read
        lf = pl.scan_parquet(FEATURE_STORE_PATH)
        schema = lf.schema
        total = lf.select(pl.len()).collect().item()
        cols = len(schema)
        log.info(
            "  Feature store already exists (%s rows, %d cols). Skipping.",
            f"{total:,}", cols,
        )
        return {"rows": total, "cols": cols, "skipped": True}

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    classification_path = CLASSIFICATION_PATH if CLASSIFICATION_PATH.exists() else None

    return _build_features_by_store(SILVER, FEATURE_STORE_PATH, classification_path)


# Shared silver files that every store chunk needs (prices, calendar, weather)
_SHARED_SILVER = [
    "silver_prices_daily.parquet",
    "silver_calendar_enriched.parquet",
    "silver_weather_daily.parquet",
]


def _build_features_by_store(
    silver_dir: Path,
    output_path: Path,
    classification_path: Path | None,
) -> dict[str, Any]:
    """Process features one store at a time to keep RAM < 4 GB per chunk.

    Strategy per store:
      1. Scan silver sales, filter to store_id, collect (~190K rows).
      2. Write into a temp Hive-partitioned dir (state=XX/year=YYYY/data.parquet).
      3. Symlink shared files (prices, calendar, weather) into the same temp dir.
      4. Call build_feature_store() on the temp dir — sees only that store's rows.
      5. Persist the chunk parquet; delete temp dir.
    Finally concatenate all 10 chunks into feature_store_v1.parquet.
    """
    import polars as pl
    from src.features.feature_store import build_feature_store
    from src.transform.pipeline import read_silver_sales

    # ── Discover stores from sales partitions ─────────────────────────────
    log.info("  Scanning store IDs from silver sales ...")
    stores: list[str] = (
        read_silver_sales(SILVER_SALES_DIR)
        .select("store_id")
        .unique()
        .sort("store_id")
        .collect()["store_id"]
        .to_list()
    )
    n_stores = len(stores)
    log.info("  Found %d stores: %s", n_stores, stores)

    chunks_dir = output_path.parent / "_feature_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    failed_stores: list[str] = []

    for idx, store_id in enumerate(stores, 1):
        chunk_out = chunks_dir / f"features_{store_id}.parquet"

        if chunk_out.exists():
            log.info("  Store %d/%d (%s): chunk already exists, skipping.",
                     idx, n_stores, store_id)
            chunk_paths.append(chunk_out)
            continue

        log.info("  Processing store %d/%d: %s ...", idx, n_stores, store_id)
        t_store = time.perf_counter()

        try:
            # 1. Load only this store's sales rows (lazy filter → collect)
            store_sales: pl.DataFrame = (
                read_silver_sales(SILVER_SALES_DIR)
                .filter(pl.col("store_id") == store_id)
                .collect()
            )
            n_rows_store = len(store_sales)
            log.info("    Loaded %s rows for %s", f"{n_rows_store:,}", store_id)

            with tempfile.TemporaryDirectory(prefix=f"feat_{store_id}_") as tmp:
                tmp_path = Path(tmp)
                tmp_silver = tmp_path / "silver"
                tmp_silver.mkdir()

                # 2. Write store sales in Hive partition layout
                #    state=XX/year=YYYY/data.parquet
                state = store_id.split("_")[0]   # e.g. "CA" from "CA_1"
                years: list[int] = (
                    store_sales.select(pl.col("date").dt.year().alias("yr"))
                    .unique()["yr"]
                    .sort()
                    .to_list()
                )
                for yr in years:
                    yr_dir = tmp_silver / "silver_sales_long" / f"state={state}" / f"year={yr}"
                    yr_dir.mkdir(parents=True, exist_ok=True)
                    (
                        store_sales.filter(pl.col("date").dt.year() == yr)
                        .write_parquet(yr_dir / "data.parquet")
                    )

                del store_sales
                gc.collect()

                # 3. Symlink shared files (Windows: copy if symlink fails)
                for fname in _SHARED_SILVER:
                    src = silver_dir / fname
                    if src.exists():
                        dst = tmp_silver / fname
                        try:
                            dst.symlink_to(src.resolve())
                        except OSError:
                            shutil.copy2(src, dst)

                # 4. Build features for this store only
                tmp_feat = tmp_path / "feat.parquet"
                chunk_df = build_feature_store(
                    tmp_silver,
                    tmp_feat,
                    force=True,
                    classification_path=classification_path,
                )

            # 5. Persist chunk (outside tempdir so it survives cleanup)
            rows, cols = len(chunk_df), len(chunk_df.columns)
            chunk_df.write_parquet(chunk_out)
            del chunk_df
            gc.collect()

            elapsed = time.perf_counter() - t_store
            log.info("    Done: %s rows x %d cols in %.1f s",
                     f"{rows:,}", cols, elapsed)
            chunk_paths.append(chunk_out)

        except Exception as exc:
            log.error("    Store %s FAILED (%s) -- skipping and continuing.",
                      store_id, exc)
            failed_stores.append(store_id)

    if not chunk_paths:
        raise PipelineError(
            "All stores failed during feature generation. "
            "Check logs above for per-store errors."
        )
    if failed_stores:
        log.warning("  %d store(s) skipped due to errors: %s",
                    len(failed_stores), failed_stores)

    # ── Concatenate ────────────────────────────────────────────────────────
    log.info("  Concatenating %d store chunks ...", len(chunk_paths))
    dfs = [pl.read_parquet(p) for p in chunk_paths]
    result = pl.concat(dfs, how="diagonal")
    result.write_parquet(output_path)
    rows, cols = len(result), len(result.columns)
    log.info("  Final feature store: %s rows x %d cols", f"{rows:,}", cols)

    shutil.rmtree(chunks_dir, ignore_errors=True)
    del result, dfs
    gc.collect()

    return {
        "rows": rows,
        "cols": cols,
        "method": "chunked_by_store",
        "n_stores": len(chunk_paths),
        "failed_stores": failed_stores,
    }


def step6_train_evaluate(force: bool = False) -> dict[str, Any]:
    """Train LightGBM (5 folds) + full backtesting evaluation."""
    import polars as pl
    from src.transform.pipeline import read_silver_sales
    from src.evaluation.backtesting import run_backtesting, summarize_backtesting, get_default_folds

    metrics_path = BACKTESTING_DIR / "summary_metrics.json"
    if metrics_path.exists() and not force:
        metrics = json.loads(metrics_path.read_text())
        log.info("  Backtesting metrics already exist. Skipping.")
        log.info("    MAE=%.4f  RMSE=%.4f  Coverage@80=%.1f%%",
                 metrics.get("mean_mae", 0), metrics.get("mean_rmse", 0),
                 metrics.get("mean_coverage_80", 0) * 100)
        return {"metrics": metrics, "skipped": True}

    log.info("  Loading silver sales …")
    sales_df = read_silver_sales(SILVER_SALES_DIR).collect()
    log.info("  Sales: %s rows", f"{len(sales_df):,}")

    log.info("  Loading feature store …")
    if not FEATURE_STORE_PATH.exists():
        raise PipelineError("Feature store not found — run step 5 first.")
    feature_df = pl.read_parquet(FEATURE_STORE_PATH)
    log.info("  Features: %s rows x %d cols", f"{len(feature_df):,}", len(feature_df.columns))

    folds = get_default_folds()
    log.info("  Starting backtesting — %d folds, horizon=28d each …", len(folds))

    BACKTESTING_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with timed("Backtesting (5 folds × LightGBM)"):
        results = run_backtesting(
            sales_df=sales_df,
            feature_df=feature_df,
            output_dir=BACKTESTING_DIR,
            folds=folds,
            use_mlflow=True,
        )

    summary = summarize_backtesting(results)
    metrics_path.write_text(json.dumps(summary, indent=2))

    log.info("  -- Backtesting Results ------------------------------------------")
    log.info("    MAE          = %.4f +/- %.4f", summary.get("mean_mae", 0), summary.get("std_mae", 0))
    log.info("    RMSE         = %.4f +/- %.4f", summary.get("mean_rmse", 0), summary.get("std_rmse", 0))
    log.info("    sMAPE        = %.2f%%", summary.get("mean_smape", 0) * 100)
    log.info("    Bias         = %+.4f", summary.get("mean_bias", 0))
    log.info("    Coverage@80  = %.1f%%", summary.get("mean_coverage_80", 0) * 100)
    log.info("    Pinball      = %.4f", summary.get("mean_pinball_loss", 0))
    log.info("  -----------------------------------------------------------------")

    del sales_df, feature_df
    gc.collect()
    return {"n_folds": len(results), "metrics": summary}


def step7_inventory(force: bool = False) -> dict[str, Any]:
    """Run Monte Carlo inventory simulation + scenario analysis."""
    import numpy as np
    import polars as pl
    from src.transform.pipeline import read_silver_sales
    from src.inventory.safety_stock import compute_safety_stock_batch
    from src.inventory.engine import run_inventory_simulation, save_inventory_snapshot
    from src.inventory.simulator import run_mc_batch
    from src.inventory.scenario_engine import run_scenario_batch, build_scenario_results_json, export_scenario_results

    snapshot_path = INVENTORY_DIR / "fact_inventory_snapshot.parquet"
    if snapshot_path.exists() and not force:
        df = pl.read_parquet(snapshot_path)
        fill_rate = df["fill_rate"].mean() if "fill_rate" in df.columns else None
        log.info("  Inventory snapshot already exists (%s rows). Skipping.", f"{len(df):,}")
        return {"n_series": len(df), "fill_rate_mean": fill_rate, "skipped": True}

    if not SYNTHETIC_PARAMS.exists():
        raise PipelineError(
            f"Synthetic params not found at {SYNTHETIC_PARAMS}.\n"
            "Run the inventory param generator or check data/gold/."
        )

    log.info("  Loading data …")
    sales_df = read_silver_sales(SILVER_SALES_DIR).collect()
    params_df = pl.read_parquet(SYNTHETIC_PARAMS)
    log.info("  Params: %s series", f"{len(params_df):,}")

    # Load latest backtesting predictions (used as forecast proxy)
    forecast_df: pl.DataFrame | None = None
    pred_files = sorted(BACKTESTING_DIR.glob("fold_*/predictions.parquet"))
    if pred_files:
        log.info("  Loading predictions from %d fold files …", len(pred_files))
        forecast_df = pl.concat([pl.read_parquet(p) for p in pred_files])

    # Compute safety stock
    log.info("  Computing safety stock …")
    with timed("Safety stock batch"):
        ss_df = compute_safety_stock_batch(params_df, forecast_df)

    # Reorder point: ROP = avg_demand × lead_time + safety_stock
    if "avg_daily_demand" in params_df.columns and "lead_time_days" in params_df.columns:
        rop_df = params_df.select([
            pl.col("id") if "id" in params_df.columns else pl.concat_str(["item_id", "store_id"], separator="_").alias("id"),
            (pl.col("avg_daily_demand") * pl.col("lead_time_days")).alias("reorder_point"),
        ])
    else:
        # Fallback: ROP from safety stock
        rop_df = ss_df.with_columns(
            (pl.col("safety_stock") * 2).alias("reorder_point")
        ).select(["id", "reorder_point"])

    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)

    with timed("Inventory simulation (engine)"):
        inv_df = run_inventory_simulation(
            sales_df=sales_df,
            params_df=params_df,
            ss_df=ss_df,
            rop_df=rop_df,
            forecast_df=forecast_df,
        )

    save_inventory_snapshot(inv_df, INVENTORY_DIR)

    fill_rate_mean = inv_df["fill_rate"].mean() if "fill_rate" in inv_df.columns else None
    stockout_prob = inv_df["stockout_probability"].mean() if "stockout_probability" in inv_df.columns else None
    log.info("  Fill rate (mean):       %.1f%%", (fill_rate_mean or 0) * 100)
    log.info("  Stockout prob (mean):   %.1f%%", (stockout_prob or 0) * 100)

    # ── Monte Carlo (sample: 500 series for speed) ─────────────────────────
    log.info("  Running Monte Carlo simulation (sample 500 series) …")
    series_list = _build_mc_series_list(params_df, forecast_df, n_sample=500)
    with timed("Monte Carlo (500 series × 1000 paths)"):
        mc_results = run_mc_batch(series_list, n_simulations=1000, horizon_days=90)

    fill_rates = [r.fill_rate_mean for r in mc_results]
    log.info("  MC fill rate p50=%.1f%%  p5=%.1f%%  p95=%.1f%%",
             np.percentile(fill_rates, 50) * 100,
             np.percentile(fill_rates, 5) * 100,
             np.percentile(fill_rates, 95) * 100)

    # ── Scenario analysis (first 50 series) ───────────────────────────────
    log.info("  Running scenario analysis (50 series) …")
    scenario_series = series_list[:50]
    with timed("Scenario engine (50 series × 5 scenarios)"):
        scenario_results = run_scenario_batch(scenario_series, n_simulations=1000, horizon_days=90)

    SERVING.mkdir(parents=True, exist_ok=True)
    export_scenario_results(scenario_results, SERVING)
    log.info("  Scenario results saved.")

    del sales_df, inv_df
    gc.collect()

    return {
        "n_series": len(params_df),
        "fill_rate_mean": fill_rate_mean,
        "stockout_prob_mean": stockout_prob,
        "mc_sample": len(mc_results),
        "scenario_sample": len(scenario_results),
    }


def _build_mc_series_list(
    params_df,
    forecast_df,
    n_sample: int = 500,
) -> list[dict]:
    """Build series_list dicts for run_mc_batch from params + forecast."""
    import numpy as np
    import polars as pl

    # Get id column name
    id_col = "id" if "id" in params_df.columns else None
    if id_col is None and "item_id" in params_df.columns:
        params_df = params_df.with_columns(
            pl.concat_str(["item_id", "store_id"], separator="_").alias("id")
        )
        id_col = "id"

    params_sample = params_df.sample(min(n_sample, len(params_df)), seed=42)

    series_list = []
    for row in params_sample.iter_rows(named=True):
        sid = row.get("id", "unknown")
        lt_mean = float(row.get("lead_time_days", 7))
        lt_std = max(1.0, lt_mean * 0.2)

        # Extract forecast arrays from forecast_df or use defaults
        if forecast_df is not None and id_col in forecast_df.columns:
            s_fc = (
                forecast_df
                .filter(pl.col(id_col) == sid)
                .sort("date" if "date" in forecast_df.columns else forecast_df.columns[1])
                .head(90)
            )
            if len(s_fc) >= 10:
                p10 = s_fc["p10"].to_numpy() if "p10" in s_fc.columns else np.full(90, 1.0)
                p50 = s_fc["p50"].to_numpy() if "p50" in s_fc.columns else np.full(90, 3.0)
                p90 = s_fc["p90"].to_numpy() if "p90" in s_fc.columns else np.full(90, 6.0)
                # Ensure length 90
                p10 = np.resize(p10, 90)
                p50 = np.resize(p50, 90)
                p90 = np.resize(p90, 90)
            else:
                avg = float(row.get("avg_daily_demand", 3.0))
                p10, p50, p90 = np.full(90, avg * 0.5), np.full(90, avg), np.full(90, avg * 1.8)
        else:
            avg = float(row.get("avg_daily_demand", 3.0))
            p10, p50, p90 = np.full(90, avg * 0.5), np.full(90, avg), np.full(90, avg * 1.8)

        rop = float(row.get("reorder_point", p50.mean() * lt_mean))
        oq = float(row.get("order_quantity", max(10.0, p50.sum() / 4)))
        initial_stock = oq * 1.5

        series_list.append({
            "series_id": sid,
            "forecast_p10": p10,
            "forecast_p50": p50,
            "forecast_p90": p90,
            "initial_stock": initial_stock,
            "lead_time_mean": lt_mean,
            "lead_time_std": lt_std,
            "reorder_point": rop,
            "order_quantity": oq,
            "holding_cost_per_unit_day": float(row.get("holding_cost_pct", 0.20)) / 365,
            "stockout_cost_per_unit": 1.0,
            "fixed_order_cost": 10.0,
        })

    return series_list


def step8_export_serving(force: bool = False) -> dict[str, Any]:
    """Generate all serving JSON assets."""
    import polars as pl
    from src.transform.pipeline import read_silver_sales
    from src.export.serving_exporter import export_serving_assets

    manifest_path = SERVING / "asset_manifest.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        log.info("  Serving assets already exist. Skipping.")
        return {"n_assets": len(manifest.get("assets", manifest)), "skipped": True}

    SERVING.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info("  Loading sales for serving export …")
    sales_df = read_silver_sales(SILVER_SALES_DIR).collect()

    # Load forecast (last fold predictions)
    forecast_df: pl.DataFrame | None = None
    pred_files = sorted(BACKTESTING_DIR.glob("fold_*/predictions.parquet"))
    if pred_files:
        forecast_df = pl.read_parquet(pred_files[-1])
        log.info("  Using forecast from: %s", pred_files[-1].name)

    # Load inventory
    inventory_df: pl.DataFrame | None = None
    inv_path = INVENTORY_DIR / "fact_inventory_snapshot.parquet"
    if inv_path.exists():
        inventory_df = pl.read_parquet(inv_path)

    # Load backtesting metrics
    metrics: dict = {}
    metrics_path = BACKTESTING_DIR / "summary_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    # Load SHAP summary (if exists)
    shap_summary: dict | None = None
    shap_path = BACKTESTING_DIR / "shap_summary.json"
    if shap_path.exists():
        shap_summary = json.loads(shap_path.read_text())

    # Load scenario results (if exists)
    scenario_results: dict | None = None
    scen_path = SERVING / "scenario_results.json"
    if scen_path.exists():
        scenario_results = json.loads(scen_path.read_text())

    with timed("Exporting serving assets"):
        asset_paths = export_serving_assets(
            output_dir=SERVING,
            sales_df=sales_df,
            forecast_df=forecast_df,
            inventory_df=inventory_df,
            metrics=metrics,
            shap_summary=shap_summary,
            scenario_results=scenario_results,
            is_synthetic=True,
        )

    sizes = {name: Path(p).stat().st_size for name, p in asset_paths.items() if Path(p).exists()}
    total_kb = sum(sizes.values()) / 1024
    log.info("  Assets generated: %d files, total %.1f KB", len(asset_paths), total_kb)
    for name, size in sizes.items():
        log.info("    %-40s %6.1f KB", name, size / 1024)

    if total_kb > 5120:
        log.warning("  WARN: Total serving bundle > 5 MB (%.1f KB)", total_kb)

    del sales_df
    gc.collect()
    return {"n_assets": len(asset_paths), "total_kb": round(total_kb, 1)}


def step9_copy_to_app(force: bool = False) -> dict[str, Any]:
    """Copy serving JSONs to app/public/data/."""
    APP_DATA.mkdir(parents=True, exist_ok=True)
    json_files = list(SERVING.glob("*.json"))
    copied = []
    for src in json_files:
        dst = APP_DATA / src.name
        if not dst.exists() or force or src.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(src, dst)
            copied.append(src.name)

    log.info("  Copied %d JSON files to app/public/data/", len(copied))
    if copied:
        log.info("    %s", "  ".join(copied))
    return {"copied": len(copied), "destination": str(APP_DATA)}


# ── Report ─────────────────────────────────────────────────────────────────────

def save_pipeline_report(step_results: dict[str, Any], step_times: dict[str, float]) -> Path:
    """Assemble and save the full pipeline report to data/gold/metrics/."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "full_pipeline_report.json"

    # Extract key metrics from step results
    classify_res = step_results.get("classify", {})
    train_res = step_results.get("train", {})
    features_res = step_results.get("features", {})
    inventory_res = step_results.get("inventory", {})
    export_res = step_results.get("export", {})

    bt_metrics = train_res.get("metrics", {})
    inv_fill = inventory_res.get("fill_rate_mean")

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "V1-full",
        "n_series_total": classify_res.get("n_series", 0),
        "demand_class_distribution": classify_res.get("distribution", {}),
        "feature_store": {
            "rows": features_res.get("rows", 0),
            "cols": features_res.get("cols", 0),
            "method": features_res.get("method", "unknown"),
        },
        "backtesting": {
            "n_folds": train_res.get("n_folds", 5),
            "mae_mean": bt_metrics.get("mean_mae"),
            "mae_std": bt_metrics.get("std_mae"),
            "rmse_mean": bt_metrics.get("mean_rmse"),
            "smape_mean": bt_metrics.get("mean_smape"),
            "bias_mean": bt_metrics.get("mean_bias"),
            "coverage_80_mean": bt_metrics.get("mean_coverage_80"),
            "pinball_loss_mean": bt_metrics.get("mean_pinball_loss"),
        },
        "inventory": {
            "n_series_simulated": inventory_res.get("n_series", 0),
            "fill_rate_mean": round(inv_fill * 100, 2) if inv_fill else None,
            "stockout_prob_mean": inventory_res.get("stockout_prob_mean"),
            "mc_sample": inventory_res.get("mc_sample", 0),
        },
        "serving_export": {
            "n_assets": export_res.get("n_assets", 0),
            "total_kb": export_res.get("total_kb", 0),
        },
        "step_times_seconds": {k: round(v, 1) for k, v in step_times.items()},
        "total_time_seconds": round(sum(step_times.values()), 1),
    }

    out_path.write_text(json.dumps(report, indent=2))
    log.info("  Report saved to %s", out_path)
    return out_path


def print_final_summary(step_results: dict, step_times: dict, total_time: float) -> None:
    classify_res = step_results.get("classify", {})
    train_res = step_results.get("train", {})
    features_res = step_results.get("features", {})
    inventory_res = step_results.get("inventory", {})
    export_res = step_results.get("export", {})
    bt_metrics = train_res.get("metrics", {})
    inv_fill = inventory_res.get("fill_rate_mean")

    sep = "=" * 65
    lines = [
        "",
        sep,
        "  PIPELINE COMPLETE -- AI Supply Chain Control Tower",
        sep,
        f"  Total runtime:         {total_time / 60:.1f} min",
        f"  Series classified:     {classify_res.get('n_series', 0):,}",
        f"  Feature store:         {features_res.get('rows', 0):,} rows x {features_res.get('cols', 0)} cols",
        f"  Build method:          {features_res.get('method', '-')}",
        "",
        "  -- Backtesting Results ------------------------------------------",
        f"  MAE (mean):            {bt_metrics.get('mean_mae', 0):.4f}",
        f"  RMSE (mean):           {bt_metrics.get('mean_rmse', 0):.4f}",
        f"  sMAPE (mean):          {bt_metrics.get('mean_smape', 0) * 100:.2f}%",
        f"  Bias:                  {bt_metrics.get('mean_bias', 0):+.4f}",
        f"  Coverage@80:           {bt_metrics.get('mean_coverage_80', 0) * 100:.1f}%",
        "",
        "  -- Inventory ----------------------------------------------------",
        f"  Fill rate (mean):      {(inv_fill or 0) * 100:.1f}%",
        f"  MC sample:             {inventory_res.get('mc_sample', 0)} series",
        "",
        "  -- Serving Export -----------------------------------------------",
        f"  JSON assets:           {export_res.get('n_assets', 0)} files",
        f"  Bundle size:           {export_res.get('total_kb', 0):.1f} KB",
        "",
        "  -- Step Times ---------------------------------------------------",
    ]
    for step, secs in step_times.items():
        lines.append(f"  {step:<20}   {secs:>6.1f} s")
    lines += [
        sep,
        "  Report: data/gold/metrics/full_pipeline_report.json",
        sep,
        "",
    ]
    print("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

STEPS = {
    1: ("extract",  "Extract M5 zip"),
    2: ("bronze",   "Bronze writer"),
    3: ("silver",   "Silver transform"),
    4: ("classify", "Demand classification"),
    5: ("features", "Feature store"),
    6: ("train",    "Train + evaluate"),
    7: ("inventory","Inventory simulation"),
    8: ("export",   "Export serving JSONs"),
    9: ("copy",     "Copy to app/"),
}

STEP_FNS = {
    "extract":   step1_extract,
    "bronze":    step2_bronze,
    "silver":    step3_silver,
    "classify":  step4_classify,
    "features":  step5_features,
    "train":     step6_train_evaluate,
    "inventory": step7_inventory,
    "export":    step8_export_serving,
    "copy":      step9_copy_to_app,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full V1 pipeline (30,490 series)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-run all steps even if outputs already exist",
    )
    parser.add_argument(
        "--start-from", "-s",
        type=int,
        default=1,
        metavar="STEP",
        choices=range(1, 10),
        help="Start from step N (1-9). Skips earlier steps. Default: 1",
    )
    parser.add_argument(
        "--only", "-o",
        type=int,
        metavar="STEP",
        choices=range(1, 10),
        help="Run ONLY step N (useful for debugging)",
    )
    args = parser.parse_args()

    log.info("=" * 65)
    log.info("  AI Supply Chain Control Tower -- Full Pipeline")
    log.info("  Started: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("  Force re-run: %s", args.force)
    log.info("=" * 65)

    step_results: dict[str, Any] = {}
    step_times: dict[str, float] = {}
    pipeline_start = time.perf_counter()

    # Determine which steps to run
    if args.only is not None:
        steps_to_run = {args.only: STEPS[args.only]}
    else:
        steps_to_run = {k: v for k, v in STEPS.items() if k >= args.start_from}

    for step_num, (step_key, step_label) in steps_to_run.items():
        log.info("")
        log.info("+-- Step %d / 9: %s", step_num, step_label)

        step_fn = STEP_FNS[step_key]
        step_start = time.perf_counter()
        try:
            result = step_fn(force=args.force)
            step_results[step_key] = result or {}
        except PipelineError as exc:
            log.error("+-- FAILED: %s", exc)
            log.error("    Cannot continue. Fix the issue and re-run with --start-from %d", step_num)
            return 1
        except Exception as exc:
            log.exception("+-- UNEXPECTED ERROR in step %d (%s): %s", step_num, step_key, exc)
            return 1

        elapsed = time.perf_counter() - step_start
        step_times[step_key] = elapsed
        log.info("+-- Done in %.1f s", elapsed)

    total_time = time.perf_counter() - pipeline_start
    log.info("")
    log.info("All steps completed in %.1f s (%.1f min)", total_time, total_time / 60)

    # Save report
    try:
        save_pipeline_report(step_results, step_times)
    except Exception as exc:
        log.warning("Could not save report: %s", exc)

    print_final_summary(step_results, step_times, total_time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
