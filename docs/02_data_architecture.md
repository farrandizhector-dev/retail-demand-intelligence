# Data Architecture — AI Supply Chain Control Tower

**Document type:** Technical architecture
**Audience:** Data engineers, analytics engineers, technical reviewers
**Last updated:** 2026-03-15

---

## 1. Data Flow: Raw → Bronze → Silver → Gold

```
Raw / Landing
  ├── data/raw/m5/           M5 CSV files (Kaggle)
  ├── data/raw/weather/      Open-Meteo JSON responses
  └── data/raw/macro/        FRED API JSON responses

Bronze Layer  (src/ingest/ + src/transform/)
  ├── bronze_sales.parquet        Wide-format sales (as-is from M5)
  ├── bronze_calendar.parquet     Calendar events, SNAP flags
  ├── bronze_prices.parquet       Weekly sell prices
  ├── bronze_weather.parquet      Daily weather by state
  └── bronze_macro.parquet        Monthly macro indicators

Silver Layer  (src/transform/)
  ├── silver_sales_long.parquet          Long format, partitioned state/year
  ├── silver_prices_daily.parquet        Forward-filled weekly -> daily
  ├── silver_calendar_enriched.parquet   Holidays, events, SNAP enriched
  ├── silver_weather_daily.parquet       Aligned to sales dates
  ├── silver_macro_daily.parquet         Forward-filled with publication lag
  └── silver_demand_classification.parquet  ADI/CV2 + ABC/XYZ per SKU-store

Gold / Warehouse  (sql/intermediate/ + sql/marts/ via dbt)
  ├── fact_sales_daily         ~58.3M rows, grain: date x store x item
  ├── fact_price_daily         Daily prices (interpolated from weekly)
  ├── fact_forecast_daily      Model predictions, all folds, all models
  ├── fact_inventory_snapshot  Daily inventory state (SYNTHETIC)
  ├── dim_product (SCD Type 2) Item classifications with version history
  ├── dim_store                Store metadata and geography
  ├── dim_date                 Date spine with Walmart week mapping
  └── mart_executive           Pre-aggregated KPIs for dashboard
```

---

## 2. Domain Ownership Model

Each data domain has a responsible owner and clear SLA contracts with consumers. This is the model used in real enterprise data platforms:

```
+------------------------------------------------------------------+
|                    DATA DOMAIN OWNERSHIP                          |
+-----------------+------------------+-----------------------------+
| Domain          | Owner (role)     | Responsibilities            |
+-----------------+------------------+-----------------------------+
| Demand          | Data Engineer    | M5 ingestion, data quality, |
| (raw->silver)   |                  | contracts, long-format      |
+-----------------+------------------+-----------------------------+
| Enrichment      | Data Engineer    | Weather API, FRED API,      |
| (external)      |                  | rate limits, caching        |
+-----------------+------------------+-----------------------------+
| Warehouse       | Analytics Eng.   | dbt models, marts,          |
| (gold/marts)    |                  | schema evolution, SCD       |
+-----------------+------------------+-----------------------------+
| Features        | ML Engineer      | Feature store, leakage      |
|                 |                  | guard, versioning           |
+-----------------+------------------+-----------------------------+
| Models          | Data Scientist   | Training, evaluation,       |
|                 |                  | model cards, registry       |
+-----------------+------------------+-----------------------------+
| Reconciliation  | Data Scientist   | Hierarchy def, method       |
|                 |                  | selection, coherence tests  |
+-----------------+------------------+-----------------------------+
| Inventory Sim   | Operations Res.  | Synthetic gen, simulation,  |
|                 |                  | policy comparison           |
+-----------------+------------------+-----------------------------+
| Serving         | Product Eng.     | JSON export, compression,   |
|                 |                  | asset manifest, < 5MB       |
+-----------------+------------------+-----------------------------+
| Frontend        | Product Eng.     | Dashboard, UX, deploy,      |
|                 |                  | accessibility               |
+-----------------+------------------+-----------------------------+
| Monitoring      | Platform Eng.    | Drift detection, alerts,    |
|                 |                  | runbooks, observability     |
+-----------------+------------------+-----------------------------+
```

---

## 3. Inter-Domain Contracts

Explicit contracts between domains prevent interface drift and enable independent evolution:

| Provider → Consumer | Interface | SLA | Schema |
|--------------------|-----------|-----|--------|
| Demand → Warehouse | `data/silver/sales/*.parquet` | Updated within 1h of raw data change | `contracts/silver_sales.yaml` |
| Warehouse → Features | PostgreSQL marts via dbt | Refreshed after silver update | `sql/marts/schema.yml` |
| Features → Models | `data/features/feature_store_v{N}.parquet` | Versioned and immutable | `configs/features.yaml` |
| Models → Inventory | `data/gold/forecasts/*.parquet` | Includes p10/p50/p90 for all series | `contracts/forecast_output.yaml` |
| Inventory → Serving | `data/gold/serving/*.json` | < 5MB total compressed | `contracts/serving_assets.yaml` |

**Breaking change policy:** any schema change that removes or renames a column in a shared interface requires notifying all consumers at least 1 sprint in advance.

---

## 4. Data Quality Strategy

Pandera contract validation runs at every layer transition:

| Transition | Checks | Action on failure |
|-----------|--------|------------------|
| raw → bronze | File checksums, row counts, column presence | `DATA_QUALITY_FAILURE` alert → pipeline blocked |
| bronze → silver | Schema types, null rates, temporal coverage | Quarantine to `data/quarantine/{timestamp}/` |
| silver → features | No future data leakage (6 rules), value ranges | Fail-fast with detailed error message |
| features → gold | Serving asset size budget < 5MB | `SERVING_BUDGET_EXCEEDED` alert |

Contracts are defined in `contracts/*.yaml` and enforced by `src/validation/contracts.py`.

---

## 5. Lineage Traceability

Every KPI visible on the dashboard can be traced back to its source raw data:

```
Dashboard KPI: "Fill Rate 94.2% (CA, FOODS)"
  | derived from
mart_executive.fill_rate_avg
  | aggregated from
fact_inventory_snapshot.is_stockout (SYNTHETIC)
  | generated by
src/inventory/simulator.py (Monte Carlo, seed=42)
  | using demand from
fact_forecast_daily.forecast_p50 (LightGBM global, fold 5)
  | trained on features from
data/features/feature_store_v1.parquet (cutoff: 2016-03-27)
  | built from
data/silver/sales/state=CA/*.parquet (SHA-256 logged in bronze)
  | extracted from
data/raw/m5/sales_train_validation.csv (SHA-256 in asset_manifest.json)
```

---

## 6. Design Principles

1. **Idempotence:** same input → same output, no hidden side effects
2. **Traceable lineage:** from any KPI to raw data row
3. **Strict separation:** observed data != calculated features != predictions != simulations
4. **Config-driven:** all thresholds, policies, hyperparameters in YAML
5. **Fail-fast:** contract validation at every layer transition
6. **Reproducibility:** fixed seeds, pinned versions, SHA-256 checksums in MLflow
7. **Clear ownership:** every domain has an owner and SLA with consumers

---

## 7. Partitioning Strategy

Silver sales data is partitioned by `(state, year)` for efficient querying:

```
data/silver/sales/
  state=CA/
    year=2011/part-0.parquet
    year=2012/part-0.parquet
    ...
    year=2016/part-0.parquet
  state=TX/
    ...
  state=WI/
    ...
```

This partitioning allows the feature engineering pipeline to process one state at a time, keeping peak memory usage below 8 GB even on the full 58.3M row dataset. Polars lazy evaluation reads only the partitions needed for each query.

---

## 8. Bronze Layer SHA-256 Checksum Policy

Every file written to the bronze layer has its SHA-256 hash logged to `data/bronze/checksums.json`. This enables:

- **Reproducibility:** given the same checksums, the full pipeline output is deterministic
- **Audit trail:** MLflow run artifacts include the checksum file
- **Data freshness detection:** if checksums change between runs, downstream layers are invalidated and re-processed
