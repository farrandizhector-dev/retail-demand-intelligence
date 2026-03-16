# Decision Log — AI Supply Chain Control Tower

**Project:** Retail Demand Intelligence & Inventory Optimization Platform
**Author:** Héctor Ferrándiz Sanchis
**Last updated:** 2026-03-15

This log records all significant architectural and design decisions (Architecture Decision Records) made during the project lifecycle. Each ADR includes the context, decision, consequences, and current status.

---

## ADR-001: Polars as Primary Data Processor

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted |
| **Deciders** | Héctor Ferrándiz Sanchis |

### Context
The M5 dataset contains ~58M rows (30,490 series × 1,913 days). Processing this with Pandas requires significant memory and CPU time. Multiple libraries are available: Pandas, Polars, DuckDB, Spark.

### Decision
Use **Polars** as the primary DataFrame library for all ETL and feature engineering. Pandas is allowed only for LightGBM compatibility shims (`to_pandas()` before `lgb.Dataset`).

### Consequences
- **Positive**: 5–10× faster than Pandas on 58M rows; lazy evaluation for memory efficiency; strict null safety
- **Negative**: Smaller ecosystem (fewer third-party integrations); team must learn Polars API
- **Mitigation**: Polars-Pandas interop is one-line; maintained `_to_pandas()` helpers where needed

---

## ADR-002: LightGBM Global Model Strategy

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted |

### Context
30,490 individual series need forecasts. Options: (a) one model per series (local), (b) one model for all series (global), (c) hybrid per-segment.

### Decision
Train a **single LightGBM model** across all series simultaneously (global model strategy). Series identity is encoded as categorical features (store_id, dept_id, cat_id).

### Consequences
- **Positive**: Scalable (single fit/predict call); cross-series information sharing; proven in top M5 solutions; single MLflow run per fold
- **Negative**: Under-fits niche series with unique patterns; cannot specialise per series without additional complexity
- **Mitigation**: Demand class segmentation (ADR-003) + Croston TSB for intermittent outliers

---

## ADR-003: ADI/CV² Classification Before Modelling

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted |

### Context
M5 has >50% series with intermittent or lumpy demand (many zero observations). Applying a single forecasting strategy to all demand types leads to poor quantile coverage.

### Decision
Classify all series into four demand types using the **ADI/CV² framework** (Syntetos-Boylan-Croston):
- **Smooth**: ADI ≤ 1.32, CV² ≤ 0.49
- **Erratic**: ADI ≤ 1.32, CV² > 0.49
- **Intermittent**: ADI > 1.32, CV² ≤ 0.49
- **Lumpy**: ADI > 1.32, CV² > 0.49

### Consequences
- **Positive**: Differentiating portfolio feature; enables class-conditional model selection; honest performance reporting by segment
- **Negative**: Adds classification step before features/modelling; thresholds are heuristic
- **Mitigation**: Thresholds from Syntetos & Boylan (2005) literature; classification run once at silver layer

---

## ADR-004: Synthetic Inventory Layer with Transparent Labelling

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted |

### Context
M5 contains sales data but no inventory parameters (lead times, holding costs, order costs). Inventory optimisation (safety stock, ROP, EOQ) requires these parameters.

### Decision
Generate **synthetic inventory parameters** from plausible retail distributions:
- Lead time: Uniform(2, 7) days
- Holding cost: Uniform(0.10, 0.30) × unit cost per day
- Order cost: Uniform(10, 50) USD per order

All synthetic parameters and derived inventory metrics are **explicitly labelled SYNTHETIC** in code, JSON serving assets, and the frontend.

### Consequences
- **Positive**: Enables full end-to-end demonstration of inventory optimisation pipeline; honest technical communication
- **Negative**: Inventory decisions cannot be used for real business operations
- **Mitigation**: "SYNTHETIC DATA" badge on all frontend displays; prominent disclaimer on About page

---

## ADR-005: Three-Release Delivery Strategy (V1 → V2 → V3)

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted |

### Context
The full spec covers 15+ technical capabilities. Building everything at once risks never shipping a working demo.

### Decision
Structure delivery into three sequential releases:
- **V1 (Core Demo)**: Ingest → Features → LightGBM → Inventory basics → 4 frontend pages → Deploy
- **V2 (Technical Depth)**: Hierarchical reconciliation → Conformal intervals → Monte Carlo simulation → 7 frontend pages → MLflow
- **V3 (Enterprise Polish)**: Monitoring → SCD Type 2 → CI/CD → E2E tests → documentation

No V2 work begins until V1 is complete. No V3 work begins until V2 is complete.

### Consequences
- **Positive**: Working demo available after V1; each release adds verifiable depth; prevents scope creep
- **Negative**: Some V2 features could simplify V1 implementation (e.g., conformal could replace ad-hoc quantile training)
- **Mitigation**: V1 quantile training is reused/augmented by conformal in V2 (no throw-away work)

---

## ADR-006: Pre-Computed JSON for Frontend (No Live Inference)

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted |

### Context
The target deploy platform is **Hugging Face Static Space** (pure static file hosting — no Python runtime). Running LightGBM inference in the browser is infeasible for 30K series.

### Decision
The export pipeline (`src/export/serving_exporter.py`) generates **13 pre-computed JSON files** covering all dashboard views. The frontend reads these static files; there is no live model inference path.

### Consequences
- **Positive**: Zero-cost static hosting; instant load times; no backend maintenance
- **Negative**: Forecasts are stale after pipeline re-run cadence; no interactive what-if at row level (aggregated scenarios only)
- **Mitigation**: Scenario engine pre-computes 5 what-if scenarios; manifest includes generation timestamp

---

## ADR-007: ECharts over Plotly for Dashboards

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted |

### Context
The frontend requires 10+ charts across 7 pages including heatmaps, time series, bar charts, and donut charts. Candidates: Recharts, Plotly.js, ECharts, Victory.

### Decision
Use **Apache ECharts 5.x** via the `echarts` npm package with tree-shakeable imports.

### Consequences
- **Positive**: Best-in-class performance on dense dashboards (Canvas renderer); rich chart types including heatmap and data zoom; excellent dark theme support
- **Negative**: Larger bundle than Recharts (~628 KB gzipped ECharts chunk); steeper API learning curve
- **Mitigation**: Tree-shaking used (echarts/core with explicit chart/component imports); single BaseEChart wrapper component shared by all pages

---

## ADR-008: Hierarchical Reconciliation with MinT Shrinkage

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted (V2-Fase 1) |

### Context
Forecasts at different hierarchy levels (SKU-store → dept → category → store → state → total) may be incoherent: individual forecasts do not sum to aggregate forecasts. This breaks inventory planning at higher aggregation levels.

### Decision
Implement **MinT with shrinkage estimator** as the primary reconciliation method, with Bottom-Up and Top-Down as fallbacks. The S matrix covers 12 hierarchy levels.

### Consequences
- **Positive**: Guarantees coherent forecasts across all aggregation levels; MinT is theoretically optimal under Gaussian errors
- **Negative**: MinT requires covariance matrix inversion (O(n³) but feasible for 30K series via shrinkage); adds ~15% pipeline runtime
- **Mitigation**: Shrinkage estimator avoids singular matrix issues; sub-hierarchy support for tractability

---

## ADR-009: Conformal Prediction for Uncertainty Quantification

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted (V2-Fase 2) |

### Context
LightGBM quantile regression produces [p10, p90] intervals, but empirical coverage is not guaranteed to be exactly 80%. Coverage may be 72% or 88% depending on the series/horizon.

### Decision
Apply **split conformal calibration** post-training:
1. Compute non-conformity scores on a held-out calibration set
2. Find the (1−α)-quantile of scores: q̂
3. Adjust p10 downward and p90 upward by q̂ to achieve empirical 80% coverage

### Consequences
- **Positive**: Distribution-free coverage guarantee (80% ± finite sample error); no model retraining needed
- **Negative**: Requires a separate calibration set (reduces training data slightly); intervals may be too conservative if calibration distribution differs from test
- **Mitigation**: Calibration set is the most recent 20% of training data (temporally contiguous, distribution-similar)

---

## ADR-010: Monte Carlo Simulation for Inventory Risk Assessment

| Field | Value |
|-------|-------|
| **Date** | 2026-03-15 |
| **Status** | Accepted (V2-Fase 3) |

### Context
Deterministic inventory formulas (SS = z×σ×√LT) assume Gaussian demand and fixed lead times. Real retail demand is often non-Gaussian (intermittent, heavy-tailed) with stochastic lead times.

### Decision
Implement **vectorised Monte Carlo simulation** (1,000 simulations × 90-day horizon) using:
- Demand sampling: piecewise-linear CDF interpolated from (p10, p50, p90) conformal forecasts
- Lead times: LogNormal distribution with empirical σ
- Policy evaluation: (s,Q), (s,S), (R,S), and SL-driven policies evaluated in parallel

Performance target: < 10 ms per series (achieved: 5.8 ms single-thread).

### Consequences
- **Positive**: Accurate fill-rate distributions; captures non-Gaussian demand; enables 5 what-if scenario analyses
- **Negative**: Computationally heavier than analytic formulas (~6× for 500 series); 1,000 sims introduces sampling variance
- **Mitigation**: NumPy vectorisation (no Python loops over simulations); joblib parallelism for batch runs; seed-reproducible results
