# ML System Design — AI Supply Chain Control Tower

**Document type:** ML technical specification
**Audience:** ML engineers, data scientists, technical reviewers
**Last updated:** 2026-03-15

---

## 1. Problem Framing

**Task:** Multi-step hierarchical demand forecasting
**Horizon:** 28 days ahead
**Granularity:** Daily, per SKU-store combination
**Series count:** 30,490 SKU-store series (bottom level)
**Hierarchy levels:** 12 (from individual SKU-store to total)
**Output:** Point forecast (p50) + uncertainty intervals (p10, p90), reconciled across hierarchy

**Key constraints:**
- >50% of observations are zero (intermittent demand) — naive regression fails
- No future data leakage — strict rolling-origin validation
- Must scale to 30K+ series — local ARIMA is infeasible
- Forecasts must be coherent (sum from bottom = top) — reconciliation required

---

## 2. Demand Classification (ADI/CV²) — Why It Matters

Before training any model, every SKU-store series is classified into one of four demand types using the Syntetos-Boylan-Croston (SBC) matrix:

```
                    CV2 (squared coefficient of variation)
                    Low (< 0.49)         High (>= 0.49)
ADI (avg demand  +---------------------+---------------------+
interval)        |     SMOOTH          |     ERRATIC         |
                 |   ADI < 1.32        |   ADI < 1.32        |
Low (< 1.32)     |   CV2 < 0.49        |   CV2 >= 0.49       |
                 +---------------------+---------------------+
                 |     INTERMITTENT    |     LUMPY           |
                 |   ADI >= 1.32       |   ADI >= 1.32       |
High (>= 1.32)   |   CV2 < 0.49        |   CV2 >= 0.49       |
                 +---------------------+---------------------+
```

**Why this matters:** applying a single regression model to all 4 demand types produces poor results. Lumpy demand (near-zero, burst-based) cannot be modeled like smooth demand. Classification before modeling is standard practice in supply chain forecasting (M5 top solutions all used this approach) and is missing from >99% of portfolio projects.

**Pipeline implications:**

| Demand Class | Primary Baseline | Feature Emphasis | Inventory Policy |
|-------------|-----------------|-----------------|-----------------|
| Smooth | Seasonal Naive | Lags, rolling stats | EOQ, standard SS |
| Erratic | Seasonal Naive | Volatility features | Safety buffer |
| Intermittent | Croston TSB | Zero-inflation features | Demand-driven SS |
| Lumpy | Croston TSB | Zero-inflation, burst features | Conservative |

---

## 3. Feature Engineering: 8 Families

Total: ~85 features engineered from 5 data sources.

| Family | Features | Key Design Decisions |
|--------|----------|---------------------|
| **Lags** | lag_1, lag_7, lag_14, lag_21, lag_28, lag_56, lag_91, lag_364 | lag_1 only safe for h=1; lag_7 is the safe minimum for h>1 |
| **Rolling Stats** | rolling_mean/std/min/max/q25/q75_{7,14,28,56,91} | All windows end at cutoff_date, never overlap with test |
| **Calendar** | dow, dom, woy, month, quarter, is_weekend, event distances, SNAP | Events encoded as distance-to/from, never binary future flags |
| **Price** | sell_price, price_delta_wow, ratio_vs_rolling_28, is_promo_proxy | Forward-filled from weekly; no future prices |
| **Intermittency** | pct_zero_last_28/91d, days_since_last_sale, burstiness, streak_zeros | Critical for 50%+ zero series in M5 |
| **Weather** | temp_max/min/mean, precipitation, humidity, weathercode, temp_anomaly | State-level aggregation aligned to store geography |
| **Macro** | CPI YoY, unemployment, consumer sentiment, oil price, retail sales | All with publication lag (45-day safety margin) |
| **Interactions** | target_enc_item/store/dept, price×weekend, SNAP×dept, temp×cat | Leave-one-out target encoding to avoid leakage |

---

## 4. Leakage Control Protocol (6 Rules)

Leakage is the most common failure mode in time-series ML. This system enforces 6 explicit rules:

| Rule | Description | Check |
|------|-------------|-------|
| No future prices | sell_price features use only price up to t-1 inclusive | `assert max(price_date) < forecast_start_date` |
| No future calendar info | Events only as distance features, not binary future flags | Event features must be distance-based |
| No future weather | Weather features use only data up to t-1 | `assert max(weather_date) < forecast_start_date` |
| No future target in rolling stats | Rolling windows computed before forecast origin | `rolling_window_end <= cutoff_date` per fold |
| No target encoding on test | Target encoder fitted only on train of each fold | `encoder.fit()` only on train indices |
| Macro publication lag | CPI/UNRATE published ~1 month late; use publication_date - 45d | Forward-fill with explicit lag |

Tests: `tests/unit/test_leakage_guard.py` enforces all 6 rules automatically.

---

## 5. Model Stack

```
Baselines (statsforecast)
  ├── Seasonal Naive (7-day seasonality) -- universal baseline
  ├── Moving Average 28d -- smooth demand baseline
  ├── Croston Classic -- intermittent demand
  └── TSB (Teunter-Syntetos-Babai) -- lumpy demand, bias-corrected

LightGBM Global (src/models/training.py)
  ├── Point model: objective=regression, metric=MAE
  └── Quantile models: alpha in {0.10, 0.50, 0.90}

Conformal Calibration (src/models/conformal.py)
  └── Post-hoc split conformal -> guaranteed >=80% coverage

Hierarchical Reconciliation (src/reconciliation/)
  ├── Bottom-Up (BU) -- forecast bottom, aggregate
  ├── Top-Down (TD) -- forecast top, disaggregate by historical proportions
  └── MinT-Shrink -- minimize covariance trace (selected if best MAE in backtesting)
```

---

## 6. Validation: 5-Fold Rolling-Origin Backtesting

```
Fold 1: Train [d_1 -> d_1773]    Test [d_1774 -> d_1801]  (28 days)
Fold 2: Train [d_1 -> d_1801]    Test [d_1802 -> d_1829]
Fold 3: Train [d_1 -> d_1829]    Test [d_1830 -> d_1857]
Fold 4: Train [d_1 -> d_1857]    Test [d_1858 -> d_1885]
Fold 5: Train [d_1 -> d_1885]    Test [d_1886 -> d_1913]
```

- No random shuffling — temporal integrity preserved
- Features rebuilt per fold with fold's cutoff_date (no future leakage)
- Evaluation metrics: MAE, RMSE, WRMSSE, sMAPE, Bias, Coverage@80, Pinball Loss
- Segmentation: by category, department, store, state, demand_class, ABC, XYZ

All results logged to MLflow experiment `forecast_lgbm_{date}`.

---

## 7. Hierarchical Reconciliation

After base forecasts, reconciliation ensures that predictions are coherent across the 12-level M5 hierarchy. Without reconciliation, the sum of SKU-store forecasts does not equal the total-level forecast, making planning unreliable.

| Method | Mechanism | When to Use |
|--------|-----------|-------------|
| Bottom-Up (BU) | Forecast bottom level, aggregate upward | Bottom level is most accurate |
| Top-Down (TD) | Forecast total, disaggregate proportionally | Top level is most accurate |
| MinT-Shrink | Minimize covariance trace of reconciliation errors | General purpose (optimal under Gaussian errors) |

Selection criterion: the method with lowest MAE × revenue-weight on the validation fold.

4 mandatory coherence checks run after every reconciliation:
1. Bottom-level sums match upper-level aggregations (within 0.01 tolerance)
2. No negative forecasts post-reconciliation
3. Quantile ordering preserved: p10 <= p50 <= p90
4. Total coherence across all dates

---

## 8. MLflow Experiment Tracking

Every training run logs to MLflow:

| Log Type | What is Logged |
|----------|---------------|
| Parameters | n_estimators, num_leaves, learning_rate, min_child_samples, all feature flags, fold number |
| Metrics | MAE, RMSE, WRMSSE, sMAPE, Bias, Coverage@80, Pinball Loss — per fold and aggregate |
| Artifacts | Trained model (`.txt`), feature importance JSON, SHAP values parquet, backtest metrics JSON |
| Tags | model_version, demand_class_split, reconciliation_method, dataset_sha256 |

To view results: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000

---

## 9. SHAP Feature Importance

Post-training, `src/models/shap_analysis.py` computes SHAP values using TreeExplainer (fast, exact for tree models):

- Top-30 features by mean absolute SHAP value
- Segmented by demand class (smooth/erratic/intermittent/lumpy)
- Results exported to `data/gold/serving/shap_summary.json` for the Model Quality dashboard page
- Typical top features: lag_7, lag_14, rolling_mean_28, sell_price, pct_zero_last_28d, days_since_last_sale, event_distance_snap

SHAP is used for model transparency (dashboard) and for feature selection in future iterations (if a feature has near-zero SHAP across all segments, it is a candidate for removal).
