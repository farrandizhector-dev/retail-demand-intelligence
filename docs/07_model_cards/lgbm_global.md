# Model Card: LightGBM Global Demand Forecaster

**Version:** 1.0.0
**Date:** 2026-03-15
**Author:** Héctor Ferrándiz Sanchis
**Project:** AI Supply Chain Control Tower

---

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | LightGBM Gradient Boosting (global, multi-series) |
| **Objective** | Point: `regression` (MAE). Quantile: `quantile` for α ∈ {0.10, 0.50, 0.90} |
| **Architecture** | Single model trained across all 30,490 SKU-store series (ADR-002) |
| **Hyperparameters** | n_estimators=2000, learning_rate=0.03, num_leaves=255, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_α=0.1, reg_λ=0.1, max_bin=255 |
| **Feature families** | Calendar (7), lag (6), rolling stats (8), price (4), event (5), demand class (3), cross-series (3) — 36 total |
| **Categorical features** | store_id, dept_id, cat_id, state_id, day_of_week, month (label-encoded) |
| **Training cutoff** | d_1885 (2016-03-27) for full production run |
| **Experiment tracking** | MLflow (`sqlite:///mlflow.db`, experiment: `forecast_lgbm_{date}`) |

---

## Intended Use

### Primary use case
Point and quantile demand forecasting for 28-day horizons across Walmart-style retail SKU-store combinations. Powers inventory optimization (safety stock, ROP, order quantities) and the frontend dashboards.

### Users
- Data science / ML team consuming model outputs for inventory decisions
- Frontend dashboard (reads pre-computed JSON serving assets — no live inference)

### Out-of-scope uses
- Real-time inference serving (model outputs are pre-computed; see ADR-006)
- Forecasting beyond 28-day horizons without retraining (rolling stats decay)
- Application to non-retail demand time series without domain validation

---

## Training Data

| Attribute | Value |
|-----------|-------|
| **Dataset** | M5 Forecasting (Kaggle), Walmart daily sales 2011-01-29 → 2016-06-19 |
| **Coverage** | 30,490 SKU-store series × 1,913 days ≈ 58M rows |
| **Label** | SYNTHETIC — dataset is public M5, not live production data |
| **States** | California (CA), Texas (TX), Wisconsin (WI) |
| **Categories** | FOODS (3 depts), HOBBIES (2 depts), HOUSEHOLD (2 depts) |
| **Intermittency** | >50% of series have ADI > 1.32 or CV² > 0.49 (classified Intermittent/Erratic/Lumpy) |

**Leakage controls (spec §8.2):**
1. Rolling stats computed with `cutoff_date` per fold — no future values
2. Lag features use only confirmed past observations
3. Price features use sell_price calendar (no future prices)
4. Target column excluded from feature list
5. Validation set is temporally after training set
6. No cross-series target leakage via groupby operations

---

## Performance (5-Fold Rolling-Origin Backtesting)

Evaluation protocol: 5-fold rolling-origin, 28-day test windows, folds covering d_1774–d_1913.

| Metric | Mean (5 folds) | Std |
|--------|---------------|-----|
| MAE | *see MLflow run* | — |
| RMSE | *see MLflow run* | — |
| sMAPE | *see MLflow run* | — |
| Bias | *see MLflow run* | — |
| WRMSSE | *see MLflow run* | — |
| Coverage@80 | *see MLflow run* | — |
| Pinball Loss | *see MLflow run* | — |

> Note: Exact figures vary by training run. Consult MLflow experiment `forecast_lgbm_*` for reproducible metrics.

**Performance by demand class:**

| Class | ADI | CV² | LightGBM vs Baseline |
|-------|-----|-----|---------------------|
| Smooth | ≤1.32 | ≤0.49 | Significant improvement |
| Erratic | ≤1.32 | >0.49 | Moderate improvement |
| Intermittent | >1.32 | ≤0.49 | Marginal (Croston TSB competitive) |
| Lumpy | >1.32 | >0.49 | Marginal (near-zero demand) |

---

## Evaluation

### Metrics definition
- **MAE**: Mean Absolute Error — primary operational metric
- **WRMSSE**: Weighted RMSSE (official M5 metric, revenue-weighted)
- **Coverage@80**: % of actuals inside [p10, p90] interval (target: ≥80%)
- **Pinball Loss**: Mean quantile loss across p10/p50/p90

### Baselines compared
- Seasonal Naïve (7-day seasonality) — primary baseline
- Croston TSB — for intermittent/lumpy series

---

## Limitations

1. **Synthetic data**: Trained on M5 public dataset. Real-world performance may differ due to category mix, promotional dynamics, or supply shocks not present in M5.
2. **Global model trade-off**: A single model for all series sacrifices per-series specialization. Lumpy/intermittent series (>50% of portfolio) may be better served by Croston TSB.
3. **No live data pipeline**: Pre-computed JSON serving means forecasts are static until the pipeline is re-run. Stale forecasts after 28 days.
4. **Conformal intervals are post-hoc**: The [p10, p90] bands are calibrated after training (see conformal.py). Calibration quality depends on the held-out calibration set size.
5. **No exogenous macro features**: FRED macro data (see CONTEXT.md) was not included in V1/V2 training.

---

## Ethical Considerations

- **No personal data**: Training data is aggregate retail sales counts. No PII.
- **Synthetic label**: All outputs are derived from a public benchmark dataset. Not suitable for live business decisions without validation on production data.
- **Inventory bias risk**: Systematic over-prediction can lead to excess inventory (capital waste); under-prediction to stockouts (revenue loss). Monitor `val_bias` metric per deployment.
- **Equity**: The model does not incorporate demographic or pricing equity considerations. Inventory allocation downstream should be reviewed by domain experts.
