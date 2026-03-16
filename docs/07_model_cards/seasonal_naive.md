# Model Card: Seasonal Naïve Baseline

**Version:** 1.0.0
**Date:** 2026-03-15
**Author:** Héctor Ferrándiz Sanchis
**Project:** AI Supply Chain Control Tower

---

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | Seasonal Naïve (rule-based, no learning) |
| **Seasonality** | 7-day (weekly cycle) |
| **Formula** | ŷ_{t+h} = y_{t+h−7⌈h/7⌉} (repeat last observed same weekday) |
| **Parameters** | None (non-parametric) |
| **Role** | Primary baseline for all demand classes |
| **Implementation** | `statsforecast.SeasonalNaive(season_length=7)` |

---

## Intended Use

### Primary use case
Establish a lower-bound performance floor for evaluation of ML models. If LightGBM does not outperform Seasonal Naïve on a given demand class, the ML model provides no value for that segment.

### Users
- Evaluation pipeline (backtesting.py) — automatic comparison
- Model quality dashboard page (ModelQuality.tsx)

### Out-of-scope uses
- Production forecasting without ML comparison (too simple for FOODS/HOBBIES)
- Series with seasonal periods other than 7 days (daily data only)

---

## Training Data

No training — rule-based method. Requires only the last 7+ days of observed history per series.

---

## Performance

Seasonal Naïve sets the WRMSSE baseline. LightGBM global model is expected to achieve 20-40% WRMSSE improvement on Smooth and Erratic demand classes. For Intermittent/Lumpy classes, Seasonal Naïve is often competitive or superior.

---

## Limitations

1. Cannot capture trend, promotions, or holiday effects.
2. Fails on new/sparse series with <7 days of history.
3. Produces negative forecasts if historical values were negative (data quality issue).
4. Horizon > 7 days simply repeats the same 7-day cycle without uncertainty widening.

---

## Ethical Considerations

No learning component — no bias from training data. Performance is entirely determined by historical demand patterns.
