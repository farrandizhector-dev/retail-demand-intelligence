# Model Card: Croston TSB (Intermittent Demand Baseline)

**Version:** 1.0.0
**Date:** 2026-03-15
**Author:** Héctor Ferrándiz Sanchis
**Project:** AI Supply Chain Control Tower

---

## Model Details

| Field | Value |
|-------|-------|
| **Model type** | Croston TSB (Teunter-Syntetos-Babai variant) |
| **Category** | Exponential smoothing for intermittent demand |
| **Parameters** | α (demand smoothing), β (interval smoothing) — fitted per series |
| **Formula** | p̂_t = (1−β)p̂_{t−1} + β·(1/q_t); ẑ_t = (1−α)ẑ_{t−1} + α·d_t; ŷ = ẑ·p̂ |
| **Role** | Specialist baseline for Intermittent and Lumpy demand classes |
| **Implementation** | `statsforecast.CrostonTSB(alpha_d=0.1, alpha_p=0.1)` |

---

## Intended Use

### Primary use case
Produce bias-corrected forecasts for SKUs with intermittent demand (ADI > 1.32). The original Croston method has a positive bias; TSB corrects this by smoothing the non-zero demand probability.

### Users
- Evaluation pipeline — compared against LightGBM for Intermittent/Lumpy segments
- Can be used as production forecaster for deep-intermittent SKUs where LightGBM underperforms

### Out-of-scope uses
- Smooth or Erratic demand (Seasonal Naïve / LightGBM dominate these classes)
- Multi-step probabilistic intervals (produces only point forecasts by default)

---

## Training Data

Per-series exponential smoothing — no cross-series learning. Requires a minimum of 10 non-zero observations for reliable parameter estimation.

---

## Performance

Croston TSB is competitive with LightGBM on Intermittent (ADI > 1.32, CV² ≤ 0.49) and Lumpy (ADI > 1.32, CV² > 0.49) demand classes. For these segments, which represent >50% of M5 series, Croston TSB may produce lower MAE due to its explicit intermittency modelling.

---

## Limitations

1. **Point forecast only**: No native uncertainty quantification. Prediction intervals require post-hoc conformal calibration.
2. **Stationary assumption**: Assumes demand probability and non-zero demand size are stationary. Poor fit for trending or seasonal intermittent series.
3. **Parameter sensitivity**: α and β must be fitted per series; poor fit on short or very sparse series.
4. **No cross-series information**: Cannot leverage shared patterns (price promotions, holidays) across similar SKUs.

---

## Ethical Considerations

No training data bias — operates only on each series' own history. Stocking decisions based on Croston TSB forecasts for low-velocity SKUs should be reviewed for inventory equity (avoiding systematic under-stocking of specialty/diverse SKUs).
