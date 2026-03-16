# Demand Classification Framework — AI Supply Chain Control Tower

**Document type:** Methodology
**Audience:** Data scientists, supply chain analysts
**Last updated:** 2026-03-15

---

## 1. Why Classify Before Modeling?

The M5 dataset is dominated by intermittent and lumpy demand: over 50% of SKU-store series have more than 50% zero-sale days. A regression model trained on all series together will underfit the zero-heavy series (predicting small positive values where zero is correct) and overfit the smooth series (overly influenced by lumpy demand outliers).

Classification before modeling is standard practice in supply chain forecasting (it appears in all top-10 M5 solutions) and is missing from >99% of data science portfolio projects. It is the first signal a senior hiring manager looks for.

---

## 2. ADI/CV² Framework (Syntetos-Boylan-Croston Matrix)

### Metrics

**ADI — Average Demand Interval:**

```
ADI = total_periods / non_zero_periods
    = 1 / (fraction of days with non-zero sales)

Interpretation: average number of days between non-zero demand events.
ADI = 1.0  -> demand every day (perfectly continuous)
ADI = 7.0  -> demand once per week on average
ADI = 14.0 -> demand once per two weeks
```

**CV² — Squared Coefficient of Variation:**

```
CV2 = (std of non-zero demand values)^2 / (mean of non-zero demand values)^2
    = Var(demand | demand > 0) / E[demand | demand > 0]^2

Interpretation: how variable the non-zero demand sizes are.
CV2 = 0.0 -> all non-zero demands are identical (perfectly regular)
CV2 = 1.0 -> standard deviation equals mean (Poisson-like)
CV2 = 2.0 -> very variable demand sizes
```

### Classification Matrix

| | CV² < 0.49 | CV² >= 0.49 |
|---|---|---|
| **ADI < 1.32** | **SMOOTH** — frequent, regular | **ERRATIC** — frequent, irregular sizes |
| **ADI >= 1.32** | **INTERMITTENT** — rare, regular sizes | **LUMPY** — rare, irregular sizes |

Thresholds from Syntetos & Boylan (2005), validated on M5.

### Implementation

```python
# src/classification/demand_classifier.py

def classify_series(sales_series: np.ndarray) -> DemandClass:
    n = len(sales_series)
    non_zero = sales_series[sales_series > 0]

    if len(non_zero) == 0:
        return DemandClass.LUMPY  # all-zero series -> most conservative

    adi = n / len(non_zero)
    cv2 = (non_zero.std() / non_zero.mean()) ** 2 if non_zero.mean() > 0 else 0.0

    if adi < 1.32 and cv2 < 0.49:
        return DemandClass.SMOOTH
    elif adi < 1.32 and cv2 >= 0.49:
        return DemandClass.ERRATIC
    elif adi >= 1.32 and cv2 < 0.49:
        return DemandClass.INTERMITTENT
    else:
        return DemandClass.LUMPY
```

---

## 3. ABC/XYZ Classification

Runs in parallel with ADI/CV² to provide revenue-based and variability-based segmentation:

**ABC (by revenue contribution):**
- **A:** top 20% of cumulative revenue → 98% service level target
- **B:** next 30% → 95% service level target
- **C:** remaining 50% → 90% service level target

Cumulative revenue is computed as `sum(units_sold × sell_price)` over the full training period per SKU-store series.

**XYZ (by demand variability, coefficient of variation):**
- **X:** CV < 0.5 → regular demand (easy to forecast)
- **Y:** 0.5 <= CV < 1.0 → moderate variability
- **Z:** CV >= 1.0 → highly variable demand (hard to forecast)

CV here uses all periods (including zeros), unlike the CV² in ADI/CV² which conditions on non-zero periods.

**Combined ABC/XYZ matrix** drives service level and safety stock policy:

| | X (CV < 0.5) | Y (0.5 <= CV < 1.0) | Z (CV >= 1.0) |
|---|---|---|---|
| **A** | AX: 98% SL, tight | AY: 98% SL, moderate buffer | AZ: 98% SL, large buffer |
| **B** | BX: 95% SL, tight | BY: 95% SL, standard | BZ: 95% SL, buffer |
| **C** | CX: 90% SL, lean | CY: 90% SL, lean | CZ: 90% SL, minimal |

---

## 4. Impact on Pipeline Stages

| Stage | Smooth | Erratic | Intermittent | Lumpy |
|-------|--------|---------|-------------|-------|
| **Primary baseline** | Seasonal Naive | Seasonal Naive | Croston TSB | Croston TSB |
| **Feature emphasis** | Lags, rolling stats | Volatility features | Zero-inflation features | Zero-inflation + burst |
| **Evaluation metric** | MAE, RMSE | RMSE weighted | Scaled Pinball Loss | Scaled Pinball Loss |
| **Safety stock** | Standard z×sigma×sqrt(LT) | Safety buffer | Demand-driven | Conservative |
| **Service level** | As per ABC | As per ABC | Reduced (sparse data) | Very conservative |

---

## 5. M5 Dataset Distribution (Approximate)

Based on ADI/CV² analysis of the 30,490 SKU-store series:

| Class | Approximate % | Count |
|-------|--------------|-------|
| Smooth | ~25% | ~7,600 |
| Erratic | ~20% | ~6,100 |
| Intermittent | ~30% | ~9,150 |
| Lumpy | ~25% | ~7,640 |

> Exact distribution computed at runtime by `src/classification/demand_classifier.py`. >50% of series have ADI > 1.32 (consistent with published M5 findings).

---

## 6. SCD Type 2: Classification History

Since ABC/XYZ/demand_class are recomputed every 90 days (rolling), they change over time. An item that was Class C in 2012 may become Class B by 2015 if sales grow. Tracking this change is critical for historical analysis ("why did fill rate improve in Q3 2014?").

`dim_product` uses SCD Type 2 (`src/classification/scd_manager.py`):

| Event | Action |
|-------|--------|
| No change | Current row unchanged |
| Classification change | Old row closed: `valid_to = effective_date - 1`, `is_current = False`; New row inserted: `version += 1`, `valid_from = effective_date`, `valid_to = 9999-12-31` |
| New item | Insert with `version = 1`, `valid_from = first_sale_date`, `valid_to = 9999-12-31` |

Surrogate key `item_key` auto-increments on each new version. Natural key is `(item_id, store_id)`.

**Example dim_product history:**

```
item_key | item_id              | store | abc | xyz | demand_class | valid_from | valid_to   | is_current
---------|----------------------|-------|-----|-----|-------------|------------|------------|----------
1        | FOODS_1_001_CA_1     | CA_1  | C   | Z   | LUMPY       | 2011-01-29 | 2013-06-14 | False
2        | FOODS_1_001_CA_1     | CA_1  | C   | Y   | INTERMITTENT| 2013-06-15 | 2015-01-10 | False
3        | FOODS_1_001_CA_1     | CA_1  | B   | Y   | SMOOTH      | 2015-01-11 | 9999-12-31 | True
```

---

## 7. Classification Recomputation Schedule

| Trigger | Frequency | Notes |
|---------|-----------|-------|
| Initial load | Once at pipeline setup | All 30,490 series |
| Rolling recompute | Every 90 days | Uses last 365 days of data |
| Manual override | As needed | Allows analyst to fix misclassifications |
| SCD Type 2 write | On any change | `scd_manager.apply_scd_type2()` |

The 365-day lookback window is long enough to capture seasonal patterns while remaining responsive to genuine demand regime changes.
