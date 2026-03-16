# Hierarchical Reconciliation — AI Supply Chain Control Tower

**Document type:** ML methodology
**Audience:** Data scientists, demand planners
**Last updated:** 2026-03-15

---

## 1. The Problem: Incoherent Forecasts

Without reconciliation, independently trained forecasts at different hierarchy levels are incoherent: the sum of CA store forecasts does not equal the CA state forecast. This breaks planning — a demand planner cannot trust numbers that don't add up.

**Example incoherence:**
```
SKU-store level (sum of 4 CA stores):  CA_1=45, CA_2=38, CA_3=41, CA_4=52 -> sum=176
State CA forecast (separately trained): 155
Gap: 21 units (13.5%) -> which level do you trust for reorder decisions?
```

This gap is not random noise — it is a systematic property of independently trained models. Reconciliation eliminates it by construction.

---

## 2. M5 Hierarchy: 12 Levels, 42,840 Series

```
                                    Total (1)
                                       |
                    +------------------+------------------+
                    |                  |                  |
                  CA (1)             TX (1)             WI (1)          Level 2: State (3)
                    |                  |                  |
            +---+---+---+      +---+---+---+      +---+---+---+
            |   |   |   |      |   |   |   |      |   |   |   |
          CA_1 CA_2 CA_3 CA_4 TX_1 TX_2 TX_3   WI_1 WI_2 WI_3        Level 3: Store (10)
```

| Level | Aggregation | # Series |
|-------|-------------|---------|
| 1 | Total | 1 |
| 2 | State | 3 |
| 3 | Store | 10 |
| 4 | Category | 3 |
| 5 | Department | 7 |
| 6 | State × Category | 9 |
| 7 | State × Department | 21 |
| 8 | Store × Category | 30 |
| 9 | Store × Department | 70 |
| 10 | Product | 3,049 |
| 11 | Product × State | 9,147 |
| 12 | SKU-Store (bottom) | 30,490 |
| **Total** | | **42,840** |

---

## 3. Reconciliation Methods

| Method | Mechanism | Optimality | Scalability |
|--------|-----------|-----------|------------|
| **Bottom-Up (BU)** | Forecast SKU-store level, aggregate upward | Optimal when bottom level has best accuracy | O(n) — trivially scalable |
| **Top-Down (TD)** | Forecast total, disaggregate by historical proportions | Optimal when top level is most reliable | O(n) — fast |
| **MinT-Shrink** | Solve constrained optimization: minimize sum of reconciliation errors, shrinkage estimator for covariance matrix | Theoretically optimal under Gaussian errors (Wickramasuriya et al. 2019) | O(n^2) with shrinkage — feasible for 30K series |

**Selection criterion:** the method with lowest revenue-weighted MAE on the backtesting validation fold is selected for production.

---

## 4. MinT-Shrink: Mathematical Detail

MinT (Minimum Trace) reconciliation solves:

```
Minimize: trace(Var(y_hat_reconciled - y_true))
Subject to: S' y_hat_reconciled = y_hat_base  (coherence constraint)

Solution:
  y_hat_r = S (S' W^{-1} S)^{-1} S' W^{-1} y_hat_base

Where:
  S     = summation matrix (maps bottom-level to all levels)
  W     = covariance matrix of base forecast errors
  y_hat_base = base forecasts (from LightGBM, all levels)
```

**Shrinkage estimator for W:**

Direct estimation of W (30K × 30K) is infeasible. The shrinkage estimator approximates:

```
W_shrink = (1 - lambda) * W_sample + lambda * diag(W_sample)

Where lambda is chosen by Ledoit-Wolf to minimize expected error.
This reduces the effective problem to O(n) per level.
```

---

## 5. S Matrix Structure

The S (summation) matrix maps bottom-level series (rows = 30,490) to all aggregation levels (columns = 42,840). It is a boolean/integer matrix:

```
S[i, j] = 1 if bottom-level series j contributes to aggregate i
S[i, j] = 0 otherwise
```

For implementation, S is stored as a sparse matrix (`scipy.sparse.csr_matrix`) since it has density ~0.02% for the full hierarchy.

---

## 6. Mandatory Coherence Tests

Four checks run after every reconciliation (in `src/reconciliation/evaluate_reconciliation.py`):

```python
# Test 1: Bottom-level sums match upper levels
assert np.allclose(aggregate_bottom(level), upper_level_forecast, atol=0.01)

# Test 2: No negative forecasts post-reconciliation
assert (reconciled_forecasts["forecast_p50"] >= 0).all()

# Test 3: Quantile ordering preserved
assert (p10 <= p50).all() and (p50 <= p90).all()

# Test 4: Total coherence across all dates
for each date:
    assert abs(sum_of_bottom_level - total_level_forecast) < 1.0
```

If any check fails, a `RECONCILIATION_INCOHERENCE` alert (severity: CRITICAL) is raised and the serving export is blocked.

---

## 7. Sub-Hierarchy Decomposition

For tractability, reconciliation is applied independently per store (reducing the problem from 30K series to ~3K series per sub-run) before aggregating to higher levels:

```
For each store s in {CA_1, CA_2, CA_3, CA_4, TX_1, TX_2, TX_3, WI_1, WI_2, WI_3}:
  1. Build sub-hierarchy S_s for store s (3,049 items × 5 hierarchy levels within-store)
  2. Run MinT-Shrink on S_s
  3. Store reconciled bottom-level for store s

Aggregate:
  4. Sum across stores to produce state-level, total-level coherent forecasts
  5. Run coherence checks on full hierarchy
```

This reduces peak memory from O(30K²) to O(3K²) × 10 sequential runs.

---

## 8. Implementation

`src/reconciliation/reconciler.py` wraps `hierarchicalforecast` (Nixtla):

```python
from hierarchicalforecast.methods import MinTrace

reconciler = MinTrace(method='mint_shrink')
reconciled = reconciler.fit(S, base_forecasts).reconcile(base_forecasts)
```

The `hierarchicalforecast` library (by Nixtla) implements MinT-Shrink with sparse S matrices and is tested on the full M5 hierarchy. It is the reference implementation for the M5 competition evaluation methodology.
