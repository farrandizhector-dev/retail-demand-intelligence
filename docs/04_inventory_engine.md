# Inventory Engine — AI Supply Chain Control Tower

**Document type:** Operations Research specification
**Audience:** Operations researchers, supply chain engineers
**Last updated:** 2026-03-15

> **SYNTHETIC:** All inventory metrics (lead times, costs, policies, stockout events) are generated synthetically from calibrated distributions over real demand data. See [docs/08_synthetic_data_disclaimer.md](08_synthetic_data_disclaimer.md).

---

## 1. Mathematical Formulations

### Safety Stock (SS)

```
SS = z(SL) x sigma_forecast x sqrt(LT)

Where:
  z(SL)       = inverse normal CDF at service level (e.g., z(0.95) = 1.645)
  sigma_forecast = standard deviation of forecast error during lead time
  LT          = mean lead time in days

Superior alternative (used in this system):
  SS = forecast_p90_during_LT - forecast_p50_during_LT
  (uses conformal intervals directly -- no Gaussian assumption)
```

### Reorder Point (ROP)

```
ROP = sum_{t=1}^{LT} y_hat_p50(t) + SS
    = expected demand during lead time + safety stock
```

### Economic Order Quantity (EOQ)

```
EOQ = sqrt(2 x D x S / H)

Where:
  D = annual demand (units)
  S = order setup cost ($)
  H = annual holding cost per unit ($)
```

### Newsvendor Model (Single-Period Optimal Quantity)

```
Q* = F^{-1}(Cu / (Cu + Co))

Where:
  Cu = underage cost (stockout cost per unit)
  Co = overage cost (holding + markdown cost per unit)
  F  = CDF of demand distribution (from conformal forecast samples)
```

### Fill Rate

```
FR = 1 - E[lost_sales] / E[demand]
```

---

## 2. Monte Carlo Simulator Design

**Motivation:** deterministic inventory formulas (SS = z×σ×√LT) assume Gaussian demand and fixed lead times. Real M5 demand is non-Gaussian (intermittent, heavy-tailed) with stochastic lead times. Monte Carlo simulation directly samples from the empirical forecast distribution.

**Simulation parameters:**
- Paths: 1,000 simulations per SKU-store
- Horizon: 90 days
- Demand sampling: piecewise-linear CDF interpolated from (p10, p50, p90) conformal forecasts
- Lead time: LogNormal distribution (mu=ln(7)±sigma depending on category)
- Performance: 5.8 ms/series single-thread (NumPy vectorized, no Python loops over simulations)

**Output per series (MonteCarloResult):**
- `fill_rate_mean`, `fill_rate_p5`, `fill_rate_p95`
- `stockout_probability`
- `daily_stock_percentiles` (5th/25th/50th/75th/95th across 1000 paths × 90 days)
- `avg_inventory_level`

**Implementation detail — vectorized demand sampling:**

```python
# src/inventory/simulator.py
# Shape: (n_paths, horizon)
# Sample demand from piecewise-linear CDF implied by (p10, p50, p90)
u = rng.uniform(size=(n_paths, horizon))
demand_samples = np.where(
    u < 0.4,
    p10 + (u / 0.4) * (p50 - p10),      # interpolate p10->p50
    p50 + ((u - 0.4) / 0.6) * (p90 - p50)  # interpolate p50->p90
)
```

This avoids a Python loop over 1,000 paths and achieves the 5.8 ms/series benchmark.

---

## 3. Four Inventory Policies Compared

| Policy | Description | Reorder Trigger | Order Quantity |
|--------|-------------|-----------------|----------------|
| **(s,Q)** | Continuous review, fixed quantity | stock <= s (reorder point) | Fixed Q = EOQ |
| **(s,S)** | Continuous review, order up to S | stock <= s | S - current_stock |
| **(R,S)** | Periodic review, order up to S | Every R days | S - current_stock |
| **SL-driven** | Service-level target drives SS | CUSUM signal or ROP | Dynamic, demand-driven |

Policy comparison runs all 4 policies through the same Monte Carlo simulation and selects the policy with minimum total cost (holding + stockout penalty).

**Cost function:**
```
total_cost = holding_cost_per_unit_day x avg_inventory_level x 90
           + stockout_cost_per_unit x expected_lost_sales
           + order_cost x num_orders
```

---

## 4. Five What-If Scenarios

| Scenario | Modification | Business Question |
|----------|-------------|-------------------|
| **Demand surge +30%** | forecast_p50 × 1.30 | How many additional stockouts in 30 days? |
| **Lead time delay +5d** | lead_time_mean += 5 days | How much additional safety stock is needed? |
| **Cost increase +15%** | holding_cost × 1.15 | How does EOQ and inventory value change? |
| **High service level 95%→99%** | service_level = 0.99 | What is the safety stock increase? |
| **Combined stress** | Demand +20% AND LT +3d | What is the worst-case fill rate? |

Each scenario uses the same random seed as baseline for comparable results. Exported to `data/gold/serving/scenario_results.json`.

---

## 5. Synthetic Data Generation Methodology

Inventory parameters are generated from calibrated distributions designed to produce realistic retail behavior:

| Parameter | Distribution | Calibration |
|-----------|-------------|-------------|
| Lead time (FOODS) | LogNormal(mu=ln(7), sigma=0.3) | ~5–9 day range, typical grocery |
| Lead time (other) | LogNormal(mu=ln(14), sigma=0.4) | ~10–18 day range, durable goods |
| Service level target | A=98%, B=95%, C=90% | ABC-driven, per APICS guidelines |
| Holding cost | 15–25% of unit cost, annualized | Industry benchmark range |
| Stockout cost multiplier | A=5×, B=3×, C=1.5× margin | ABC-driven importance weighting |
| Supplier reliability | Beta(alpha=8, beta=2) tier-1; Beta(alpha=5, beta=3) tier-2 | Calibrated for 85–99% fill rate range |

**Validation:** synthetic parameters produce fill rates between 85–99%, stockout rates between 1–8%, and inventory turns between 8–15 — all within realistic retail benchmarks.

---

## 6. Integration with Forecast Layer

The inventory engine consumes forecast outputs from the ML layer:

```
Forecast layer output (data/gold/forecasts/):
  forecast_p10, forecast_p50, forecast_p90  (per series, per day, 28-day horizon)

Inventory engine consumes:
  1. p50 -> expected demand during lead time -> ROP numerator
  2. (p90 - p50) -> uncertainty term -> safety stock
  3. (p10, p50, p90) -> piecewise-linear CDF -> Monte Carlo demand sampling

This direct integration between conformal intervals and inventory safety stock
is the key architectural link: better-calibrated intervals -> lower safety stock
-> lower holding cost while maintaining the same service level.
```

---

## 7. Output Schema

All inventory outputs include an explicit `synthetic_tag` field:

```json
{
  "item_id": "FOODS_1_001_CA_1",
  "store_id": "CA_1",
  "reorder_point": 42,
  "safety_stock": 18,
  "eoq": 120,
  "fill_rate_mean": 0.943,
  "fill_rate_p5": 0.891,
  "fill_rate_p95": 0.978,
  "stockout_probability": 0.057,
  "avg_inventory_level": 67.3,
  "recommended_policy": "sQ",
  "lead_time_mean_days": 7.2,
  "service_level_target": 0.95,
  "synthetic_tag": "SYNTHETIC",
  "simulation_seed": 42,
  "n_paths": 1000,
  "horizon_days": 90
}
```
