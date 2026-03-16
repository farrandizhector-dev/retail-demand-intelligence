# Synthetic Data Disclaimer — AI Supply Chain Control Tower

**IMPORTANT: Read before interpreting any inventory or operational metrics.**

---

## What is REAL vs SYNTHETIC

### REAL Data (from public sources)

| Data | Source | Usage |
|------|--------|-------|
| Daily sales demand | M5 Forecasting (Kaggle/Walmart) | All forecasting and classification |
| Weekly sell prices | M5 Forecasting (Kaggle/Walmart) | Price features, revenue proxy |
| Calendar, events, SNAP | M5 Forecasting (Kaggle/Walmart) | Calendar features, event modeling |
| Daily weather (CA, TX, WI) | Open-Meteo Historical API | Weather feature family |
| Macroeconomic indicators | FRED API (Federal Reserve) | Macro feature family (optional) |

### SYNTHETIC Data (generated programmatically)

| Data | Generation Method | Label in System |
|------|------------------|----------------|
| Lead times (days) | LogNormal distribution by category | `SYNTHETIC` in all outputs |
| Holding costs | Uniform 15–25% of unit cost | `SYNTHETIC` |
| Order setup costs | Uniform $10–$50 per order | `SYNTHETIC` |
| Safety stock calculations | Derived from synthetic params | `SYNTHETIC` |
| Monte Carlo inventory paths | Sampled from forecast distribution | `SYNTHETIC` |
| Policy comparison results | Computed over synthetic params | `SYNTHETIC` |
| Stockout events | Simulated from MC paths | `SYNTHETIC` |
| Purchase order data | Generated with synthetic lead times | `SYNTHETIC` |

---

## Why Synthetic?

The M5 dataset is a forecasting competition dataset. It contains demand, prices, and calendar data — but not operational parameters (lead times, costs, supplier data). These parameters are proprietary to Walmart and not publicly available.

Generating synthetic operational parameters over real demand data is exactly the methodology used by supply chain consultants building PoC systems for retailers: the demand signal is real, the operational layer is simulated to demonstrate the analytical framework.

**This is not a limitation — it is a documented design choice (ADR-004).**

---

## How Synthetic Data is Generated

```python
# src/inventory/synthetic_params.py
# All generation uses fixed seeds for reproducibility

lead_time_days:
  FOODS:     np.random.lognormal(mean=np.log(7),  sigma=0.3, seed=42)
  HOBBIES:   np.random.lognormal(mean=np.log(14), sigma=0.4, seed=42)
  HOUSEHOLD: np.random.lognormal(mean=np.log(14), sigma=0.4, seed=42)

service_level_target:
  A class items: 0.98
  B class items: 0.95
  C class items: 0.90

holding_cost_pct (annualized):
  FOODS:     Uniform(0.15, 0.20)  # perishable
  HOUSEHOLD: Uniform(0.18, 0.25)  # durable
  HOBBIES:   Uniform(0.20, 0.30)  # seasonal

stockout_cost_multiplier (x unit margin):
  A class items: 5.0x   # high-value, high-visibility
  B class items: 3.0x
  C class items: 1.5x
```

---

## Where Synthetic Data Appears

Every synthetic-derived output is labeled in the codebase and frontend:

1. **Code:** functions in `src/inventory/` include `# SYNTHETIC` comments and return dataframes with `synthetic_tag="SYNTHETIC"` column
2. **JSON serving assets:** all inventory JSONs include `"synthetic_tag": "SYNTHETIC"` field
3. **Frontend:** `EmptyState` component displays "SYNTHETIC DATA" badge; About page has prominent disclaimer
4. **Model cards:** `docs/07_model_cards/lgbm_global.md` explicitly notes synthetic inventory layer

---

## Validity of This Approach

A PoC system built with synthetic operational parameters is valid for:

- Demonstrating the analytical framework and methodology
- Showing the full end-to-end pipeline architecture
- Evaluating forecasting model quality (demand signal is real)
- Testing inventory policies under realistic demand distributions
- Portfolio demonstration of supply chain optimization skills

It is NOT valid for:

- Making actual business inventory decisions without real operational data
- Claiming specific ROI figures as actual outcomes
- Representing as a production-ready system without real parameter validation

---

## Reproducibility

All synthetic data generation is fully reproducible:

- Fixed random seed (`seed=42`) in `configs/inventory.yaml`
- Seed is logged as an MLflow parameter in every inventory run
- `synthetic_tag`, `simulation_seed`, and `n_paths` are included in all JSON outputs
- Running `make generate_inventory` with the same config and forecast inputs produces bit-identical outputs

This means that any reviewer can verify the synthetic generation independently by re-running the pipeline.
