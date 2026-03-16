# Business Case — AI Supply Chain Control Tower

**Document type:** Business justification
**Audience:** Executive sponsors, hiring managers, technical reviewers
**Last updated:** 2026-03-15

---

## 1. The Problem: Cost of Poor Demand Forecasting

Poor demand forecasting creates two symmetric costs: stockouts (lost revenue, customer switching) and overstock (excess capital tied up, markdowns, spoilage). These are not hypothetical — they are the dominant operational cost driver for retail supply chains.

### 1.1 Cost of Stockout

| Cost Component | Conservative Estimate | Source / Benchmark |
|----------------|----------------------|--------------------|
| Direct lost sale | $3–5 per unit not sold | IHL Group 2023 |
| Customer switching | 30–40% of customers buy from competitor | ECR Europe |
| Loyalty erosion | ~10% of customers do not return after 3 consecutive stockouts | Harvard Business Review |
| Emergency replenishment | 2–5× normal replenishment cost | Industry benchmark |
| Opportunity cost (overstock capital) | 15–25% of excess inventory value, annualized | APICS/ASCM |

### 1.2 Projected Financial Impact (M5 Scope)

Projected on the M5 dataset scope (10 Walmart stores, 3,049 items, 5.3 years):

```
Revenue proxy total (train period):
  Total units sold (estimated):    ~30M units
  Avg sell price (estimated):      ~$5.50
  Revenue proxy total:             ~$165M (5.3 years)
  Revenue proxy annual:            ~$31M/year

Without optimization (retail industry average):
  Stockout rate:                   6%
  Revenue at risk:                 $31M x 6% = $1.86M/year

With this system (target):
  Stockout rate:                   2.5%
  Revenue at risk:                 $31M x 2.5% = $0.78M/year

Revenue recovery delta:
  $1.86M - $0.78M = $1.08M/year (58% reduction)
  Conservative capture rate (70%): $1.08M x 0.70 = $756K/year

Overstock reduction:
  Inventory value (6-week supply):  ~$5.2M
  Excess inventory reduced:         15-20%
  Holding cost savings:             ~$130K-$195K/year

TOTAL ESTIMATED IMPACT: $886K - $951K/year (10 stores)
PER STORE:              ~$89K - $95K/year
```

> **Methodological note:** These figures are estimates based on public industry benchmarks, not actual system results. The revenue proxy will be calculated from real M5 data during pipeline execution.

---

## 2. Value of Forecast Accuracy

Forecast improvements translate directly into operational and financial outcomes:

| Forecast Improvement | Operational Impact | Financial Impact |
|---------------------|-------------------|-----------------|
| MAE reduced 10% | Safety stock decreases ~7% | Holding cost savings ~$36K/year |
| Bias reduced to <1% | Stockouts decrease ~15% | Revenue recovery ~$160K/year |
| Coverage@80 reaches 82%+ | Emergency orders decrease ~25% | Logistics savings ~$50K/year |
| Hierarchical coherence | Reliable cross-level planning | Decisional trust (not quantifiable) |

**Core principle:** a forecast has no intrinsic value. Its value emerges entirely from the quality of the inventory decisions it enables. This system is designed end-to-end with that principle: every modeling choice is justified by downstream operational impact.

---

## 3. User Personas

### Executive (Operations VP / Supply Chain Director)
- **Question:** "How is our fill rate this week? Are we at risk of stockouts before the end of the quarter?"
- **Dashboard page:** Executive Overview — 6 KPIs, revenue trend, fill rate by store
- **Decision:** Approve/reject inventory investment proposal

### Operational (Demand Planner / Category Manager)
- **Question:** "Which SKUs should I reorder this week? What if supplier lead time increases?"
- **Dashboard pages:** Forecast Lab, Inventory Risk, Scenario Lab
- **Decision:** Adjust order quantities, trigger emergency orders

### Technical (Data Scientist / ML Engineer)
- **Question:** "Is the model degrading? Which features are drifting? What's the WRMSSE on Fold 3?"
- **Dashboard pages:** Model Quality, About (architecture)
- **Tools:** MLflow UI, monitoring/health_report.json, backtesting metrics

---

## 4. Why This Approach vs Alternatives

| Approach | Limitation | This System |
|----------|-----------|-------------|
| Single-series ARIMA per SKU | Does not scale to 30K series; no cross-series learning | Global LightGBM shares patterns across series |
| Simple moving average | Ignores price, events, intermittency | 8 feature families including weather and macro |
| Uncalibrated quantiles | No coverage guarantee | Conformal post-hoc calibration → ≥80% coverage |
| No reconciliation | Level inconsistencies break planning | MinT-Shrink across 12 hierarchy levels |
| Deterministic inventory formulas | Assumes Gaussian demand | Monte Carlo with empirical demand distribution |
| Point-in-time classification | No history of ABC/XYZ changes | SCD Type 2 dim_product tracks changes over time |

---

## 5. Strategic Differentiators

This system addresses the four gaps that separate junior forecasting work from senior supply chain ML engineering:

1. **Domain-first classification:** ADI/CV² demand classification before modeling is standard practice in supply chain (APICS, Syntetos-Boylan). Ignoring it means applying regression to data where the correct output is often exactly zero.

2. **Calibrated uncertainty:** a p90 interval that is correct only 65% of the time destroys trust. Conformal calibration makes the guarantee explicit and verifiable.

3. **Hierarchical coherence:** planning at store level while forecasting at item level requires that numbers add up. MinT reconciliation is the mathematically optimal method (Wickramasuriya et al. 2019).

4. **Operational traceability:** every dashboard KPI can be traced from the metric back to the raw data file and the code that produced it. This is the standard expected in production data systems at companies like Amazon and Palantir.
