# Known Issues — AI Supply Chain Control Tower

**Document type:** Issue log
**Last updated:** 2026-03-15

---

## Active Issues

### KI-001: ECharts Bundle Size Warning in Vite Build

**Severity:** Low (cosmetic warning, not a build failure)

**Symptom:**
```
(!) Some chunks are larger than 500 kB after minification.
  dist/assets/index-[hash].js (628.17 kB)
```

**Root cause:** Apache ECharts is a large library (~628 KB minified). This is expected and documented behavior. The chunk size warning is a Vite default threshold, not a performance problem.

**Status:** Accepted. ECharts was chosen over Recharts/Victory for its superior performance on dense dashboards (ADR-007). The bundle is served via HF Space CDN with gzip compression, reducing effective transfer to ~180 KB.

**Mitigation:** Tree-shaking is already applied (`echarts/core` with explicit imports). Further reduction would require switching to a lighter charting library, which is not planned for the current scope.

---

### KI-002: FRED API Key Required for Macro Features

**Severity:** Low (macro features are optional)

**Symptom:** `make ingest` completes without macro data if `FRED_API_KEY` is not set.

**Root cause:** FRED API requires a free API key. The key is not included in the repository.

**Workaround:** The pipeline runs without macro features. The feature store falls back to `macro_available=False` in `configs/features.yaml`. Macro features (CPI, unemployment, consumer sentiment) are logged as missing in MLflow.

**Resolution:** Register at https://fred.stlouisfed.org/docs/api/api_key.html (free, instant). Set `FRED_API_KEY` in `.env`.

---

### KI-003: Pandera LazyFrame `membership` PerformanceWarning

**Severity:** Low (warning only, no functional impact)

**Symptom:**
```
PerformanceWarning: Checking the membership of a column with a large number of
unique values can be slow.
```

**Root cause:** Pandera internals perform membership checks differently on Polars LazyFrames vs DataFrames. This is a known Pandera upstream issue.

**Status:** Not actionable on our side. The warning appears in test output but does not affect validation results. Tracking: upstream Pandera issue tracker.

---

### KI-004: Full Pipeline Execution Time

**Severity:** Informational

**Details:**

| Step | Estimated Time (full 30K series) |
|------|----------------------------------|
| `make ingest` | 10–30 min (download-dependent) |
| `make transform_silver` | 2–5 min (Polars vectorized) |
| `make classify_demand` | 1–2 min |
| `make build_features` | 5–15 min |
| `make train_lgbm` (full) | 15–45 min |
| `make evaluate` (5 folds) | 60–120 min (full 30K series) |
| `make generate_inventory` | 10–30 min (1000 sims × 500 series) |
| `make export_serving` | 1–2 min |

**Development mode:** use `--n_sample 500` flag in training/backtesting for 5–10× speedup.

---

### KI-005: HF Space Deployment Requires Manual HF_USER Configuration

**Severity:** Low

**Symptom:** `.github/workflows/deploy_space.yml` contains placeholder `HF_USER` string.

**Resolution:** Replace `HF_USER` with your actual Hugging Face username in `deploy_space.yml`. Add `HF_TOKEN` as a GitHub repository secret. See [docs/09_deployment_guide.md](09_deployment_guide.md) for full instructions.

---

### KI-006: MinT-Shrink Memory Usage at Full Scale

**Severity:** Informational

**Details:** Running MinT-Shrink on the full 42,840-series hierarchy requires approximately 8–12 GB of RAM due to the covariance matrix estimation. The sub-hierarchy decomposition (per store) reduces this to ~2 GB peak, but may be noticeable on machines with less than 8 GB available.

**Workaround:** The `configs/reconciliation.yaml` setting `method: bottom_up` can be used to skip MinT-Shrink if memory is constrained. Bottom-Up is computationally trivial and typically within 5% of MinT-Shrink accuracy.

---

## Resolved Issues

*(No resolved issues yet — project at initial release.)*
