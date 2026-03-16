# Monitoring & Operations Runbook — AI Supply Chain Control Tower

**Document type:** Operational runbook
**Audience:** On-call engineers, data scientists
**Last updated:** 2026-03-15

---

## 1. Weekly Monitoring Checklist

```bash
# 1. Check health report
cat monitoring/health_report.json | python -m json.tool

# 2. Check drift indicators (KS test + PSI)
python -c "
import json
with open('monitoring/health_report.json') as f:
    report = json.load(f)
print('KS drift alerts:', report.get('ks_drift_alerts', []))
print('PSI drift alerts:', report.get('psi_drift_alerts', []))
print('CUSUM alerts:', report.get('cusum_alerts', []))
"

# 3. Review active alerts
python -c "
import json
with open('monitoring/health_report.json') as f:
    report = json.load(f)
for alert in report.get('active_alerts', []):
    print(f\"{alert['severity']}: {alert['alert_type']} - {alert['message']}\")
"

# 4. Verify serving assets
pytest tests/e2e/test_frontend_smoke.py -v
```

---

## 2. Alert Reference

| Severity | Alert Type | Trigger Condition | Default Action |
|----------|-----------|-------------------|---------------|
| CRITICAL | `DATA_QUALITY_FAILURE` | Pandera contract violation at any layer | Block pipeline, quarantine bad data |
| CRITICAL | `RECONCILIATION_INCOHERENCE` | Any of 4 coherence checks fails | Block serving export |
| HIGH | `FORECAST_DEGRADATION` | Rolling MAE > baseline_MAE × 1.15 for 14+ consecutive days | Investigate drift, consider retraining |
| HIGH | `INVENTORY_ANOMALY` | Simulated fill rate drops >10pp below baseline | Review affected store policies |
| MEDIUM | `DRIFT_DETECTED` | KS statistic > 0.15 (p < 0.05) on key features | Investigate feature distributions |
| MEDIUM | `SERVING_BUDGET_EXCEEDED` | Total JSON serving assets > 5MB uncompressed | Increase aggregation in export |

---

## 3. Emergency Procedures

### 3.1 Pipeline Failure

**Symptom:** `make ci` fails at any step.

**Diagnosis:**
```bash
# 1. Identify failed step
cat logs/pipeline.log | tail -100

# 2. Check data contracts
make validate_bronze
make validate_silver
```

**Resolution by failure type:**

| Failure | Likely Cause | Fix |
|---------|-------------|-----|
| `ingest` | Kaggle API rate limit or network | Wait 1h, retry. Check `~/.kaggle/kaggle.json`. |
| `transform` | Schema change in source data or OOM | Check Pandera logs. If OOM, reduce partition size in `configs/data.yaml`. |
| `train` | Feature leakage or OOM | Run `pytest tests/unit/test_leakage_guard.py -v`. If OOM, use `--n_sample`. |
| `export` | JSON too large or invalid JSON | Check `data/gold/serving/asset_manifest.json`. Increase aggregation. |

**Rollback:**
```bash
make rollback
# Restores last known good serving assets from data/gold/serving/backup/
```

---

### 3.2 Model Retraining Decision

**Trigger:** `FORECAST_DEGRADATION` alert sustained for 14+ days (MAE > baseline × 1.15).

**Procedure:**
```bash
# 1. Run full backtesting with current data
make evaluate

# 2. Compare metrics vs baseline in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Compare current vs previous experiment

# 3a. If metrics improved: deploy
make export_serving && make build_frontend && make deploy

# 3b. If metrics NOT improved: investigate
python -c "
from src.monitoring.drift_detector import run_full_drift_report
report = run_full_drift_report(
    reference_path='data/features/feature_store_v1.parquet',
    current_path='data/features/feature_store_latest.parquet'
)
print(report)
"

# 4. Document decision in docs/11_decision_log.md
```

---

### 3.3 Data Quality Incident

**Trigger:** `DATA_QUALITY_FAILURE` alert (Pandera contract violation).

**Procedure:**
```bash
# 1. Identify which contract failed
cat logs/quality_checks.log | grep ERROR

# 2. Quarantine bad data
mv data/bronze/{failed_file}.parquet data/quarantine/$(date +%Y%m%d_%H%M%S)/

# 3. Fix source or transformation
# Edit src/transform/{failing_module}.py

# 4. Re-run from failed step
make transform_silver  # or make ingest if source data changed

# 5. Verify
make validate_bronze && make validate_silver
```

**Escalation:** if the contract violation cannot be fixed, document as a known issue in `docs/12_known_issues.md`.

---

### 3.4 Frontend Broken / Stale Data

**Trigger:** Dashboard not loading or showing stale data.

**Diagnosis:**
```bash
# 1. Check serving assets exist and are recent
ls -la data/gold/serving/*.json
cat data/gold/serving/asset_manifest.json | python -c "
import sys, json; d = json.load(sys.stdin); print(d.get('generated_at', 'no timestamp'))
"

# 2. Check app/public/data/ is current
ls -la app/public/data/

# 3. Run E2E tests
pytest tests/e2e/ -v
```

**Fix:**
```bash
make export_serving          # Regenerate JSON assets
cp data/gold/serving/*.json app/public/data/
cd app && npm run build      # Rebuild
make deploy                  # Push to HF Space
```

---

## 4. Monitoring Architecture

The monitoring layer (`src/monitoring/`) consists of four modules:

| Module | File | Function |
|--------|------|---------|
| Drift Detector | `drift_detector.py` | KS test (sales distribution), PSI (feature distributions), zero-inflation drift, price-regime detection |
| Performance Tracker | `performance_tracker.py` | CUSUM for MAE decay, segmented tracking (by demand class, store, category), calibration check |
| Alert Engine | `alert_engine.py` | 6 alert rules, severity classification, deduplication, rate limiting |
| Health Report Generator | `health_report_generator.py` | Aggregates all signals into `monitoring/health_report.json` |

**Health report structure:**
```json
{
  "generated_at": "2026-03-15T10:30:00Z",
  "overall_status": "GREEN",
  "ks_drift_alerts": [],
  "psi_drift_alerts": ["feature: rolling_mean_28, PSI=0.18 (MEDIUM)"],
  "cusum_alerts": [],
  "active_alerts": [],
  "model_mae_current": null,
  "model_mae_baseline": null,
  "serving_asset_size_bytes": 57344,
  "serving_budget_bytes": 5242880,
  "last_training_run_id": null
}
```

---

## 5. Maintenance Schedule

### Weekly
- Review `monitoring/health_report.json`
- Check drift indicators (KS test, PSI)
- Review active alerts and their resolution

### Monthly
- Recalculate ABC/XYZ classifications (triggers SCD Type 2 if changed)
- Evaluate whether retraining is needed (compare current MAE vs backtesting baseline)
- Update `docs/11_decision_log.md` with any decisions made

### Quarterly
- Full backtesting re-evaluation on latest data
- Review feature importance changes
- Update cost-benefit estimates with latest metrics
- Review and update this runbook

---

## 6. Key File Locations

| File | Purpose |
|------|---------|
| `monitoring/health_report.json` | Latest system health snapshot |
| `data/gold/serving/asset_manifest.json` | Serving asset inventory + timestamps |
| `mlflow.db` | MLflow experiment tracking database |
| `logs/pipeline.log` | Pipeline execution logs |
| `data/quarantine/` | Quarantined bad data |
| `data/gold/serving/backup/` | Last known good serving assets |
| `docs/11_decision_log.md` | Architecture and operational decisions |
| `docs/12_known_issues.md` | Known issues and workarounds |
