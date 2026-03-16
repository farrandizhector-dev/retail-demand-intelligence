# Deployment Guide — AI Supply Chain Control Tower

**Document type:** Operations guide
**Audience:** Engineers setting up or deploying the system
**Last updated:** 2026-03-15

---

## 1. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required for match syntax and type hints |
| Node.js | 20+ | For frontend build |
| Git | 2.x+ | |
| Kaggle API | — | For M5 dataset download |
| FRED API key | — | Optional, for macro features |

---

## 2. Local Setup

### Clone and Install

```bash
git clone https://github.com/hferrándiz/retail-demand-intelligence.git
cd retail-demand-intelligence

# Python environment (choose one)
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
python -m venv .venv && .venv\Scripts\activate       # Windows

# Install package with dev dependencies
pip install -e ".[dev]"
```

### Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials:
#   KAGGLE_USERNAME=your_username
#   KAGGLE_KEY=your_kaggle_api_key
#   FRED_API_KEY=your_fred_api_key  (optional)
```

---

## 3. Running the Pipeline

```bash
# Step 1: Ingest raw data
make ingest                  # Downloads M5 + weather (requires KAGGLE_USERNAME + KAGGLE_KEY)

# Step 2: Validate and transform
make validate_bronze         # Pandera checks on bronze layer
make transform_silver        # Bronze -> Silver transformations
make validate_silver         # Pandera checks on silver layer

# Step 3: Classification
make classify_demand         # ADI/CV2 + ABC/XYZ classifications

# Step 4: Feature Engineering
make build_features          # Feature store construction
make validate_features       # Leakage guard checks

# Step 5: Training
make train_baselines         # Seasonal Naive, MA, Croston, TSB
make train_lgbm              # LightGBM global + quantile (15-30 min full data)

# Step 6: Evaluation
make evaluate                # 5-fold rolling-origin backtesting

# Step 7: Inventory
make generate_inventory      # Monte Carlo simulation, policy comparison

# Step 8: Export
make export_serving          # JSON serving assets -> data/gold/serving/

# Or run everything at once:
make ci                      # lint + test + frontend build
```

### Development (Fast Mode)

```bash
# Use n_sample to train on subset for fast iteration
python -m src.evaluation.backtesting --n_sample 500 --n_estimators 100
```

---

## 4. Frontend Development

```bash
cd app

# Install dependencies
npm install

# Copy serving assets (after export_serving)
cp ../data/gold/serving/*.json public/data/

# Development server
npm run dev
# -> http://localhost:5173

# Production build
npm run build
# -> app/dist/ (static files)

# Type check
npx tsc --noEmit
```

---

## 5. Running Tests

```bash
# All tests
pytest tests/ -q

# Unit tests only (fast)
pytest tests/unit/ -v --tb=short

# Integration tests
pytest tests/integration/ -v --tb=short -m "not slow"

# E2E smoke tests (checks serving assets + optional build)
pytest tests/e2e/ -v

# Lint
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

---

## 6. Hugging Face Space Deployment

### Manual Deploy

```bash
# Build frontend
cd app && npm ci && npm run build && cd ..

# Push to HF Space (replace HF_USER and HF_TOKEN)
cd app/dist
git init
git add .
git commit -m "Deploy"
git push --force https://HF_USER:HF_TOKEN@huggingface.co/spaces/HF_USER/ai-supply-chain-control-tower main
```

### Automated Deploy (GitHub Actions)

Push to `main` branch with changes to `app/**` or `data/gold/serving/**` triggers `.github/workflows/deploy_space.yml`:

1. Add `HF_TOKEN` secret to your GitHub repository settings
2. Update `HF_USER` in `deploy_space.yml` to your Hugging Face username
3. Push to `main` → automatic build + deploy

### HF Space Configuration

The space is configured as a static HTML space:

```yaml
# README.md at HF Space root (app/dist/)
---
title: AI Supply Chain Control Tower
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---
```

---

## 7. MLflow UI

```bash
# Start MLflow tracking server (local)
mlflow ui --backend-store-uri sqlite:///mlflow.db
# -> http://localhost:5000
```

---

## 8. PostgreSQL (Optional, for dbt models)

```bash
# Start PostgreSQL (Docker)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=dev postgres:16

# Run dbt models
cd sql && dbt run --profiles-dir .
```

The system works without PostgreSQL for development — all silver layer outputs are Parquet files. PostgreSQL + dbt is required only for gold layer mart queries.

---

## 9. Directory Structure After Full Pipeline Run

```
data/
├── raw/
│   ├── m5/                  # M5 CSVs (from Kaggle)
│   ├── weather/             # Open-Meteo JSON responses
│   └── macro/               # FRED JSON responses
├── bronze/
│   ├── bronze_sales.parquet
│   ├── bronze_calendar.parquet
│   ├── bronze_prices.parquet
│   ├── bronze_weather.parquet
│   └── checksums.json
├── silver/
│   ├── sales/state=CA/year=2011/...
│   ├── silver_prices_daily.parquet
│   ├── silver_calendar_enriched.parquet
│   ├── silver_demand_classification.parquet
│   └── ...
├── features/
│   └── feature_store_v1.parquet
└── gold/
    ├── forecasts/
    │   └── forecast_output.parquet
    ├── inventory/
    │   └── monte_carlo_results.parquet
    ├── backtest_metrics.json
    └── serving/
        ├── asset_manifest.json
        ├── executive_summary.json
        ├── forecast_data.json
        ├── inventory_data.json
        ├── model_quality.json
        ├── shap_summary.json
        └── scenario_results.json

monitoring/
└── health_report.json

mlflow.db                    # MLflow experiment database
```

---

## 10. Troubleshooting Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Kaggle download fails | `403 Forbidden` | Check `~/.kaggle/kaggle.json` exists and has correct credentials |
| OOM during training | `MemoryError` | Use `--n_sample 1000` flag or reduce `num_leaves` in `configs/lgbm.yaml` |
| Serving assets too large | `SERVING_BUDGET_EXCEEDED` | Increase aggregation level in `configs/export.yaml` |
| Frontend build fails | `TypeScript errors` | Run `npx tsc --noEmit` to see type errors; check `app/src/types/` |
| FRED API not available | `ConnectionError` | Set `macro_enabled: false` in `configs/features.yaml` |
| Pandera validation fails | `SchemaError` | Check `logs/quality_checks.log`; inspect quarantined files in `data/quarantine/` |
