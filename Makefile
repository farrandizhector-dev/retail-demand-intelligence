.PHONY: setup ingest validate_bronze transform_silver validate_silver classify_demand \
	build_features validate_features train_baselines train_lgbm evaluate \
	generate_inventory export_serving build_frontend deploy

PYTHON := python

setup:
	@echo "Setup environment (poetry/pip) – pendiente de implementar"

ingest:
	@echo "Ingest pipeline (raw -> bronze) – pendiente de implementar"

validate_bronze:
	@echo "Validate bronze layer – pendiente de implementar"

transform_silver:
	@echo "Transform bronze -> silver – pendiente de implementar"

validate_silver:
	@echo "Validate silver layer – pendiente de implementar"

classify_demand:
	@echo "Demand classification (ADI/CV², ABC/XYZ) – pendiente de implementar"

build_features:
	@echo "Feature store build – pendiente de implementar"

validate_features:
	@echo "Validate feature store – pendiente de implementar"

train_baselines:
	@echo "Train baseline models – pendiente de implementar"

train_lgbm:
	@echo "Train LightGBM global model – pendiente de implementar"

evaluate:
	@echo "Run backtesting evaluation – pendiente de implementar"

generate_inventory:
	@echo "Generate synthetic inventory and policies – pendiente de implementar"

export_serving:
	@echo "Export JSON serving assets – pendiente de implementar"

build_frontend:
	@echo "Build frontend (Vite + React) – pendiente de implementar"

deploy:
	@echo "Deploy static assets to Hugging Face Space – pendiente de implementar"

# ---------------------------------------------------------------------------
# V3: Quality, Testing, CI
# ---------------------------------------------------------------------------

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

test:
	pytest tests/unit/ -v --tb=short

test_integration:
	pytest tests/integration/ -v --tb=short -m "not slow"

test_e2e:
	pytest tests/e2e/ -v --tb=short

test_all: lint test test_integration test_e2e

ci: lint test test_integration
	cd app && npm ci && npm run build && npx tsc --noEmit

rollback:
	@echo "Restore last known good serving assets from data/gold/serving/backup/"
