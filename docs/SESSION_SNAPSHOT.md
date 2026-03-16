# SESSION SNAPSHOT — Work Log
# Este archivo registra cada sesión de trabajo con IA o manual.
# Cada entrada incluye: qué se hizo, qué archivos cambiaron, y cómo verificar.

---

## Session 17 — Frontend Layout Fixes (2026-03-15)
Phase: Frontend CSS/Layout fixes
Tool: Claude Code

Fixes:
- Sidebar: rebuilt with inline styles (position:fixed, flex-col) — eliminates horizontal nav bug
- Layout: uses children prop with ml-[260px] main — eliminates content overlap
- KPICard: inline styles with hex colors — eliminates Tailwind token compilation issues; icon prop changed from ReactNode to React.ComponentType
- SectionCard: inline styles with hex colors — ensures card backgrounds render; title made optional
- ExecutiveOverview: inline-style grid layout (3-col KPIs, 3fr/2fr charts row, full-width trend); icon props changed to component refs
- ForecastLab: inline-style grid layout (full-width forecast, 2-col comparison, full-width bias)
- InventoryRisk: inline-style grid layout (3-col KPIs, full-width heatmap, 2-col table+histogram); icon props to component refs
- ProductDrilldown: inline-style grid layout (dropdown row, full-width forecast, 2fr/1fr inventory+profile); select styled with inline CSS
- ModelQuality: inline-style grid layout (3-col KPIs, 2-col SHAP+table, full-width calibration); icon props to component refs
- ScenarioLab: inline-style grid layout (header text, 2-col scenario cards, full-width policy table); scenario cards use inline styles
- About: fully converted to inline styles (no Tailwind custom tokens); LAYERS color strings changed to hex values; max-width 800px centered
- Header: rebuilt with inline styles
- executive_summary.json: revenue_by_category only contains FOODS (500-series sample is FOODS-dominant); pages use fallback to show all 3 categories
CONTEXT.md: v1.8.0 → v1.9.0

---

### Session 16 — Frontend Premium Rebuild (2026-03-15)
Phase: Frontend v2.0 — Full rebuild
Tool: Claude Code

Actions:
- Installed: lucide-react, framer-motion
- Rewrote: tailwind.config.ts (dark design tokens: background-base/surface/elevated/hover, primary/success/warning/danger with subtle variants, border/text tokens, card/button/badge border-radius, metric font sizes)
- Rebuilt: src/index.css (Inter font, dark scrollbars, 14px body)
- Created: src/hooks/useServingData.ts (useJsonData generic hook)
- Created: src/utils/echartTheme.ts (dark-premium ECharts theme + COLORS palette)
- Created: Layout components — Sidebar (new nav routes: /overview /forecast /inventory /product /quality /scenarios /about), Header (props-based title+actions), FilterBar (props-based state/category), Layout (260px sidebar + main)
- Created: UI components — KPICard (accent bar + icon overlay + delta), SectionCard (header+subtitle+action), DataTable (zebra stripe + highlightRow), Badge (5 variants), EmptyState, LoadingSpinner, EChartWrapper (echarts.init + ResizeObserver)
- Rebuilt: 7 pages with default exports (lazy-loaded from App.tsx) — ExecutiveOverview, ForecastLab, InventoryRisk, ProductDrilldown, ModelQuality, ScenarioLab, About
- Updated: App.tsx (lazy routing with Navigate redirect, no more AppRouter/router.tsx)
- Updated: main.tsx (echarts.registerTheme dark-premium)
- Build: SUCCESS — 2323 modules, 1.4 MB dist, 0 TypeScript errors
- Tests: 795/795 passing
CONTEXT.md: v1.7.0 → v1.8.0

---

### Session 15 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V1 Pipeline Execution — v1.7.0

**Estado anterior:** V3 completado, 795 tests passing. Pipeline nunca ejecutado con datos reales.

**Que se hizo:**

Ejecucion completa del pipeline V1 end-to-end con datos M5 reales:

- **STEP 0 (Env check):** Python 3.12.3, deps OK (polars, lightgbm, pandera, mlflow).
- **STEP 1 (Extract M5 zip):** Extraidos 5 CSVs del zip (45.8 MB) — calendar.csv (102 KB), sales_train_validation.csv (117 MB), sell_prices.csv (194 MB), etc.
- **STEP 2 (Bronze writer):** 3 parquets generados — bronze_sales (29.4 MB, 30,490 rows x 1919 cols), bronze_prices (4.3 MB, 6.84M rows), bronze_calendar (0.04 MB, 1,969 rows).
- **STEP 3 (Silver transform):** 28.5s. silver_sales_long: 58,327,370 rows x 9 cols (18 partitions, 185.2 MB). silver_prices_daily: 47,735,397 rows x 4 cols. silver_calendar_enriched: 1,969 rows x 16 cols.
- **STEP 4 (Demand classification):** ADI/CV2 + ABC/XYZ para 30,490 series. smooth=980, erratic=497, intermittent=23,102, lumpy=5,911. ABC: A=12,640 / B=9,230 / C=8,620.
- **STEP 5 (Feature store — SAMPLE MODE):** OOM/segfault con 58M filas en proceso Python unico. Usado sample de 500 series (956,500 filas). Feature store: 76 columnas (7 familias: lags, rolling, intermittency, calendar, prices, weather, interactions). Correccion aplicada: columnas String (event_name_1, demand_class, abc_class, xyz_class) codificadas como Int32 para LightGBM.
- **STEP 6 (LightGBM training — SAMPLE MODE):** 5-fold backtesting en 500 series. 292.7s. Fold metrics guardados en data/gold/backtest_metrics.json.
- **STEP 7 (Backtesting metrics):** MAE=0.6260, RMSE=1.8171, SMAPE=0.4108, Coverage@80=90.0%, WRMSSE=0.5511, Pinball=0.1798.
- **STEP 8 (Inventory simulation):** synthetic_params generados para 30,490 series. Simulacion en 500-series sample: fill_rate=89.7%, stockout_days=4.36, avg_inventory=23.8.
- **STEP 9 (Export serving JSONs):** 7 JSONs generados — executive_summary, forecast_series_CA/TX/WI_FOODS, inventory_risk_matrix, model_metrics, asset_manifest. Bundle: 58.7 KB.
- **STEP 10 (Sync frontend):** 13 JSONs sincronizados a app/public/data/.
- **Bug fix:** _build_manifest() en serving_exporter.py no incluia self-reference (manifest_file) ni campo generated_at — test e2e fallaba. Corregido.
- **Tests:** 795/795 passing (14.13s) despues del fix.

**Modo:** sample-500 para feature store, training e inventario. Clasificacion y synthetic_params en dataset completo (30,490 series).

**Archivos cambiados:**
- `src/export/serving_exporter.py` — _build_manifest() con manifest_file + generated_at
- `data/bronze/` — 3 parquets generados
- `data/silver/` — silver_sales_long (18 partitions), silver_prices_daily, silver_calendar_enriched
- `data/features/feature_store_v1.parquet` — 956,500 rows x 76 cols (sample)
- `data/gold/synthetic_params.parquet` — 30,490 rows
- `data/gold/inventory_snapshot.parquet` — 500 rows
- `data/gold/backtest_predictions.parquet` — 70,000 rows
- `data/gold/backtest_metrics.json` — 5-fold metrics
- `data/gold/serving/*.json` — 7 serving JSONs
- `app/public/data/*.json` — sincronizado
- `docs/CONTEXT.md` — bump v1.6.0 -> v1.7.0

**Comandos para verificar:**
```bash
python -c "import polars as pl; dc = pl.read_parquet('data/silver/demand_classification.parquet'); print(len(dc), 'series clasificadas')"
python -c "import json; m=json.load(open('data/gold/backtest_metrics.json')); print('MAE:', sum(r['mae'] for r in m)/len(m))"
pytest tests/ -q --tb=no 2>&1 | tail -3
```

---

### Session 14 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — Pre-Release Audit — v1.6.0

**Estado anterior:** V3 completado, 795 tests passing. Audit pendiente.

**Qué se hizo:**

Auditoría pre-release completa del proyecto. Resultados:

- **Tests:** 795/795 passing (13.47s). 4 warnings (pandera PerformanceWarning + FutureWarning en integration test — known, non-blocking).
- **Frontend build:** Vite build exitoso en 2.71s. 1 warning: ECharts chunk 628 KB > 500 KB (conocido, KI-001).
- **Serving JSONs:** 13 JSONs válidos en data/gold/serving/ y app/public/data/. Bundle 57 KB / 5 MB budget (1.1%).
- **Documentación:** 15/15 archivos de docs presentes y con el contenido mínimo requerido.
- **Código fuente:** 54 módulos Python en src/ bajo estructura correcta (src/ingest/, src/classification/, src/features/, src/models/, src/inventory/, src/evaluation/, src/export/, src/reconciliation/, src/monitoring/, etc.).
- **CI/CD:** .github/workflows/ci.yml + deploy_space.yml presentes.
- **Fix aplicado:** asset_manifest.json en data/gold/serving/ carecía de campo `generated_at` — añadido y sincronizado a app/public/data/.

**Issues NO bloqueantes (pipeline no ejecutado):**
- M5 CSVs no extraídos del zip (zip presente: 45.8 MB). Requiere `make ingest`.
- data/bronze/, data/silver/, data/features/ vacíos (pipeline no ejecutado).
- synthetic_params.parquet, dim_product_scd.parquet, data/gold/models/ ausentes (generados por pipeline).
- Documentados como DT-002, DT-003 en CONTEXT.md.

**Archivos cambiados:**
- `data/gold/serving/asset_manifest.json` — añadido campo `generated_at`
- `app/public/data/asset_manifest.json` — sincronizado con serving
- `docs/CONTEXT.md` — bump v1.5.0 → v1.6.0, añadidos DT-002 a DT-005
- `docs/SESSION_SNAPSHOT.md` — esta entrada

**Verificar:**
```bash
cd c:/Users/Hector/Documents/retail-demand-intelligence
python -m pytest tests/ -v --tb=short 2>&1 | tail -5
python -c "import json; d=json.load(open('data/gold/serving/asset_manifest.json')); print('generated_at' in d)"
```

---

### Session 13 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V3-Fase 4: Documentation — V3 COMPLETADO

**Estado anterior:** V3-Fases 1-3 completadas (795 tests passing). V3-Fase 4 (Documentation) pendiente.

**Qué se hizo:**

Creados 11 archivos de documentación completos (sin stubs ni placeholders excepto métricas de backtesting que requieren ejecución real de pipeline):

- `README.md` (raíz del proyecto) — descripción completa del sistema, arquitectura Mermaid, tabla de features, tech stack, quick start, data sources, resultados (placeholder para métricas reales), disclaimer sintético, estructura de proyecto, autor.
- `docs/01_business_case.md` — caso de negocio con estimaciones financieras ($886K–$951K/year, 10 stores), personas de usuario (Executive/Operational/Technical), comparativa vs alternativas, diferenciadores estratégicos.
- `docs/02_data_architecture.md` — flujo Raw→Bronze→Silver→Gold, modelo de ownership por dominio (10 dominios), contratos inter-dominio, estrategia de calidad (Pandera), trazabilidad de lineaje (de KPI a CSV raw), principios de diseño, estrategia de particionamiento, política SHA-256.
- `docs/03_ml_system_design.md` — framing del problema, clasificación ADI/CV², 8 familias de features (~85 features), protocolo de leakage (6 reglas), model stack (baselines→LightGBM→conformal→reconciliación), backtesting 5-fold rolling-origin, reconciliación jerárquica, MLflow tracking, SHAP importance.
- `docs/04_inventory_engine.md` — fórmulas matemáticas (SS, ROP, EOQ, newsvendor, fill rate), diseño del simulador Monte Carlo (1000 paths × 90 días, 5.8 ms/serie, vectorización NumPy), 4 políticas de inventario comparadas, 5 escenarios what-if, metodología de generación sintética, integración con capa de forecast, schema de output JSON.
- `docs/05_demand_classification.md` — por qué clasificar antes de modelar, ADI/CV² con fórmulas y código de implementación, ABC/XYZ (revenue + variabilidad), matriz combinada ABC/XYZ→service level, impacto en pipeline por clase, distribución aproximada en M5 (30,490 series), SCD Type 2 para historial de clasificación con ejemplo de dim_product.
- `docs/06_reconciliation.md` — problema de incoherencia con ejemplo numérico, jerarquía M5 de 12 niveles (42,840 series), 3 métodos (BU/TD/MinT-Shrink), detalle matemático MinT-Shrink con shrinkage estimator, estructura S matrix, 4 coherence tests obligatorios con código, descomposición sub-jerarquía por store, implementación con hierarchicalforecast.
- `docs/08_synthetic_data_disclaimer.md` — tabla completa REAL vs SYNTHETIC, justificación de la elección, código de generación con distribuciones y seeds, dónde aparece el label SYNTHETIC (código/JSON/frontend/model cards), validez de la metodología, reproducibilidad.
- `docs/09_deployment_guide.md` — prerequisites, setup local (clone+install+env vars), pipeline completo paso a paso (8 steps + modo desarrollo), frontend dev, tests (unit/integration/e2e/lint), HF Space deploy (manual + automático), MLflow UI, PostgreSQL opcional, estructura de directorios post-pipeline, troubleshooting.
- `docs/10_monitoring_runbook.md` — checklist semanal, tabla de 6 alertas (CRITICAL/HIGH/MEDIUM), 4 procedimientos de emergencia (pipeline failure / model retraining / data quality incident / frontend broken), arquitectura de monitoring (4 módulos + estructura health_report JSON), calendario de mantenimiento (weekly/monthly/quarterly), ubicaciones de archivos clave.
- `docs/12_known_issues.md` — 6 issues documentados: KI-001 ECharts bundle size warning, KI-002 FRED API key opcional, KI-003 Pandera LazyFrame PerformanceWarning, KI-004 tiempos de ejecución por step, KI-005 HF_USER placeholder en CI, KI-006 MinT-Shrink memory at full scale.

**Archivos cambiados:**
- `README.md` — creado (raíz)
- `docs/01_business_case.md` — creado
- `docs/02_data_architecture.md` — creado
- `docs/03_ml_system_design.md` — creado
- `docs/04_inventory_engine.md` — creado
- `docs/05_demand_classification.md` — creado
- `docs/06_reconciliation.md` — creado
- `docs/08_synthetic_data_disclaimer.md` — creado
- `docs/09_deployment_guide.md` — creado
- `docs/10_monitoring_runbook.md` — creado
- `docs/12_known_issues.md` — creado
- `docs/CONTEXT.md` — actualizado a v1.5.0, V3-Fase 4 completada, V3 marcado COMPLETADO, Fase actual = "V3 COMPLETADO — proyecto terminado"

**Tests:** 795 tests passing (docs no afectan tests).

**Cómo verificar:**
```bash
pytest tests/ -q          # debe dar 795 passed
ls docs/                  # debe mostrar todos los archivos nuevos
cat docs/CONTEXT.md | head -5  # Version: v1.5.0
```

---

### Session 12 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V3-Fases 2+3: SCD Type 2 + CI/CD + E2E Tests

**Estado anterior:** V3-Fase 1 completada (753 tests passing). V3-Fases 2 y 3 pendientes.

**Qué se hizo:**

**V3-Fase 2 — SCD Type 2:**
- Creado `src/classification/scd_manager.py`:
  - `detect_classification_changes(current_df, previous_df)` — detecta cambios en abc_class, xyz_class, demand_class; filtra is_current=True antes de comparar; incluye items nuevos con old_* nulls.
  - `apply_scd_type2(dim_product_df, new_classification_df, effective_date)` — lógica completa SCD Type 2: first load (keys 1..N, version=1) + incremental (cierre old row con valid_to=effective_date-1, nueva row con key max+1 y version+1).
  - `save_dim_product(dim_product_df, output_path)` — escribe parquet, crea directorios padre.
  - Constantes: SCD_MAX_DATE=date(9999,12,31), TRACKED_COLS, SCD_COLS.
- Creado `sql/intermediate/int_product_scd2.sql` — dbt snapshot spec §7.2 con strategy='check', check_cols=['abc_class','xyz_class','demand_class'], invalidate_hard_deletes=True.
- Creado `tests/unit/test_scd_manager.py` — 25 tests: TestDetectClassificationChanges (8 tests), TestApplyScdType2FirstLoad (5 tests), TestApplyScdType2WithChanges (9 tests), TestSaveDimProduct (3 tests). Todos pasan.

**V3-Fase 3 — CI/CD + E2E Tests:**
- Creado `.github/workflows/ci.yml` — 3 jobs: lint (ruff+mypy), test (unit+integration), frontend (npm ci + build + tsc).
- Creado `.github/workflows/deploy_space.yml` — deploy a HF Space en push a main cuando cambia app/** o data/gold/serving/**.
- Creado `tests/e2e/__init__.py` — paquete vacío.
- Creado `tests/e2e/test_frontend_smoke.py` — 17 tests: TestServingAssetsExist (6), TestServingAssetsValidJson (5), TestServingBudget (2), TestAssetManifestComplete (2), TestBuildSucceeds (2, skipif npm unavailable). Todos pasan.
- Actualizado `app/public/data/asset_manifest.json` — añadido `generated_at` timestamp y entrada self-referencial.
- Actualizado `Makefile` — añadidos targets: lint, test, test_integration, test_e2e, test_all, ci, rollback.

**Archivos cambiados:**
- `src/classification/scd_manager.py` (nuevo)
- `sql/intermediate/int_product_scd2.sql` (nuevo)
- `.github/workflows/ci.yml` (nuevo)
- `.github/workflows/deploy_space.yml` (nuevo)
- `tests/e2e/__init__.py` (nuevo)
- `tests/e2e/test_frontend_smoke.py` (nuevo)
- `tests/unit/test_scd_manager.py` (nuevo)
- `app/public/data/asset_manifest.json` (actualizado: +generated_at, +self-entry)
- `Makefile` (actualizado: +V3 targets)
- `docs/CONTEXT.md` (bump v1.3.0 → v1.4.0)

**Tests:** 753 → 795 passing (+42: 25 unit SCD + 17 e2e smoke)

**Verificar:**
```bash
pytest tests/unit/test_scd_manager.py -v       # 25 passed
pytest tests/e2e/test_frontend_smoke.py -v     # 17 passed
pytest tests/ -q                                # 795 passed
```

---

### Session 11 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V3-Fase 1: Monitoring + Alerting

**Estado anterior:** V2 completado (614 tests passing). V3-Fase 1 marcada como pendiente.

**Qué se hizo:**
- Creado `src/monitoring/__init__.py` (paquete vacío).
- Creado `src/monitoring/drift_detector.py` con 4 detectores:
  - `sales_distribution_drift` — KS test on daily aggregate sales (scipy.stats.ks_2samp, lazy import).
  - `compute_psi` — Population Stability Index using equal-frequency binning.
  - `feature_distribution_drift` — PSI per feature with 4-tier status labels (ok/warning/alert/retrain).
  - `zero_inflation_shift` — % zero sales per category, 5pp threshold.
  - `price_regime_change` — Mean sell_price by dept, 10% threshold.
- Creado `src/monitoring/performance_tracker.py` con 3 monitores:
  - `cusum_detector` — CUSUM on weekly MAE, configurable threshold_factor and min_weeks.
  - `segment_performance_check` — MAE by segment vs baseline (25% alert threshold).
  - `calibration_monitor` — empirical coverage of [p10, p90] intervals, 70%/90% bounds.
- Creado `src/monitoring/alert_engine.py` con 6 reglas:
  - `check_data_quality_failure` (CRITICAL), `check_forecast_degradation` (HIGH),
    `check_drift_detected` (MEDIUM), `check_inventory_anomaly` (HIGH),
    `check_reconciliation_incoherence` (CRITICAL), `check_serving_budget` (MEDIUM).
  - `check_all_alerts` orchestrator, `alerts_to_json` serialiser.
- Creado `src/monitoring/health_report_generator.py`:
  - `HealthReport` dataclass, `determine_overall_status`, `build_recommendations`,
    `generate_health_report`, `save_health_report`.
- Creado 4 ficheros de tests: `tests/unit/test_drift_detector.py` (40 tests),
  `tests/unit/test_performance_tracker.py` (25 tests), `tests/unit/test_alert_engine.py` (34 tests),
  `tests/unit/test_health_report.py` (23 tests) — 139 tests nuevos.

**Resultado:** 753 tests passing (614 anteriores + 139 nuevos), 0 failures.

**Archivos creados/modificados:**
- NEW: `src/monitoring/__init__.py`
- NEW: `src/monitoring/drift_detector.py`
- NEW: `src/monitoring/performance_tracker.py`
- NEW: `src/monitoring/alert_engine.py`
- NEW: `src/monitoring/health_report_generator.py`
- NEW: `tests/unit/test_drift_detector.py`
- NEW: `tests/unit/test_performance_tracker.py`
- NEW: `tests/unit/test_alert_engine.py`
- NEW: `tests/unit/test_health_report.py`
- UPDATED: `docs/CONTEXT.md` (v1.2.0 → v1.3.0)

**Cómo verificar:**
```bash
pytest tests/unit/test_drift_detector.py tests/unit/test_performance_tracker.py tests/unit/test_alert_engine.py tests/unit/test_health_report.py -v
pytest tests/ -q  # 753 passed
```

---

### Session 6 — 2026-03-15 — Cursor (GPT-5.1) — V1-Fase 5: Frontend + Deploy

**Estado anterior:** V1-Fase 4 completada con backend y serving JSON (`data/gold/serving/*.json` y `asset_manifest.json`) listos. V1-Fase 5 marcada como pendiente; en `app/` sólo existía el scaffolding mínimo (config de Vite/Tailwind/TS), sin páginas ni routing ni build probado.

**Qué se hizo:**
- Leídos `docs/CONTEXT.md` (v0.6.0), `docs/MASTER_SPEC_v3.md` (secciones 13.1–13.4) y `data/gold/serving/asset_manifest.json` para alinear las páginas con los assets reales (`executive_summary.json`, `forecast_series_{state}_{category}.json`, `inventory_risk_matrix.json`, `model_metrics.json`).
- Instaladas dependencias de frontend en `app/` (`npm install`) incluyendo `react-router-dom` para routing.
- Creado entrypoint de la SPA:
  - `app/index.html` con tema dark, fuente Inter desde Google Fonts y `<div id="root">`.
  - `app/src/index.css` con `@tailwind` y estilos base (14px, fondo `#0F1117`, texto `#E5E7EB`).
  - `app/src/main.tsx` usando `ReactDOM.createRoot` + `BrowserRouter`.
  - `app/src/App.tsx` envolviendo el router en el layout principal.
- Configurado routing según la spec:
  - `app/src/router.tsx` con React Router v6: rutas `/overview`, `/forecast`, `/risk`, `/about` y redirect de `/` → `/overview`.
- Layout y navegación (design system dark, alta densidad):
  - `app/tailwind.config.ts` ajustado a los tokens clave: colores `background.base/surface/elevated`, `primary`, `success`, `warning`, `danger`, `text.primary/secondary`, `border.default/subtle/focus`, `shadow.card` = `0 1px 3px rgba(0,0,0,0.24)`, radius `sm` y `md`.
  - `app/src/components/layout/Sidebar.tsx`: sidebar oscuro con logo “AI Supply Chain Control Tower” y navegación colapsable conceptual a 4 rutas (Overview, Forecast Lab, Inventory Risk, About).
  - `app/src/components/layout/Header.tsx`: título dinámico según ruta + `FilterBar` global.
  - `app/src/components/layout/FilterBar.tsx`: filtros de `state`, `category`, `department` conectados al store global (Zustand), con selects compactos.
  - `app/src/components/layout/Layout.tsx`: composición final lado a lado (sidebar + header + main) respetando los tamaños del design system.
- Estado global y carga de datos:
  - `app/src/stores/filterStore.ts`: store Zustand con `selectedState`, `selectedCategory`, `selectedDepartment` y setters, inicializados en `"all"`.
  - `app/src/hooks/useData.ts`: hook genérico para cargar JSON desde `/data/{path}` con estados `loading`/`error` y manejo de cancelación.
- Integración con ECharts:
  - `app/src/components/charts/BaseEChart.tsx`: wrapper centralizado para ECharts (`Bar`, `Line`, `Pie`, `Heatmap`, `Gauge`) con init/dispose y resize responsive.
- Páginas implementadas:
  1. `app/src/pages/ExecutiveOverview.tsx` (Executive Overview):
     - Consume `executive_summary.json` (estructura esperada: `kpis`, `revenue_by_state`, `revenue_by_category`, `monthly_trend`).
     - Renderiza 6 KPI cards en grid (Revenue Proxy, Fill Rate, Forecast MAE, Stockout Rate, Inventory Value, Avg Days of Supply) con valor grande formateado, delta `▲/▼` y vs-period textual; estiladas con background `surface`, `shadow.card`, radius `md`.
     - Gráfico “Revenue by State” como bar chart (ECharts) y “Revenue by Category” como donut chart.
     - “Revenue Trend (Last 12 Months)” como line chart con área sombreada.
  2. `app/src/pages/ForecastLab.tsx` (Forecast Lab):
     - Usa filtros globales para elegir `state` y `category` (fall back a CA/FOODS cuando están en `"all"`).
     - Carga `forecast_series_{state}_{category}.json` y `model_metrics.json`.
     - Time series chart: actual (línea amarilla), forecast_p50 (línea azul), intervalo p10–p90 como área sombreada.
     - Model comparison table: filas `segment/model_id/MAE`.
     - MAE by category: horizontal bar chart.
     - Bias indicator agregado: calcula bias medio y muestra mensaje semántico (over- vs under-forecast) con colores `warning`/`danger`.
  3. `app/src/pages/InventoryRisk.tsx` (Inventory Risk):
     - Consume `inventory_risk_matrix.json` con `heatmap` (store×department), `at_risk_items`, `days_of_supply_distribution`, `fill_rate_by_state`.
     - Heatmap stores (filas) × departments (columnas) con visualMap 0→1 (verde→amarillo→rojo) según `stockout_probability`.
     - Tabla de Top 10 at-risk items ordenados por probabilidad de stockout (con días de supply).
     - Histograma de distribución de Days of Supply.
     - Fill Rate gauge por estado resumido en un gauge multi-etiqueta.
  4. `app/src/pages/About.tsx` (About / Methodology):
     - Secciones estáticas: Architecture, Data Sources (M5, Open-Meteo, FRED con links), Methodology (ADI/CV², LightGBM global + quantiles), Synthetic Data Disclaimer (bloque destacado que explica claramente qué es SYNTHETIC y qué es real) y Tech Stack badges.
     - Incluye enlace placeholder a GitHub.
- Build:
  - Ejecutado `npm run build` en `app/` con éxito: Vite genera `dist/` sin errores; único warning por tamaño de chunk >500KB (aceptable para V1).

**Archivos creados/modificados:**
- Nuevos:
  - `app/index.html`
  - `app/src/index.css`
  - `app/src/main.tsx`
  - `app/src/App.tsx`
  - `app/src/router.tsx`
  - `app/src/stores/filterStore.ts`
  - `app/src/hooks/useData.ts`
  - `app/src/components/layout/Sidebar.tsx`
  - `app/src/components/layout/Header.tsx`
  - `app/src/components/layout/FilterBar.tsx`
  - `app/src/components/layout/Layout.tsx`
  - `app/src/components/charts/BaseEChart.tsx`
  - `app/src/pages/ExecutiveOverview.tsx`
  - `app/src/pages/ForecastLab.tsx`
  - `app/src/pages/InventoryRisk.tsx`
  - `app/src/pages/About.tsx`
- Modificados:
  - `app/package.json` (añadido `react-router-dom` y limpieza de script `lint` no usado).
  - `app/tailwind.config.ts` (tokens color, shadow.card, radius `sm`/`md`).
  - `docs/CONTEXT.md` (bump a v0.7.0, V1-Fase 5 marcada como completada, V1 como release completado).
  - `docs/SESSION_SNAPSHOT.md` (esta entrada).

**Comandos de verificación:**
```bash
cd retail-demand-intelligence

# 1) Ver que el build de frontend compila sin errores
cd app
npm run build

# 2) (Opcional) Servir estático para probar localmente
npm run preview
# Abrir la URL que indique Vite (por defecto http://localhost:4173) y navegar:
# - /overview
# - /forecast
# - /risk
# - /about

# 3) Verificar que los assets JSON existen y son pequeños
cd ..
ls data/gold/serving
cat data/gold/serving/asset_manifest.json
```

**CONTEXT.md actualizado:** Sí — versión a v0.7.0, V1-Fase 5 marcada como completada (frontend 4 páginas + build exitoso), V1 señalada como release core demo completo (pendiente sólo de deploy a HF Space).

**Estado después:** V1-Fase 5 (Frontend + Deploy) completada a nivel de código y build local. El repositorio ahora tiene backend completo + 4 páginas de dashboard (Executive Overview, Forecast Lab, Inventory Risk, About) consumiendo JSON precalculado y respetando el design system dark y las constraints de la spec. Siguiente paso natural: configurar y apuntar un Hugging Face Static Space al contenido de `app/dist/` para publicar la demo.

---

### Session 7 — 2026-03-15 — Cursor (GPT-5.1) — V1-Fase 5: Frontend UI Polish

**Estado anterior:** V1-Fase 5 completada funcionalmente con 4 páginas y build correcto, pero con diseño todavía genérico (sidebar sin iconos, filtros y cards poco pulidos, Inventory Risk mostrando errores cuando sólo hay placeholder SYNTHETIC).

**Qué se hizo:**
- Instalado `lucide-react` en `app/` para iconografía profesional en el sidebar.
- Mejorado el layout global:
  - `Sidebar.tsx`: sidebar fijo de 260px a la izquierda (`bg=#1A1D29` vía `background.surface`), columna vertical con 4 links (Overview, Forecast Lab, Inventory Risk, About) cada uno con icono (`LayoutDashboard`, `LineChart`, `ShieldAlert`, `Info`) de `lucide-react`. Mantiene borde derecho y título “AI Supply Chain Control Tower”.
  - `Layout.tsx`: contenido principal a la derecha con `px-8 py-6` y `max-w-6xl`, respetando la densidad alta del design system.
- Pulido de Filter Bar:
  - `FilterBar.tsx`: selects con estilo dark consistente (bg `#242736`, borde `#2D3348`, texto `#E5E7EB`, `rounded-md`, padding compacto, `focus:ring` en color de foco).
- Estilo consistente de KPI cards y charts:
  - `ExecutiveOverview.tsx`: cards de KPIs en grid 3 columnas, cada card con `bg=#1A1D29`, `border-radius=8px`, `padding≈20px`, `shadow.card`; label `text-sm` gris, valor `2rem` `font-bold`, delta `text-xs` en verde/rojo según signo. Gráficos de Revenue by State / Category en grid de 2 columnas, cada uno en su propia card estilada; line chart de trend en card full-width debajo.
  - `ForecastLab.tsx`: se han aplicado las mismas clases de card y grid a la serie temporal, la tabla de comparación de modelos y el gráfico de MAE by category, más la card de Bias Indicator.
  - `InventoryRisk.tsx`: mismas clases de card para heatmap, lista de Top 10 y gráficos de Days of Supply y Fill Rate gauge.
- Mejora de robustez en Inventory Risk:
  - `InventoryRisk.tsx` ya contemplaba el JSON placeholder (`{"stores":[],"synthetic_tag":"SYNTHETIC"}`); se afinó el mensaje para mostrar un bloque informativo en card cuando no hay datos de simulación, evitando errores de `.map()` sobre estructuras vacías.
- Ajuste de `ExecutiveOverview.tsx` (sesión previa, pero consolidado): alineado a la estructura real de `executive_summary.json` (objetos para `revenue_by_state`/`revenue_by_category` y `sales` en `monthly_trend`), transformando a arrays con `Object.entries` antes de alimentar ECharts; KPIs ahora derivados de `revenue_proxy_total`, `fill_rate_avg`, etc.

**Archivos creados/modificados:**
- Dependencias:
  - `app/package.json` (añadido `lucide-react`).
- Layout y estilo:
  - `app/src/components/layout/Sidebar.tsx` (iconos lucide, sidebar 260px, nav vertical).
  - `app/src/components/layout/Layout.tsx` (padding de contenido).
  - `app/src/components/layout/FilterBar.tsx` (estilo dark refinado para selects).
- Páginas:
  - `app/src/pages/ExecutiveOverview.tsx` (tipos y mapeo acorde a `executive_summary.json`, nueva construcción de KPIs + cards y cards de charts).
  - `app/src/pages/ForecastLab.tsx` (cards y layout pulidos).
  - `app/src/pages/InventoryRisk.tsx` (cards y layout pulidos, manejo claro del placeholder SYNTHETIC).
- Documentos:
  - `docs/CONTEXT.md` (bump menor a v0.7.1 indicando UI polish en V1-Fase 5).
  - `docs/SESSION_SNAPSHOT.md` (esta entrada).

**Comandos de verificación:**
```bash
cd retail-demand-intelligence/app

# Ver que lucide-react está instalado
cat package.json | findstr "lucide-react"

# Lanzar la app en modo dev y revisar el layout actualizado
npm run dev
# Navegar en el navegador a /overview, /forecast, /risk, /about
# - Comprobar sidebar fijo de 260px con iconos
# - Verificar grid de 3 KPIs por fila en Overview
# - Confirmar cards dark con sombra y paddings correctos
```

**CONTEXT.md actualizado:** Sí — versión a v0.7.1, anotando que se trata de un refinamiento de UI sobre V1-Fase 5 (sin cambios funcionales ni de pipeline).

**Estado después:** V1 sigue completado, ahora con un dashboard visualmente más profesional: sidebar dark con iconos, KPI cards sólidas, charts en cards bien estructuradas y filtros coherentes con el design system dark. Listo para demo en HF Space sin ajustes visuales mayores.

---

## Formato de entrada

```
### Session [N] — [FECHA] — [HERRAMIENTA] — [FASE]
**Estado anterior:** [qué había antes]
**Qué se hizo:** [descripción]
**Archivos creados/modificados:** [lista]
**Comandos de verificación:** [cómo comprobar que funciona]
**CONTEXT.md actualizado:** [sí/no, qué cambió]
**Estado después:** [qué hay ahora]
```

---

## Sessions

### Session 0 — 2026-03-15 — Claude.ai (Opus) — Pre-project

**Estado anterior:** No existía el proyecto.

**Qué se hizo:**
- Creado documento de especificación v1 → v2 → v3 (iteraciones en claude.ai)
- Documento final: `AI_Supply_Chain_Control_Tower_v3_Principal_Level.md` (2,031 líneas)
- Documento Word: `AI_Supply_Chain_Control_Tower_v3.docx` (678 párrafos)
- Definida arquitectura completa, modelo de datos, feature engineering, modelado, inventario
- Cerrados 8 gaps identificados (CBA, SCD Type 2, domain ownership, monitoring, runbook, design system WCAG, reconciliación profunda, diagramas)
- Definida estrategia V1/V2/V3 con asignación de herramientas IA
- Creados docs/CONTEXT.md y docs/SESSION_SNAPSHOT.md (este archivo)
- Creado .cursorrules para inyección automática de contexto

**Archivos creados:**
- `docs/MASTER_SPEC_v3.md` (spec maestra, no editar)
- `docs/CONTEXT.md` (fuente de verdad viva)
- `docs/SESSION_SNAPSHOT.md` (este archivo)
- `.cursorrules` (reglas automáticas para Cursor)

**Comandos de verificación:**
```bash
# Verificar que los archivos existen
ls -la docs/CONTEXT.md docs/SESSION_SNAPSHOT.md docs/MASTER_SPEC_v3.md .cursorrules
```

**CONTEXT.md:** Creado en v0.1.0

**Estado después:** Proyecto especificado al 100%. Listo para V1-Fase 0 (Scaffolding con Cursor).

---

### Session 1 — 2026-03-15 — Cursor (GPT-5.1) — V1-Fase 0: Scaffolding

**Estado anterior:** Solo existían `docs/MASTER_SPEC_v3.md`, `docs/CONTEXT.md` (v0.1.0), `docs/SESSION_SNAPSHOT.md` y `.cursorrules`. V1-Fase 0 marcada como pendiente.

**Qué se hizo:**
- Leída la spec maestra (`docs/MASTER_SPEC_v3.md`) y el contexto (`docs/CONTEXT.md`, `docs/SESSION_SNAPSHOT.md`).
- Creada la estructura de carpetas completa para datos (`data/raw/*`, `data/bronze`, `data/silver`, `data/gold/serving`, `data/external`, `data/quarantine`), SQL (`sql/staging`, `sql/intermediate`, `sql/marts`), código (`src/*`), tests (`tests/unit`, `tests/integration`) y frontend (`app/src/pages`, `app/src/components`, `app/src/hooks`, `app/src/stores`, `app/src/types`, `app/src/utils`), así como `.github/workflows`.
- Creado `pyproject.toml` con dependencias principales: polars, pandas, lightgbm, statsforecast, pandera, mlflow, httpx, scipy, numpy, scikit-learn, pytest, ruff, mypy.
- Creado `Makefile` con todos los targets de V1 (`setup`, `ingest`, `validate_bronze`, `transform_silver`, `validate_silver`, `classify_demand`, `build_features`, `validate_features`, `train_baselines`, `train_lgbm`, `evaluate`, `generate_inventory`, `export_serving`, `build_frontend`, `deploy`) como placeholders sin lógica.
- Creado `.env.example` y `.gitignore` según la spec (incluyendo datos, `.env`, `mlflow.db`, `node_modules`, `app/dist`, etc.).
- Creado todos los `configs/*.yaml`: `data.yaml`, `features.yaml`, `training.yaml`, `evaluation.yaml`, `inventory.yaml`, `reconciliation.yaml`, `monitoring.yaml`, `dashboard.yaml`, alineados con las secciones 3, 8, 9, 10, 11 y 13 de la spec.
- Creado `contracts/*.yaml`: `silver_sales.yaml`, `forecast_output.yaml`, `serving_assets.yaml`, basados en los contratos de datos y serving assets de la spec.
- Creado esqueletos Python en `src/` con docstrings y firmas placeholder para los paquetes clave: `ingest/`, `validation/`, `transform/`, `classification/`, `features/`, `models/`, `reconciliation/`, `evaluation/`, `inventory/`, `export/`, `utils/`. Incluye módulos representativos como `m5_downloader.py`, `weather_fetcher.py`, `validation/contracts.py`, `transform/pipeline.py`, `classification/demand_classifier.py`, `features/feature_store.py`, `models/training.py`, `reconciliation/reconciler.py`, `evaluation/backtesting.py`, `inventory/engine.py`, `export/serving_exporter.py`, `utils/paths.py`.
- Creado `tests/conftest.py` con fixtures básicos (`project_root`, `tmp_data_dir`).
- Creado scaffolding frontend: `app/package.json` (React 18, TypeScript, Tailwind, ECharts, Vite, Zustand), `app/tsconfig.json`, `app/vite.config.ts`, `app/tailwind.config.ts`.
- Ejecutado linter estático (ruff/mypy vía herramienta de lints de Cursor) sobre los directorios creados sin errores.
- Actualizado `docs/CONTEXT.md` a v0.2.0 marcando V1-Fase 0 como completada y reflejando el estado de los archivos clave.

**Archivos creados/modificados:**
- `pyproject.toml`
- `Makefile`
- `.env.example`
- `.gitignore`
- `configs/data.yaml`
- `configs/features.yaml`
- `configs/training.yaml`
- `configs/evaluation.yaml`
- `configs/inventory.yaml`
- `configs/reconciliation.yaml`
- `configs/monitoring.yaml`
- `configs/dashboard.yaml`
- `contracts/silver_sales.yaml`
- `contracts/forecast_output.yaml`
- `contracts/serving_assets.yaml`
- `tests/conftest.py`
- `app/package.json`
- `app/tsconfig.json`
- `app/vite.config.ts`
- `app/tailwind.config.ts`
- `src/ingest/__init__.py`
- `src/ingest/m5_downloader.py`
- `src/ingest/weather_fetcher.py`
- `src/validation/__init__.py`
- `src/validation/contracts.py`
- `src/transform/__init__.py`
- `src/transform/pipeline.py`
- `src/classification/__init__.py`
- `src/classification/demand_classifier.py`
- `src/features/__init__.py`
- `src/features/feature_store.py`
- `src/models/__init__.py`
- `src/models/training.py`
- `src/reconciliation/__init__.py`
- `src/reconciliation/reconciler.py`
- `src/evaluation/__init__.py`
- `src/evaluation/backtesting.py`
- `src/inventory/__init__.py`
- `src/inventory/engine.py`
- `src/export/__init__.py`
- `src/export/serving_exporter.py`
- `src/utils/__init__.py`
- `src/utils/paths.py`
- Directorios: `configs/`, `contracts/`, `data/*`, `monitoring/`, `notebooks/`, `sql/*`, `src/*`, `tests/*`, `app/*`, `.github/workflows/` (muchos vacíos pero creados para estructura).
- `docs/CONTEXT.md` (bump de versión y tablas actualizadas)
- `docs/SESSION_SNAPSHOT.md` (esta entrada)

**Comandos de verificación:**
```bash
cd retail-demand-intelligence

# Ver estructura de carpetas clave
ls
ls configs contracts src tests app data sql .github

# Ver que los YAML existen
ls configs
ls contracts

# Ver que el frontend tiene configuración básica
cd app
cat package.json

# Desde la raíz, comprobar que los targets del Makefile existen
make -pRrq -f Makefile : 2>/dev/null | grep -E '^(setup|ingest|validate_bronze|transform_silver|validate_silver|classify_demand|build_features|validate_features|train_baselines|train_lgbm|evaluate|generate_inventory|export_serving|build_frontend|deploy):' || true
```

**CONTEXT.md actualizado:** Sí — versión a v0.2.0, V1-Fase 0 marcada como completada, estado de archivos clave actualizado.

**Estado después:** V1-Fase 0 (Scaffolding) completada. El repo tiene estructura senior completa (carpetas, configs, contracts, Makefile, esqueletos `src/`, scaffolding frontend). Listo para iniciar V1-Fase 1 (Data Engineering) con implementación real de ingestión y capas bronze/silver.

---

### Session 2 — 2026-03-15 — Claude Code (Sonnet 4.6) — V1-Fase 1: Data Engineering

**Estado anterior:** V1-Fase 0 completada. Esqueletos `src/ingest/`, `src/validation/`, `src/transform/` con `NotImplementedError`. M5 zip local en `data/raw/m5/`. Sin tests.

**Qué se hizo:**
- Implementado `src/ingest/m5_downloader.py`: extrae el zip M5 local a `data/raw/m5/` (idempotente, SHA-256 por archivo, maneja subdirectorios en zip).
- Implementado `src/ingest/weather_fetcher.py`: llama Open-Meteo Archive API para CA/TX/WI, guarda CSV por estado en `data/raw/weather/`, idempotente, retry automático, httpx como cliente inyectable para tests.
- Implementado `src/ingest/bronze_writer.py` (nuevo): convierte CSVs M5 y weather a Parquet en `data/bronze/` con checksums SHA-256 en `checksums.json`, usa Polars (ADR-001).
- Implementado `src/validation/contracts.py`: schemas Pandera/Polars para bronze (BronzeSalesSchema, BronzeCalendarSchema, BronzePricesSchema, BronzeWeatherSchema) y silver (SilverSalesSchema, SilverPricesDailySchema, SilverCalendarSchema, SilverWeatherSchema). Función `load_contract()` para leer YAMLs. Registro genérico `validate_dataframe()`.
- Implementado `src/transform/pipeline.py`: 4 transformaciones independientes + orquestador:
  - `sales_wide_to_long()`: M5 wide→long vía `unpivot`, join con calendar para fechas reales, `id = item_id + "_" + store_id`, `is_zero_sale`, output particionado por `state_id/year` (Hive-style).
  - `prices_weekly_to_daily()`: join con calendar wm_yr_wk→date, forward-fill dentro de (store_id, item_id), sin leakage (solo ffill hacia adelante).
  - `enrich_calendar()`: añade `is_weekend` (M5 wday 1=Sab, 2=Dom) y `quarter`, castea date a `pl.Date`.
  - `build_silver_weather()`: castea date a `pl.Date`, coerce numéricos.
  - `run_bronze_to_silver()`: orquestador completo, idempotente con `force=False`.
- Tests unitarios: 85 tests en 4 archivos — **85/85 passing**, 0 fallos.

**Archivos creados/modificados:**
- `src/ingest/m5_downloader.py` (reimplementado)
- `src/ingest/weather_fetcher.py` (reimplementado)
- `src/ingest/bronze_writer.py` (nuevo)
- `src/validation/contracts.py` (reimplementado)
- `src/transform/pipeline.py` (reimplementado)
- `tests/unit/test_m5_downloader.py` (nuevo, 11 tests)
- `tests/unit/test_weather_fetcher.py` (nuevo, 10 tests)
- `tests/unit/test_bronze_writer.py` (nuevo, 15 tests)
- `tests/unit/test_contracts.py` (nuevo, 23 tests)
- `tests/unit/test_pipeline.py` (nuevo, 26 tests)
- `docs/CONTEXT.md` (bump a v0.3.0)
- `docs/SESSION_SNAPSHOT.md` (esta entrada)

**Comandos de verificación:**
```bash
cd retail-demand-intelligence

# Ejecutar todos los tests unitarios (debe dar 85 passed, 0 failed)
pytest tests/unit/ -v

# Verificar que los módulos importan sin errores
python -c "from src.ingest.m5_downloader import extract_m5_zip; print('m5_downloader OK')"
python -c "from src.ingest.weather_fetcher import fetch_weather_data; print('weather_fetcher OK')"
python -c "from src.ingest.bronze_writer import write_m5_bronze; print('bronze_writer OK')"
python -c "from src.validation.contracts import SilverSalesSchema; print('contracts OK')"
python -c "from src.transform.pipeline import run_bronze_to_silver; print('pipeline OK')"

# Ejecutar la extracción M5 real (requiere zip en data/raw/m5/)
python -c "
from pathlib import Path
from src.ingest.m5_downloader import extract_m5_zip
files = extract_m5_zip(Path('data/raw/m5/m5-forecasting-accuracy.zip'), Path('data/raw/m5/'))
print('Extracted:', [f.name for f in files])
"
```

**CONTEXT.md actualizado:** Sí — versión a v0.3.0, V1-Fase 1 marcada como completada, V1-Fase 2 como fase actual, dataset M5 y weather marcados como implementados.

**Estado después:** V1-Fase 1 (Data Engineering) completada. Pipeline completo raw→bronze→silver implementado con 85 tests unitarios passing. Listo para V1-Fase 2 (Classification + Features).

---

### Session 3 — 2026-03-15 — Claude Code (Sonnet 4.6) — V1-Fase 2: Classification + Features

**Estado anterior:** V1-Fase 1 completada (85 tests). V1-Fase 2 pendiente.

**Qué se hizo:**
- Implementado `src/classification/demand_classifier.py`: `compute_adi()`, `compute_cv2()` (varianza poblacional), `classify_demand()` (4 clases: smooth/erratic/intermittent/lumpy via ADI<1.32 y CV²<0.49), `classify_all_series()` en batch con Polars group_by/agg.
- Implementado `src/classification/abc_xyz.py`: `compute_abc()` con Pareto correcto usando prev_cum_pct (cumsum − rev) / total; `compute_xyz()` con CV = std/mean de totales diarios por ítem; `enrich_with_abc_xyz()` orquestador; `save_full_classification()`.
- Implementado `src/features/lag_features.py`: lags [1,7,14,21,28,56,91,364] con `shift().over(id)`, respeta cutoff_date sin leakage.
- Implementado `src/features/rolling_features.py`: ventanas [7,14,28,56,91], estadísticos mean/std/median/min/max; extras siempre calculados: rolling_zero_pct_28, rolling_cv_28, ratio_mean_7_28, ratio_mean_28_91 (se aseguran las ventanas base aunque no estén en la lista de windows).
- Implementado `src/features/calendar_features.py`: day_of_week, day_of_month, week_of_year, month, quarter, is_weekend, is_month_start/end, snap_active (state-specific), days_to_next_event, days_since_last_event.
- Implementado `src/features/price_features.py`: sell_price, price_delta_wow, price_ratio_vs_rolling_28, price_nunique_last_52w (proxy rolling_std 364d), is_price_drop_gt_10pct, relative_price_in_dept.
- Implementado `src/features/intermittency_features.py`: pct_zero_last_28d, pct_zero_last_91d, days_since_last_sale (forward_fill), streak_zeros (cumsum trick), demand_intervals_mean (ADI), non_zero_demand_mean, burstiness (B=(σ−μ)/(σ+μ) de inter-arrival intervals) via Python loop sobre grupos.
- Implementado `src/features/weather_features.py`: join por (date, state_id), renombrado de variables Open-Meteo, temp_anomaly_vs_30d_avg.
- Implementado `src/features/interaction_features.py`: `LeaveOneOutEncoder` (fit/fit_transform con LOO+ruido gaussiano+smoothing, transform para test sin LOO, manejo de categorías no vistas vía `replace_strict(default=global_mean)`); `add_interaction_features()` (price×is_weekend, snap×dept_index).
- Implementado `src/features/leakage_guard.py`: 6 reglas (no future prices, no binary calendar look-aheads, no future weather, no future target in rolling, no target encoding on test fold, macro publication lag 45d); `check_all_rules()` devuelve `(bool, list[LeakageViolation])`.
- Implementado `src/features/feature_store.py`: orquestador que aplica las 7 familias en orden, respeta cutoff_date, escribe Parquet versionado.
- Tests: `tests/unit/test_demand_classifier.py` (30 tests), `tests/unit/test_leakage_guard.py` (24 tests), `tests/unit/test_features.py` (40 tests) — 186/186 passing, 0 fallos.
- Bugs corregidos durante la sesión: ABC prev_cum_pct fix, burstiness con Python loop (no map_elements), lag null dtype Int64 (no Float64), rolling extra exprs con ventanas base forzadas, LeaveOneOutEncoder unseen categories via replace_strict.

**Archivos creados/modificados:**
- `src/classification/demand_classifier.py` (reimplementado)
- `src/classification/abc_xyz.py` (nuevo)
- `src/features/lag_features.py` (nuevo)
- `src/features/rolling_features.py` (nuevo)
- `src/features/calendar_features.py` (nuevo)
- `src/features/price_features.py` (nuevo)
- `src/features/intermittency_features.py` (nuevo)
- `src/features/weather_features.py` (nuevo)
- `src/features/interaction_features.py` (nuevo)
- `src/features/leakage_guard.py` (nuevo)
- `src/features/feature_store.py` (reimplementado)
- `tests/unit/test_demand_classifier.py` (nuevo, 30 tests)
- `tests/unit/test_leakage_guard.py` (nuevo, 24 tests)
- `tests/unit/test_features.py` (nuevo, 40 tests)
- `docs/CONTEXT.md` (bump a v0.4.0)
- `docs/SESSION_SNAPSHOT.md` (esta entrada)

**Comandos de verificación:**
```bash
cd retail-demand-intelligence

# Todos los tests — debe dar 186 passed, 0 failed
pytest tests/ -q

# Solo clasificación
pytest tests/unit/test_demand_classifier.py -v

# Solo leakage guard
pytest tests/unit/test_leakage_guard.py -v

# Solo features
pytest tests/unit/test_features.py -v

# Import smoke test
python -c "from src.features.feature_store import build_feature_store; print('feature_store OK')"
python -c "from src.features.leakage_guard import check_all_rules; print('leakage_guard OK')"
python -c "from src.classification.abc_xyz import enrich_with_abc_xyz; print('abc_xyz OK')"
```

**CONTEXT.md actualizado:** Sí — versión a v0.4.0, V1-Fase 2 marcada como completada (186 tests), V1-Fase 3 como fase actual.

**Estado después:** V1-Fase 2 (Classification + Features) completada. 186 tests unitarios passing. Feature store completo con 7 familias de features, clasificación ADI/CV²+ABC/XYZ, y leakage guard con 6 reglas. Listo para V1-Fase 3 (Modeling + Backtesting).

---

### Session 4 — 2026-03-15 — Claude Code (Sonnet 4.6) — V1-Fase 3: Modeling + Backtesting

**Estado anterior:** V1-Fase 2 completada (186 tests). V1-Fase 3 pendiente.

**Qué se hizo:**
- Implementado `src/evaluation/metrics.py`: MAE, RMSE, sMAPE (con guard div-by-zero), Bias, WRMSSE (naive-forecast scale, revenue-proxy weights), Coverage@80, Pinball Loss (por cuantil), mean_pinball_loss (p10/p50/p90 promediado), compute_all_metrics.
- Implementado `src/models/baselines.py`: interfaz unificada sobre statsforecast 2.x — SeasonalNaive (season_length=7), MovingAverage28 (WindowAverage), Croston (CrostonClassic), TSB (Teunter-Syntetos-Babai); función `run_baselines()` acepta lista de modelos, output formato canónico (date, store_id, item_id, forecast_p50, model_name).
- Implementado `src/models/training.py`: `LightGBMTrainer` global — punto (regression/MAE) + cuantiles p10/p50/p90 (objective=quantile); `TrainedModels` dataclass con `save()` / `load()`; early_stopping en val set (últimos 28 días del fold); label-encoding de categoricals; feature importance JSON; MLflow logging opcional; hiperparámetros de spec (n_estimators=2000, lr=0.03, num_leaves=255, etc.).
- Implementado `src/models/predict.py`: `predict()` genera p10/p50/p90, `enforce_monotonicity()` asegura p10≤p50≤p90 y no-negatividad, `generate_forecast_horizon()` extrae filas futuras del feature store.
- Implementado `src/evaluation/backtesting.py`: `FoldDefinition` frozen dataclass con fechas exactas de M5 (d_1773=2015-12-06 ... d_1913=2016-04-24); `FOLDS` lista de 5 folds; `run_fold()` — train/predict/evaluate por fold; `run_backtesting()` orquestador con opción `n_sample` para subset rápido (500 series); integración MLflow por fold; `_compute_wrmsse_from_df()` con pesos por revenue proxy; `summarize_backtesting()` retorna mean/std por métrica.
- Implementado `src/evaluation/segmented_report.py`: `generate_segmented_report()` — métricas por cat_id, dept_id, store_id, state_id, demand_class, abc_class + overall; `aggregate_fold_reports()` — mean/std across folds; `save_segmented_report()` — Parquet + model_metrics.json para frontend.
- Tests: `tests/unit/test_metrics.py` (29 tests), `tests/unit/test_baselines.py` (12 tests), `tests/integration/test_training.py` (29 tests) — **256/256 passing, 0 fallos**.
- Verificaciones clave de tests: fold dates exactas (Fold 1: train_cutoff=2015-12-06), horizonte=28 para todos los folds, contiguidad de folds, monotonicity enforcement p10≤p50≤p90, non-negativity, save/load roundtrip, segmented report overhead row.

**Archivos creados/modificados:**
- `src/evaluation/metrics.py` (nuevo)
- `src/models/baselines.py` (nuevo)
- `src/models/training.py` (reimplementado)
- `src/models/predict.py` (nuevo)
- `src/evaluation/backtesting.py` (reimplementado)
- `src/evaluation/segmented_report.py` (nuevo)
- `tests/unit/test_metrics.py` (nuevo, 29 tests)
- `tests/unit/test_baselines.py` (nuevo, 12 tests)
- `tests/integration/test_training.py` (nuevo, 29 tests)
- `docs/CONTEXT.md` (bump a v0.5.0)
- `docs/SESSION_SNAPSHOT.md` (esta entrada)

**Comandos de verificación:**
```bash
cd retail-demand-intelligence

# Todos los tests — debe dar 256 passed, 0 failed
pytest tests/ -q

# Solo métricas
pytest tests/unit/test_metrics.py -v

# Solo baselines (requiere statsforecast)
pytest tests/unit/test_baselines.py -v

# Solo integración (LightGBM training smoke test)
pytest tests/integration/test_training.py -v

# Import smoke tests
python -c "from src.evaluation.metrics import compute_all_metrics; print('metrics OK')"
python -c "from src.models.baselines import run_baselines; print('baselines OK')"
python -c "from src.models.training import train_lgbm; print('training OK')"
python -c "from src.evaluation.backtesting import FOLDS; print('folds:', [f.train_cutoff for f in FOLDS])"
```

**CONTEXT.md actualizado:** Sí — versión a v0.5.0, V1-Fase 3 marcada como completada (256 tests), V1-Fase 4 como fase actual.

**Estado después:** V1-Fase 3 (Modeling + Backtesting) completada. 256 tests unitarios e integración passing. Pipeline completo: baselines + LightGBM global + quantile + backtesting 5-fold + metrics + segmented report + MLflow logging. Listo para V1-Fase 4 (Inventory + Export).

---

### Session 5 — 2026-03-15 — Claude Code (Sonnet 4.6) — V1-Fase 4: Inventory + Export

**Estado anterior:** V1-Fase 3 completada (256 tests). V1-Fase 4 pendiente.

**Qué se hizo:**

**PARTE A — Inventory:**
- Implementado `src/inventory/synthetic_generator.py`: genera parámetros sintéticos por SKU-store desde clasificación ADI/CV²+ABC — initial_stock=round(avg_demand×30), lead_time_days por LogNormal(μ, σ) por categoría (FOODS μ=ln(7)/σ=0.3, HOUSEHOLD μ=ln(10)/σ=0.35, HOBBIES μ=ln(14)/σ=0.4), service_level_target por ABC (A=0.98/B=0.95/C=0.90), holding_cost_pct por cat (FOODS=0.25/HOUSEHOLD=0.20/HOBBIES=0.15), stockout_cost_mult por ABC (A=5.0/B=3.0/C=1.5). Todas las columnas etiquetadas SYNTHETIC. Seed fijo 42.
- Implementado `src/inventory/safety_stock.py`: `z_score()` vía scipy.stats.norm.ppf; `safety_stock_classic()` = z(SL)×σ×√(LT); `safety_stock_quantile()` = max(0, p90_sum − p50_sum); `safety_stock_quantile_from_arrays()` para arrays diarios; `compute_safety_stock_batch()` batch con fallback classic→quantile.
- Implementado `src/inventory/reorder_point.py`: `reorder_point()` = expected_demand + SS; `reorder_point_from_arrays()` con truncado a LT días; `reorder_point_avg_demand()` proxy cuando no hay forecast; `compute_reorder_points_batch()` batch.
- Implementado `src/inventory/engine.py`: `simulate_series()` — política (s,S) determinista 90 días, un orden pendiente máximo, fulfilled=min(stock, demand), lost=demand−fulfilled, métricas fill_rate/stockout_days/days_of_supply/avg_inventory; `run_inventory_simulation()` batch con optional forecast extension; `save_inventory_snapshot()` a Parquet; todas las salidas etiquetadas SYNTHETIC.

**PARTE B — Export:**
- Implementado `src/export/serving_exporter.py`: `build_executive_summary()` — revenue_proxy_total, by_state, by_category, monthly_trend, fill_rate_avg, stockout_rate, inventory_value_total, days_of_supply_avg, n_skus/n_stores; `build_forecast_series()` — agrega a state×category, últimos 180 días actuals + forecast; `build_inventory_risk_matrix()` — por store×dept: fill_rate, stockout_prob, days_of_supply, n_items_at_risk, overstock_flag; `export_serving_assets()` orquestador; `_build_manifest()` con sha256+sizes; `_make_minimal_synthetic_sales()` fallback sintético.
- Assets generados en data/gold/serving/: 13 archivos, **57 KB total** (budget: 5 MB — uso <1.2%).

**Tests:** `tests/unit/test_safety_stock.py` (28 tests), `tests/unit/test_inventory_engine.py` (17 tests), `tests/integration/test_export.py` (23 tests) — **324/324 passing, 0 fallos**.

**Archivos creados/modificados:**
- `src/inventory/synthetic_generator.py` (nuevo)
- `src/inventory/safety_stock.py` (nuevo)
- `src/inventory/reorder_point.py` (nuevo)
- `src/inventory/engine.py` (reimplementado)
- `src/export/serving_exporter.py` (reimplementado)
- `data/gold/serving/` (13 archivos JSON: executive_summary, 9×forecast_series, inventory_risk_matrix, model_metrics, asset_manifest)
- `tests/unit/test_safety_stock.py` (nuevo, 28 tests)
- `tests/unit/test_inventory_engine.py` (nuevo, 17 tests)
- `tests/integration/test_export.py` (nuevo, 23 tests)
- `docs/CONTEXT.md` (bump a v0.6.0)
- `docs/SESSION_SNAPSHOT.md` (esta entrada)

**Tamaño data/gold/serving/:**
```
executive_summary.json          0.75 KB
forecast_series_CA_FOODS.json   6.04 KB  (×9 archivos state×cat)
inventory_risk_matrix.json      0.09 KB
model_metrics.json              0.03 KB
asset_manifest.json             1.81 KB
TOTAL: 57.01 KB (0.056 MB) — 1.14% del budget de 5 MB
```

**Comandos de verificación:**
```bash
cd retail-demand-intelligence

# Todos los tests — debe dar 324 passed, 0 failed
pytest tests/ -q

# Solo inventario
pytest tests/unit/test_safety_stock.py tests/unit/test_inventory_engine.py -v

# Solo export
pytest tests/integration/test_export.py -v

# Generar assets y ver tamaños
python -c "
from src.export.serving_exporter import export_serving_assets, _make_minimal_synthetic_sales
from pathlib import Path
sales = _make_minimal_synthetic_sales(n_days=365)
paths = export_serving_assets(Path('data/gold/serving'), sales_df=sales, is_synthetic=True)
total = sum(p.stat().st_size for p in paths.values())
print(f'Total: {total/1024:.1f} KB')
"
```

**CONTEXT.md actualizado:** Sí — versión a v0.6.0, V1-Fase 4 marcada como completada (324 tests), V1-Fase 5 (Frontend) como fase actual.

**Estado después:** V1-Fase 4 (Inventory + Export) completada. 324 tests passing. Backend V1 completamente implementado: Data Engineering + Classification + Features + Modeling + Backtesting + Inventory + Export. Listo para V1-Fase 5 (Frontend + Deploy con Cursor).

---

### Session 8 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V2-Fase 1: Reconciliación Jerárquica

**Estado anterior:** V1 completada (5 fases, 324 tests). V2-Fase 1 marcada como bloqueada por V1, ahora desbloqueada.

**Qué se hizo:**

1. **Instalado `hierarchicalforecast==1.5.1`** (+ dependencias: clarabel, qpsolvers).
   - Verificada API: `BottomUp()`, `TopDown(method="average_proportions")`, `MinTrace(method="mint_shrink")`.
   - Nota: `TopDown(average_proportions)` es in-sample (`insample=True`) y requiere `Y_df`; `MinTraceShrink` renombrado a `MinTrace(method="mint_shrink")` en esta versión.

2. **`src/reconciliation/hierarchy.py`** — Nuevo módulo completo:
   - `build_hierarchy_matrix(catalog_df)` → `HierarchyMatrix(S_df, tags)`.
   - S matrix: scipy sparse → `pd.DataFrame.sparse.from_spmatrix` (evita materializar 10 GB densos).
   - 12 niveles M5 (Total → SKU-Store), 42.840 series totales, 30.490 bottom.
   - `build_sub_hierarchy(catalog_df, group_cols)` para MinTrace particionado.
   - `get_level_for_series(uid, tags)`.
   - IDs bottom: `{item_id}_{store_id}`; IDs aggregados: separador `/`.

3. **`src/reconciliation/reconciler.py`** — Reescritura completa del placeholder:
   - `reconcile_forecasts(Y_hat_df, S_df, tags, method, Y_df, model_col)` — BU / TD / MinT.
   - `reconcile_all_methods(...)` — BU siempre; TD+MinT solo si `Y_df` provisto.
   - `reconcile_mint_sub_hierarchy(...)` — MinTrace por grupos (store×dept) para escalabilidad.
   - `aggregate_base_forecasts(bottom_df, catalog_df, tags)` — sube bottom hasta Total.
   - `run_reconciliation_backtest(sales_df, forecast_df, catalog_df, ...)` — 5-fold BU vs TD vs MinT, guarda `reconciliation_comparison.parquet`.

4. **`src/reconciliation/evaluate_reconciliation.py`** — Nuevo módulo:
   - `check_bottom_up_coherence(...)` — Test 1: sumas bottom == aggregados (atol=0.01).
   - `check_non_negativity(...)` — Test 2: sin valores negativos.
   - `check_quantile_monotonicity(...)` — Test 3: p10 ≤ p50 ≤ p90 (skip si faltan columnas).
   - `check_daily_total_coherence(...)` — Test 4: suma diaria bottom == Total (atol=1.0).
   - `run_coherence_tests(reconciled_df, S_df, tags)` → dict con las 4 pruebas + `all_passed`.
   - Nota: funciones nombradas `check_*` (no `test_*`) para evitar colisión con pytest.

5. **Tests:** 59 nuevos tests (21 + 22 + 16), todos pasando:
   - `tests/unit/test_hierarchy.py` (21 tests) — dimensiones S matrix, tags, IDs, membership, sub-hierarchies.
   - `tests/unit/test_coherence.py` (22 tests) — 4 tests pass/fail, orquestador.
   - `tests/integration/test_reconciliation.py` (16 tests) — smoke end-to-end: aggregate, BU/TD/MinT, sub-hierarchy, coherence post-reconciliation.

6. **`pyproject.toml`** — Añadido `[tool.pytest.ini_options]` con `testpaths = ["tests"]`.

**Archivos cambiados:**
- `src/reconciliation/hierarchy.py` (nuevo)
- `src/reconciliation/reconciler.py` (reescritura del placeholder)
- `src/reconciliation/evaluate_reconciliation.py` (nuevo)
- `tests/unit/test_hierarchy.py` (nuevo)
- `tests/unit/test_coherence.py` (nuevo)
- `tests/integration/test_reconciliation.py` (nuevo)
- `pyproject.toml` (añadido `[tool.pytest.ini_options]`)
- `docs/CONTEXT.md` (bump a v0.8.0)

**Verificar:**
```bash
pytest tests/unit/test_hierarchy.py tests/unit/test_coherence.py tests/integration/test_reconciliation.py -v
# → 59 passed

pytest tests/ -v
# → 383 passed
```

**CONTEXT.md actualizado:** Sí — versión a v0.8.0, V2-Fase 1 marcada como completada (383 tests), V2-Fase 2 como próxima.

**Estado después:** V2-Fase 1 (Reconciliación Jerárquica) completada. 383 tests passing. Módulos: hierarchy.py (S matrix 12 niveles M5), reconciler.py (BU/TD/MinT + sub-jerarquía + backtest), evaluate_reconciliation.py (4 coherence checks). Listo para V2-Fase 2 (Conformal + SHAP).

---

### Session 9 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V2-Fase 2: Conformal Prediction + SHAP

**Estado anterior:** V2-Fase 1 completada (383 tests, jerarquía + reconciliación). V2-Fase 2 desbloqueada.

**Qué se hizo:**

**PARTE A — Conformal Prediction:**

1. **`src/models/conformal.py`** — Nuevo módulo completo:
   - `ConformalCalibrator` dataclass: adj_p10, adj_p90, achieved_coverage, n_calibration, coverage_within_target, meta.
   - `fit_conformal(actuals, p10, p50, p90)` → ConformalCalibrator:
     - residuales: actual − forecast_q
     - adj_p10 = quantile(residuals_p10, 0.10); adj_p90 = quantile(residuals_p90, 0.90)
     - Iterative widening/narrowing: WIDEN_STEP=0.01, MAX_ITER=50 hasta 75%–85% coverage
   - `calibrate(p10, p50, p90, calibrator)` → (cal_p10, cal_p50, cal_p90):
     - Monotonicity: p10≤p50≤p90 enforced post-adjustment
     - Non-negativity: clip a 0
   - `evaluate_coverage(actuals, cal_p10, cal_p90)` → dict con coverage, interval_width, pct_below/above
   - `run_conformal_calibration(...)` — full pipeline: fit + calibrate + save coverage_report.json + MLflow logging

**PARTE B — SHAP Analysis:**

2. **`src/evaluation/shap_analysis.py`** — Nuevo módulo completo:
   - `compute_shap_values(model, X_sample)` → np.ndarray via `shap.TreeExplainer` (instalado: shap==0.51.0)
   - `generate_shap_summary(shap_values, feature_names, top_n=30)` → dict con top_features (ranked, mean_abs≥0), all_features, n_samples, n_features
   - `shap_by_segment(shap_values, feature_names, segment_series)` → dict por segmento (cat_id, demand_class)
   - `export_shap_for_frontend(summary, by_cat, by_demand, output_dir)` → shap_summary.json (<50KB) + shap_by_segment.json
   - `run_shap_analysis(model_dir, feature_df, output_dir, sample_size=10000)` — pipeline completo con MLflow artifact logging

3. **`src/export/serving_exporter.py`** — Actualizado:
   - `export_serving_assets` acepta nuevos parámetros: `shap_summary=None`, `coverage_report=None`
   - Escribe shap_summary.json y coverage_report.json cuando se proporcionan
   - asset_manifest.json los incluye automáticamente

**TESTS:** 59 nuevos tests (24 unit conformal + 24 unit shap + 11 integration), todos pasando:
- `tests/unit/test_conformal.py` (24 tests) — ConformalCalibrator, fit/calibrate/evaluate, iterative widening, pipeline
- `tests/unit/test_shap.py` (24 tests) — compute_shap_values, generate_shap_summary, shap_by_segment, export
- `tests/integration/test_conformal_pipeline.py` (11 tests) — E2E conformal+SHAP+serving_exporter

**Archivos cambiados:**
- `src/models/conformal.py` (nuevo)
- `src/evaluation/shap_analysis.py` (nuevo)
- `src/export/serving_exporter.py` (añadidos parámetros shap_summary + coverage_report)
- `tests/unit/test_conformal.py` (nuevo)
- `tests/unit/test_shap.py` (nuevo)
- `tests/integration/test_conformal_pipeline.py` (nuevo)
- `docs/CONTEXT.md` (bump a v0.9.0)

**Paquetes instalados:** `shap==0.51.0` (+ slicer, numpy actualizado)

**Verificar:**
```bash
pytest tests/unit/test_conformal.py tests/unit/test_shap.py tests/integration/test_conformal_pipeline.py -v
# → 59 passed

pytest tests/ -v
# → 442 passed
```

**CONTEXT.md actualizado:** Sí — versión a v0.9.0, V2-Fase 2 marcada como completada (442 tests), V2-Fase 3 (Monte Carlo + Scenarios) como próxima.

**Estado después:** V2-Fase 2 (Conformal + SHAP) completada. 442 tests passing. Módulos: conformal.py (calibración post-hoc con iterative widening, target coverage 80%±5%), shap_analysis.py (TreeExplainer top-30, segmentación por cat_id/demand_class), serving_exporter actualizado con shap_summary.json + coverage_report.json. Listo para V2-Fase 3 (Monte Carlo + Scenarios).

---

### Session 10 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V2-Fase 3: Monte Carlo + Scenarios + Policies

**Estado anterior:** V2-Fase 2 completada (442 tests). simulator.py recién creado al final de la sesión anterior sin tests.

**Qué se hizo:**

**PARTE A — Monte Carlo Simulator:**
- `src/inventory/simulator.py` — Motor Monte Carlo completo con NumPy vectorizado:
  - `MonteCarloResult` dataclass: fill_rate (mean/p5/p95/distribution), stockout_probability, expected_stockout_days, avg_inventory_mean, total_cost (mean/p5/p95/distribution), daily_stock_percentiles (5×90)
  - `_sample_demands(p10, p50, p90, n_sims, rng)` — CDF piecewise-linear con 5 puntos de anclaje, bucle de 90 días con `np.interp`
  - `simulate_inventory_mc(...)` — LogNormal lead times, pool pre-muestreado `(n_sims, max_orders)`, bucle vectorizado (pending_qty/pending_arr/order_cnt arrays), política (s,Q)
  - `run_mc_batch(series_list, n_simulations, horizon_days, n_jobs)` — runner secuencial con soporte optional joblib
- **Tests:** `tests/unit/test_monte_carlo.py` — 36 tests: régimen 0-stockout, régimen alto stockout, reproducibilidad, edge cases, batch runner

**PARTE B — Newsvendor:**
- `src/inventory/newsvendor.py`:
  - `optimal_newsvendor_quantity(forecast_quantiles, cu, co)` → Q* = F⁻¹(Cu/(Cu+Co))
  - `economic_order_quantity(annual_demand, order_cost, holding_cost)` → EOQ = √(2DS/H)
  - `safety_stock(demand_std, lead_time, service_level)` → z(SL)×σ×√LT
  - `reorder_point(mean_daily, lead_time, demand_std, service_level)` → ROP = demand_LT + SS
  - `run_newsvendor_analysis(...)` → NewsvendorResult completo
  - `compare_by_abc_segment(...)` → dict con resultados para A(95%), B(90%), C(85%)
- **Tests:** `tests/unit/test_newsvendor.py` — 43 tests: cu=co → Q*=median, cu>>co → Q*→p90, SS fórmula, comparación ABC

**PARTE C — Policy Comparator:**
- `src/inventory/policy_comparator.py`:
  - 4 políticas: (s,Q) EOQ, (s,S) order-up-to, (R,S) periodic review 7 días, SL-driven
  - `compare_policies(...)` → `PolicyComparisonResult` con best_policy (min cost)
  - `run_policy_comparison_batch(...)` → guarda `policy_comparison.parquet`
  - `build_policy_comparison_summary(...)` → dict JSON-serializable
- **Tests:** `tests/unit/test_policies.py` — 27 tests: contrato, rangos válidos, comportamiento, batch, resumen

**PARTE D — Scenario Engine:**
- `src/inventory/scenario_engine.py`:
  - 5 escenarios (spec §10.4): demand_surge (+30%), lead_time_delay (+5d), cost_increase (+15%), high_service (95%→99%), combined_stress (+20%/+3d)
  - `run_scenario_engine(...)` → `ScenarioEngineResult` con baseline + 5 `ScenarioResult` con deltas
  - `run_scenario_batch(...)` → batch sobre lista de series
  - `build_scenario_results_json(...)` → dict conciso < 200 KB
  - `export_scenario_results(...)` → escribe `scenario_results.json`
- **Tests:** `tests/unit/test_scenarios.py` — 34 tests + `tests/integration/test_scenario_pipeline.py` — 11 tests (incluye serving_exporter integration)

**serving_exporter actualizado:**
- `src/export/serving_exporter.py` — añadidos 2 parámetros opcionales:
  - `policy_comparison_summary: dict | None` → `policy_comparison_summary.json`
  - `scenario_results: dict | None` → `scenario_results.json`
  - Ambos incluidos en `asset_manifest.json`

**Archivos creados:**
- `src/inventory/simulator.py` (creado sesión anterior, testeado en esta)
- `src/inventory/newsvendor.py`
- `src/inventory/policy_comparator.py`
- `src/inventory/scenario_engine.py`
- `tests/unit/test_monte_carlo.py`
- `tests/unit/test_newsvendor.py`
- `tests/unit/test_policies.py`
- `tests/unit/test_scenarios.py`
- `tests/integration/test_scenario_pipeline.py`

**Archivos modificados:**
- `src/export/serving_exporter.py` (2 nuevos parámetros opcionales + manifest entries)
- `docs/CONTEXT.md` (bump a v1.0.0, V2-Fase 3 marcada como completada)
- `docs/SESSION_SNAPSHOT.md` (esta entrada)

**Performance:** 500 series × 1000 sims × 90 días → **2.92s total / 5.8 ms por serie** (CPU single-thread, NumPy vectorizado)

**Comandos de verificación:**
```bash
cd retail-demand-intelligence
pytest tests/unit/test_monte_carlo.py tests/unit/test_newsvendor.py tests/unit/test_policies.py tests/unit/test_scenarios.py tests/integration/test_scenario_pipeline.py -v
pytest tests/ -q  # 593 tests should pass
```

**CONTEXT.md actualizado:** Sí — versión a v1.0.0, V2-Fase 3 marcada como completada (593 tests), timing MC reportado.

**Estado después:** V2-Fase 3 (Monte Carlo + Scenarios + Policies) completada. 593 tests passing. Módulos: simulator.py (MC vectorizado, 5.8 ms/serie), newsvendor.py (Q*=F⁻¹(CR), EOQ, SS, ROP, ABC comparison), policy_comparator.py (4 políticas, parquet export), scenario_engine.py (5 what-ifs, JSON < 200 KB), serving_exporter actualizado. Listo para V2-Fase 4 (Frontend +3 páginas) o V2-Fase 5 (MLflow + Model Cards).

---

### Session 12 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V2-Fase 5: MLflow completo + Model Cards

**Estado anterior:** V2-Fases 1-4 completadas (593 tests). MLflow logging parcial en training.py (solo params/val_mae/val_rmse). Sin model cards, sin decision log.

**Qué se hizo:**

**MLflow logging completo en `src/models/training.py`:**
- Extendida firma de `train_lgbm` con 9 nuevos parámetros keyword-only: `feature_set_version`, `fold_id`, `forecast_horizon`, `n_series`, `dataset_sha256`, `is_baseline`, `is_reconciled`, `demand_class_distribution`, `log_artifacts`
- Bloque MLflow mejorado: params completos (feature_set_version, fold_id, forecast_horizon, n_series), métricas val completas (MAE, RMSE, sMAPE, Bias, best_iteration, Coverage@80, Pinball Loss), tags (is_baseline, is_reconciled, dataset_sha256, demand_class_distribution como JSON), artifact logging (modelos .lgb + feature_importance.json en tmpdir)
- Añadida función `log_conformal_calibration(run, coverage_before, coverage_after, adjustment_p10, adjustment_p90)` para log post-conformal
- Añadida función `log_reconciliation_results(run, method_selected, mae_pre, mae_post, coherence_test_passed)` para log post-reconciliation

**MLflow logging completo en `src/evaluation/backtesting.py`:**
- `run_fold` extendido con 7 parámetros: feature_set_version, n_series, dataset_sha256, is_baseline, is_reconciled, demand_class_distribution, log_artifacts — todos pasados a train_lgbm
- Bloque de métricas MLflow mejorado: log fold-prefixed (fold1_mae) + unprefixed; artifact de predictions sample (top-100 series por ventas totales, Parquet)
- `run_backtesting` extendido con 6 parámetros nuevos, propagados a run_fold

**Model Cards generadas en `docs/07_model_cards/`:**
- `lgbm_global.md` — LightGBM global: detalles técnicos, intended use, training data, leakage controls, performance por demand class, limitaciones, consideraciones éticas
- `seasonal_naive.md` — Seasonal Naïve baseline: fórmula, uso, limitaciones
- `croston_tsb.md` — Croston TSB: modelo intermitente, fórmula TSB, comparativa con LightGBM por clase de demanda

**Decision Log generado en `docs/11_decision_log.md`:**
- ADR-001: Polars como procesador primario
- ADR-002: LightGBM global model strategy
- ADR-003: ADI/CV² classification
- ADR-004: Synthetic inventory layer con labelling transparente
- ADR-005: Three-release delivery (V1→V2→V3)
- ADR-006: Pre-computed JSON para frontend (no live inference)
- ADR-007: ECharts sobre Plotly
- ADR-008: MinT Shrinkage para reconciliación jerárquica
- ADR-009: Conformal prediction para uncertainty quantification
- ADR-010: Monte Carlo para inventory risk assessment

**Tests: `tests/integration/test_mlflow_logging.py` (21 tests):**
- TestLogConformalCalibration (4 tests): 4 métricas presentes, valores correctos, no-op con run=None, tolerancia a errores MLflow
- TestLogReconciliationResults (4 tests): métricas + tags presentes, valores correctos, no-op y tolerancia
- TestTrainLgbmMlflowLogging (11 tests): params core, params opcionales, val_mae/rmse, val_smape/bias, coverage_80/pinball, tags is_baseline/is_reconciled/dataset_sha256/demand_class_distribution, no-crash sin run, tolerancia a excepciones
- TestRunBacktestingMlflow (2 tests): 1 run por fold, fold params logueados

**Archivos creados/modificados:**
- `src/models/training.py` (train_lgbm mejorado + 2 funciones nuevas)
- `src/evaluation/backtesting.py` (run_fold + run_backtesting extendidos)
- `docs/07_model_cards/lgbm_global.md` (nuevo)
- `docs/07_model_cards/seasonal_naive.md` (nuevo)
- `docs/07_model_cards/croston_tsb.md` (nuevo)
- `docs/11_decision_log.md` (nuevo, ADR-001→010)
- `tests/integration/test_mlflow_logging.py` (nuevo, 21 tests)
- `docs/CONTEXT.md` (bump a v1.2.0, V2 marcado completado)
- `docs/SESSION_SNAPSHOT.md` (esta entrada)

**Comandos de verificación:**
```bash
cd retail-demand-intelligence
pytest tests/integration/test_mlflow_logging.py -v  # 21 tests pass
pytest tests/ -q  # 614 tests should pass
```

**CONTEXT.md actualizado:** Sí — versión a v1.2.0, V2-Fase 5 marcada como completada, V2 completo, V3 desbloqueado.

**Estado después:** **V2 (Technical Depth) COMPLETADO.** 614 tests passing. MLflow logging completo con params/metrics/artifacts/tags por fold. 3 model cards documentando los 3 modelos del stack. 10 ADRs en decision log. Listo para V3-Fase 1 (Monitoring + Alerting) cuando el usuario lo decida.

---

### Session 11 — 2026-03-15 — Claude Code (claude-sonnet-4-6) — V2-Fase 4: Frontend 7 páginas (rebuild completo)

**Estado anterior:** V2-Fase 3 completada (593 tests). Frontend era V1 con 4 páginas de diseño pobre.

**Qué se hizo:**

Rebuild completo del frontend desde cero con 7 páginas premium y design system completo.

**Setup:**
- Copiado `data/gold/serving/*.json` → `app/public/data/` (13 archivos)
- Actualizado `tailwind.config.ts`: tokens completos (background, primary, success, warning, danger, info, text, border), fontFamily Inter/JetBrains Mono, boxShadow (card/dropdown/modal), borderRadius completo
- Reconstruido `app/src/index.css`: Google Fonts import, scrollbar custom, select styling, focus ring, transitions, reduced-motion

**Componentes UI nuevos (src/components/ui/):**
- `KPICard.tsx` — tarjeta de KPI con acento colored border-l, valor grande, delta con TrendingUp/Down
- `SectionCard.tsx` — card contenedor con header opcional + border-b separator
- `Badge.tsx` — badges de colores (primary/success/warning/danger/neutral/info)
- `EmptyState.tsx` — placeholder para datos faltantes con SYNTHETIC label

**BaseEChart mejorado (src/components/charts/BaseEChart.tsx):**
- Añadido ScatterChart, DataZoomComponent, MarkLineComponent
- Exporta CHART_THEME (dark theme defaults) y PALETTE para reusar en páginas
- Two-effect pattern: init once + update on option change (evita re-init innecesario)

**Layout reconstruido:**
- `Sidebar.tsx`: fixed 260px, 7 nav items con iconos lucide-react, active state = border-l-2 + bg-primary-subtle + text-primary, footer con V2.0 + author
- `Header.tsx`: PAGE_TITLES map para 7 rutas, sticky con backdrop-blur
- `FilterBar.tsx`: 3 selects estilados (h-9, bg-elevated, border, transition), usa nuevas acciones del store
- `Layout.tsx`: paddingLeft:260 para sidebar fijo, max-w-content container

**App/Router reconstruido:**
- `main.tsx`: BrowserRouter wrapping
- `App.tsx`: simplificado
- `router.tsx`: lazy loading con React.lazy + Suspense para las 7 páginas

**7 páginas implementadas:**
1. `ExecutiveOverview.tsx`: 6 KPI cards (3×2 grid con accent borders), Revenue by State (bar horizontal), Revenue by Category (donut), Revenue Trend (area chart con gradient fill)
2. `ForecastLab.tsx`: state/cat selectors locales, time series chart (actual=#E5E7EB + p50=#4F8EF7 + intervalo shaded), model comparison table, bias indicator
3. `InventoryRisk.tsx`: 3 KPIs, heatmap stores×depts (con EmptyState si no hay datos), top-10 at-risk table con color coding por severidad
4. `ProductDrilldown.tsx`: store selector, placeholder cards con info sobre drilldown_{store}.json pendiente, item details card
5. `ModelQuality.tsx`: 3 KPIs, SHAP feature importance bar chart (top-20), demand class performance table (Smooth/Erratic/Intermittent/Lumpy), error distribution y calibration placeholders
6. `ScenarioLab.tsx`: 5 scenario cards (data-driven o placeholder), policy comparison table con 4 políticas
7. `About.tsx`: pipeline architecture (9 layers con colored left borders), data sources (4 con badges Real/SYNTHETIC), methodology (4 secciones), disclaimer prominente (warning border), tech stack badges, links

**Archivos modificados/creados:**
- `app/tailwind.config.ts`, `app/src/index.css`
- `app/src/main.tsx`, `app/src/App.tsx`, `app/src/router.tsx`
- `app/src/stores/filterStore.ts` (exports STATES, CATEGORIES constants)
- `app/src/types/index.ts` (TypeScript types for all JSON serving assets)
- `app/src/components/layout/Sidebar.tsx`, `Header.tsx`, `FilterBar.tsx`, `Layout.tsx`
- `app/src/components/charts/BaseEChart.tsx`
- `app/src/components/ui/KPICard.tsx`, `SectionCard.tsx`, `Badge.tsx`, `EmptyState.tsx` (nuevos)
- `app/src/pages/ExecutiveOverview.tsx`, `ForecastLab.tsx`, `InventoryRisk.tsx`, `About.tsx` (rebuilt)
- `app/src/pages/ProductDrilldown.tsx`, `ModelQuality.tsx`, `ScenarioLab.tsx` (nuevos)
- `app/public/data/*.json` (13 serving assets copiados)

**Build result:** ✅ `npm run build` — 2332 modules, 0 errors, 2.54s. Único warning: ECharts chunk 628 kB (esperado, ya preexistente).

**CONTEXT.md actualizado:** Sí — versión a v1.1.0, V2-Fase 4 marcada como completada.

**Estado después:** 7 páginas premium implementadas con design system completo (spec §13.1-13.4). Frontend listo para V2-Fase 5 (MLflow + Model Cards). Siguiente: `V2-Fase 5 MLflow + Model Cards` o deploy a Hugging Face.

---
