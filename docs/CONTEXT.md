# PROJECT CONTEXT — Source of Truth
# Version: v1.9.0
# Last updated: 2026-03-15
# Updated by: Claude Code (Session 17 — Frontend Layout Fixes)

---

## REGLA PARA TODA IA QUE LEA ESTE ARCHIVO

```
ANTES de responder cualquier petición sobre este proyecto:
1) Lee este archivo completo (docs/CONTEXT.md)
2) Lee docs/SESSION_SNAPSHOT.md si existe
3) Trata este archivo como FUENTE DE VERDAD. Si hay contradicción con algo, dilo.
4) Responde SOLO alineado con este contexto.
5) Si tu respuesta implica cambios (código/config/pipeline), AL FINAL:
   - Actualiza este archivo (bump version vX → vX+1)
   - Añade entrada en docs/SESSION_SNAPSHOT.md con lo hecho
   - Devuelve: (a) respuesta, (b) qué se actualizó, (c) comandos para verificar
```

---

## 1. Identidad del proyecto

| Campo | Valor |
|-------|-------|
| **Nombre** | Retail Demand Intelligence & Inventory Optimization Platform |
| **Alias** | AI Supply Chain Control Tower |
| **Carpeta raíz** | `retail-demand-intelligence/` |
| **Autor** | Héctor Ferrándiz Sanchis |
| **Spec maestra** | `docs/MASTER_SPEC_v3.md` (copia del documento v3 principal) |
| **Repo** | (pendiente de crear en GitHub) |
| **HF Space** | (pendiente de deploy) |
| **Objetivo** | Portfolio senior: Data Engineering + ML + Operations Research + Product |

---

## 2. Release actual

| Campo | Valor |
|-------|-------|
| **Release activo** | **V3 — Enterprise Polish** ✅ COMPLETADO |
| **Fase actual** | **V3 COMPLETADO — proyecto terminado** |
| **Estado** | 🟢 V1 + V2 + V3 completados. 795 tests passing. README + 11 docs escritos. Pre-release audit completado. Portfolio listo. |

### Progreso V1

| Fase | Estado | Herramienta | Notas |
|------|--------|-------------|-------|
| V1-0: Scaffolding | ✅ Completada | Cursor | Estructura repo, configs, Makefile, src/app skeletons |
| V1-1: Data Engineering | ✅ Completada | Claude Code | Ingest, bronze, silver, contracts — 85 tests passing |
| V1-2: Classification + Features | ✅ Completada | Claude Code | ADI/CV², ABC/XYZ, 7 feature families, leakage guard — 186 tests passing |
| V1-3: Modeling + Backtesting | ✅ Completada | Claude Code | Baselines, LightGBM global+quantile, backtesting 5-fold, metrics, segmented report, MLflow — 256 tests passing |
| V1-4: Inventory + Export | ✅ Completada | Claude Code | Synthetic params, SS/ROP, forward sim, serving JSON (57 KB / 5 MB budget) — 324 tests passing |
| V1-5: Frontend + Deploy | ✅ Completada (build local) | Cursor/Claude Code | V2: rebuilt as 7 pages premium, dark design system v2.0, lucide-react, ECharts dark-premium theme, new tailwind tokens |

### Progreso V2 (NO iniciar hasta V1 completado)

| Fase | Estado |
|------|--------|
| V2-1: Reconciliation | ✅ Completada | Claude Code | hierarchy.py (S matrix 12 niveles), reconciler.py (BU/TD/MinT+sub-jerarquía), evaluate_reconciliation.py (4 coherence checks) — 383 tests passing |
| V2-2: Conformal + SHAP | ✅ Completada | Claude Code | conformal.py (calibración post-hoc 80% coverage), shap_analysis.py (TreeExplainer top-30), serving_exporter actualizado — 442 tests passing |
| V2-3: Monte Carlo + Scenarios | ✅ Completada | Claude Code | simulator.py (1000 sims × 90 days, NumPy vectorizado, 5.8 ms/serie), newsvendor.py (Q*=F⁻¹(CR), EOQ, SS, ROP), policy_comparator.py (4 policies), scenario_engine.py (5 what-ifs), serving_exporter actualizado — 593 tests passing |
| V2-4: Frontend +3 pages | ✅ Completada | Claude Code | 7 páginas premium (ExecutiveOverview, ForecastLab, InventoryRisk, ProductDrilldown, ModelQuality, ScenarioLab, About), design system completo, lazy routing, Vite build OK |
| V2-5: MLflow + Model Cards | ✅ Completada | Claude Code | MLflow completo (params/metrics/artifacts/tags), 3 model cards, ADR-001→010, 21 tests — 614 tests passing |

### Progreso V3 (NO iniciar hasta V2 completado)

| Fase | Estado |
|------|--------|
| V3-1: Monitoring + Alerting | ✅ Completada | Claude Code | drift_detector.py (KS+PSI+zero-inflation+price-regime), performance_tracker.py (CUSUM+segments+calibration), alert_engine.py (6 rules), health_report_generator.py — 753 tests passing |
| V3-2: SCD Type 2 | ✅ Completada | Claude Code | scd_manager.py (detect_classification_changes, apply_scd_type2, save_dim_product), int_product_scd2.sql (dbt snapshot), 25 unit tests — 778 tests passing |
| V3-3: CI/CD + E2E Tests | ✅ Completada | Claude Code | .github/workflows/ci.yml (lint+test+frontend jobs), deploy_space.yml (HF Space deploy), tests/e2e/test_frontend_smoke.py (17 smoke tests), Makefile V3 targets — 795 tests passing |
| V3-4: Documentation | ✅ Completada | Claude Code | README.md (root) + docs/01_business_case.md + 02_data_architecture.md + 03_ml_system_design.md + 04_inventory_engine.md + 05_demand_classification.md + 06_reconciliation.md + 08_synthetic_data_disclaimer.md + 09_deployment_guide.md + 10_monitoring_runbook.md + 12_known_issues.md — 795 tests still passing |

---

## 3. Stack tecnológico confirmado

| Capa | Tecnología | Versión |
|------|-----------|---------|
| Python | Python | 3.11+ |
| Processing | Polars (primary), Pandas (compat) | latest |
| Data Quality | Pandera | latest |
| Warehouse | PostgreSQL + dbt-core | 16 / 1.7+ |
| ML | LightGBM, statsforecast | 4.x / latest |
| Uncertainty | LightGBM quantile + conformal (V2) | — |
| Reconciliation | hierarchicalforecast (V2) | — |
| Experiment Tracking | MLflow | 2.x |
| Frontend | Vite + React 18 + TypeScript + Tailwind | latest |
| Charts | ECharts | 5.x |
| Deploy | Hugging Face Static Space | — |

---

## 4. Dataset y datos

| Fuente | Estado | Ubicación |
|--------|--------|-----------|
| M5 Forecasting (Kaggle) | ✅ Extraido y procesado (pipeline ejecutado 2026-03-15) | data/raw/m5/, data/bronze/, data/silver/ |
| Open-Meteo Weather | ✅ Fetcher implementado (3 estados, httpx) | data/raw/weather/ |
| FRED Macro | ⬜ No descargado | data/raw/macro/ |
| Synthetic Inventory | ✅ Generado (30,490 series, synthetic_params.parquet) | data/gold/ |

---

## 5. Archivos clave del proyecto

| Archivo | Propósito | Estado |
|---------|----------|--------|
| docs/MASTER_SPEC_v3.md | Spec completa del proyecto (no editar) | ✅ Existe |
| docs/CONTEXT.md | ESTE ARCHIVO — fuente de verdad viva | ✅ Existe (v1.2.0) |
| docs/SESSION_SNAPSHOT.md | Log de sesiones de trabajo | ✅ Existe |
| pyproject.toml | Dependencias Python | ✅ Creado (V1-0) |
| Makefile | Pipeline targets | ✅ Creado (V1-0, placeholders) |
| configs/*.yaml | Configuración por dominio | ✅ Creado (V1-0) |
| contracts/*.yaml | Contratos de datos | ✅ Creado (V1-0) |

---

## 6. Decisiones arquitectónicas tomadas

| ID | Decisión | Razón | Fecha |
|----|---------|-------|-------|
| ADR-001 | Polars como procesador primario, no Pandas | 5-10× más rápido para 58M filas | 2026-03-15 |
| ADR-002 | LightGBM global (un modelo para todas las series) | Pragmático, escalable, probado en M5 top solutions | 2026-03-15 |
| ADR-003 | Clasificación ADI/CV² antes de modelar | Diferenciador senior, condiciona features y baselines | 2026-03-15 |
| ADR-004 | Capa sintética de inventario documentada como tal | Honestidad técnica, criterio de producto | 2026-03-15 |
| ADR-005 | Ejecución en 3 releases (V1→V2→V3) | Ship core first, prove it works, layer complexity | 2026-03-15 |
| ADR-006 | JSON precalculado para frontend (no inferencia) | HF Static Space no soporta backend pesado | 2026-03-15 |
| ADR-007 | ECharts sobre Plotly para dashboards | Mejor rendimiento en dashboards densos | 2026-03-15 |

---

## 7. Problemas conocidos y deuda técnica

| ID | Problema | Severidad | Plan |
|----|---------|----------|------|
| DT-001 | pandera LazyFrame membership PerformanceWarning (pandera internals) | Low | Upstream pandera fix; not actionable on our side |
| DT-002 | ~~M5 CSVs not extracted from zip~~ | Resolved | Pipeline executed 2026-03-15: bronze/silver/features/models all generated |
| DT-003 | ~~synthetic_params.parquet absent~~ | Partially resolved | synthetic_params.parquet generated (30,490 series); dim_product_scd.parquet and data/gold/models/ still absent (non-blocking) |
| DT-006 | Feature store + LightGBM training limited to 500-series sample due to OOM (segfault) on full 58M-row dataset in single Python process | Medium | Full dataset requires chunked/distributed processing or higher-RAM machine; sample mode produces valid metrics |
| DT-004 | ECharts chunk 628 KB > 500 KB Vite warning | Low | Use dynamic import() or manualChunks; cosmetic warning, build succeeds |
| DT-005 | asset_manifest.json in data/gold/serving/ lacked generated_at field | Low | Fixed in audit v1.6.0 — field added, synced to app/public/data/ |

---

## 8. Reglas que NINGUNA IA puede romper

1. **No simplificar la arquitectura** — si el doc dice Polars, no uses solo Pandas.
2. **No centrar en notebooks** — notebooks son para exploración, NUNCA lógica core.
3. **No presentar datos sintéticos como reales** — SIEMPRE etiquetados.
4. **No mezclar releases** — V1 completo antes de V2, V2 completo antes de V3.
5. **No meter inferencia pesada en frontend** — solo JSON precalculado.
6. **No ignorar intermitencia del M5** — >50% zeros, clasificación obligatoria.
7. **No producir boilerplate mediocre** — nivel Amazon/Tesla/Palantir.
8. **No dejar lógica sin test** — every module has tests.
9. **No hacer leakage** — respetar las 6 reglas del protocolo (sección 8.2 de la spec).
10. **Actualizar CONTEXT.md y SESSION_SNAPSHOT.md** después de cada cambio.
