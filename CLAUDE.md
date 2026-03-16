# CLAUDE.md — Claude Code lee este archivo automáticamente desde la raíz del repo.

## IDENTIDAD DEL PROYECTO
Proyecto: AI Supply Chain Control Tower (Retail Demand Intelligence & Inventory Optimization Platform)
Autor: Héctor Ferrándiz Sanchis
Nivel: Principal/Staff-grade portfolio project

## REGLA DE CONTEXTO (OBLIGATORIA)
1) ANTES de responder, lee `docs/CONTEXT.md` y `docs/SESSION_SNAPSHOT.md`.
2) Trata `docs/CONTEXT.md` como fuente de verdad. Si hay contradicciones, dilo.
3) Responde SOLO alineado con ese contexto.
4) Si tu respuesta implica cambios, AL FINAL:
   - Actualiza `docs/CONTEXT.md` (bump version).
   - Añade entrada en `docs/SESSION_SNAPSHOT.md`.
   - Devuelve: (a) respuesta, (b) qué se actualizó, (c) comandos para verificar.

## REGLA DE RELEASES
- V1: Core Demo (ingest, features, LightGBM, inventory básico, 4 pages, deploy)
- V2: Technical Depth (reconciliation, Monte Carlo, SHAP, scenarios, +3 pages)
- V3: Enterprise Polish (monitoring, SCD, CI/CD, E2E, docs)
NUNCA implementes algo de V2 si V1 no está completado. Consulta CONTEXT.md para saber la fase actual.

## TU ROL COMO CLAUDE CODE
Tú implementas la lógica pesada:
- V1-Fases 1-4: Data Engineering, Classification, Features, Modeling, Inventory, Export
- V2-Fases 1-3, 5: Reconciliation, Conformal, Monte Carlo, MLflow, Model Cards
- V3-Fases 1-3: Monitoring, SCD, CI/CD, E2E tests

## REGLAS DE IMPLEMENTACIÓN
- Después de implementar cada módulo, ejecuta `pytest` sobre los tests correspondientes.
- Si un test falla, itera hasta que pase. No continúes con tests rotos.
- Logea todo a MLflow cuando corresponda (training runs).
- Sigue los contratos de datos de `contracts/*.yaml`.
- Usa Polars como procesador primario (no Pandas salvo compatibilidad).
- No dejes lógica sin test unitario.
- Sigue las 6 reglas de leakage control de la spec (sección 8.2).
- Los datos sintéticos SIEMPRE se etiquetan como SYNTHETIC en código y outputs.

## SPEC MAESTRA
Toda la especificación técnica está en `docs/MASTER_SPEC_v3.md`.
Consúltala para: arquitectura, contratos, features, modelos, inventario, métricas.

## COMANDOS ÚTILES
```bash
make setup          # Install dependencies
make ingest         # Download M5 + weather + macro
make validate       # Run all data quality checks
make train_lgbm     # Train LightGBM global + quantile
make evaluate       # Run 5-fold backtesting
make export_serving # Generate JSON serving assets
pytest tests/ -v    # Run all tests
```
