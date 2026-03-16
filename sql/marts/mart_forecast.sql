-- =============================================================================
-- mart_forecast.sql
-- Description : Forecast predictions joined with actuals for every model,
--               fold, and series. Computes a full suite of error metrics.
--               Primary read surface for the Model Quality page and any
--               downstream model comparison analysis.
-- Grain       : one row per (date × store_id × item_id × model_id × fold_id)
-- Source      : {{ source('gold', 'fact_forecast_daily') }}
--               {{ ref('mart_demand') }}          (actuals + product context)
--               {{ source('gold', 'dim_product') }}
-- Key metrics : forecast_p50, actual, error, abs_error, sq_error,
--               bias, mape_safe, smape, pinball_p10, pinball_p90,
--               is_covered_80 (actual within [p10, p90])
-- Notes       :
--   - mape_safe avoids division by zero by substituting a floor of 1
--     when actual = 0 (standard practice for intermittent demand).
--   - smape is bounded [0, 200%] by construction.
--   - Pinball loss = quantile regression loss for p10 and p90.
--   - Bias > 0 → over-forecast; Bias < 0 → under-forecast.
-- =============================================================================

WITH

-- ── 1. Forecast fact ─────────────────────────────────────────────────────────
forecasts AS (
    SELECT
        date,
        store_id,
        item_id,
        model_id,
        fold_id,
        forecast_p10,
        forecast_p50,
        forecast_p90,
        forecast_mean,
        is_reconciled,
        reconciliation_method
    FROM {{ source('gold', 'fact_forecast_daily') }}
),

-- ── 2. Actuals from demand mart ─────────────────────────────────────────────
actuals AS (
    SELECT
        date,
        store_id,
        item_id,
        units_sold          AS actual,
        abc_class,
        xyz_class,
        demand_class,
        cat_id,
        dept_id,
        state_id,
        is_weekend,
        has_event,
        snap_active
    FROM {{ ref('mart_demand') }}
),

-- ── 3. Join forecast to actuals ──────────────────────────────────────────────
joined AS (
    SELECT
        f.date,
        f.store_id,
        f.item_id,
        f.model_id,
        f.fold_id,
        f.forecast_p10,
        f.forecast_p50,
        f.forecast_p90,
        f.forecast_mean,
        f.is_reconciled,
        f.reconciliation_method,
        a.actual,
        a.abc_class,
        a.xyz_class,
        a.demand_class,
        a.cat_id,
        a.dept_id,
        a.state_id,
        a.is_weekend,
        a.has_event,
        a.snap_active
    FROM forecasts f
    LEFT JOIN actuals a
        ON  f.date     = a.date
        AND f.store_id = a.store_id
        AND f.item_id  = a.item_id
),

-- ── 4. Point-forecast error metrics ─────────────────────────────────────────
errors AS (
    SELECT
        *,

        -- Raw signed error (positive = over-forecast)
        forecast_p50 - actual                           AS error,
        ABS(forecast_p50 - actual)                      AS abs_error,
        POWER(forecast_p50 - actual, 2)                 AS sq_error,

        -- MAPE: divide by max(actual, 1) to avoid division by zero on zeros
        ABS(forecast_p50 - actual)
            / NULLIF(GREATEST(actual, 1.0), 0) * 100    AS mape_safe,

        -- sMAPE: symmetric, bounded [0,200%]
        -- Formula: 200 * |actual - forecast| / (|actual| + |forecast| + ε)
        CASE
            WHEN (ABS(actual) + ABS(forecast_p50)) = 0 THEN 0.0
            ELSE 200.0 * ABS(actual - forecast_p50)
                 / (ABS(actual) + ABS(forecast_p50))
        END                                             AS smape,

        -- Coverage: is actual within [p10, p90]?
        (actual >= forecast_p10 AND actual <= forecast_p90)
                                                        AS is_covered_80,

        -- Interval width (precision measure — narrower is better given coverage)
        forecast_p90 - forecast_p10                     AS interval_width

    FROM joined
),

-- ── 5. Pinball (quantile) loss ───────────────────────────────────────────────
-- Pinball loss at quantile q:
--   L(q, y, ŷ) = (y - ŷ) * q       if y >= ŷ
--              = (ŷ - y) * (1 - q)  if y <  ŷ
-- We compute for q=0.10 and q=0.90 (the interval boundaries).
with_pinball AS (
    SELECT
        *,

        -- Pinball at p10 (penalises under-coverage at low tail)
        CASE
            WHEN actual >= forecast_p10
            THEN (actual - forecast_p10) * 0.10
            ELSE (forecast_p10 - actual) * 0.90
        END                                             AS pinball_p10,

        -- Pinball at p90 (penalises under-coverage at high tail)
        CASE
            WHEN actual >= forecast_p90
            THEN (actual - forecast_p90) * 0.90
            ELSE (forecast_p90 - actual) * 0.10
        END                                             AS pinball_p90

    FROM errors
),

-- ── 6. Final column ordering ─────────────────────────────────────────────────
final AS (
    SELECT
        -- Natural keys
        date,
        store_id,
        item_id,
        model_id,
        fold_id,

        -- Hierarchy
        state_id,
        cat_id,
        dept_id,
        abc_class,
        xyz_class,
        demand_class,

        -- Context
        is_weekend,
        has_event,
        snap_active,

        -- Forecast values
        forecast_p10,
        forecast_p50,
        forecast_p90,
        forecast_mean,
        is_reconciled,
        reconciliation_method,

        -- Actual
        actual,

        -- Error metrics
        error,
        abs_error,
        sq_error,
        mape_safe,
        smape,
        is_covered_80,
        interval_width,
        pinball_p10,
        pinball_p90,

        CURRENT_TIMESTAMP AS _dbt_loaded_at
    FROM with_pinball
)

SELECT * FROM final
