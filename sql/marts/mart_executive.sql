-- =============================================================================
-- mart_executive.sql
-- Description : Pre-aggregated KPIs for the executive dashboard.
--               Supports three configurable aggregation grains, each
--               materialized as a separate final CTE:
--                 A) date × state        (regional view)
--                 B) date × category     (category mix view)
--                 C) month × state       (trend view for 12-month chart)
--               The dashboard reads from the appropriate alias.
--               All inventory metrics are SYNTHETIC; demand metrics are REAL.
-- Grain       : see per-CTE documentation above
-- Source      : {{ ref('mart_demand') }}
--               {{ ref('mart_forecast') }}   (lgbm_global, fold latest)
--               {{ ref('mart_inventory') }}
-- Key metrics : total_revenue_proxy, total_units_sold, forecast_accuracy_mae,
--               fill_rate_avg, stockout_rate, avg_days_of_supply,
--               inventory_value_proxy, n_skus_at_risk
-- =============================================================================

WITH

-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION A — Demand aggregates (REAL data)
-- ══════════════════════════════════════════════════════════════════════════════

-- ── A1. Daily demand by state ────────────────────────────────────────────────
demand_by_state AS (
    SELECT
        date,
        state_id,
        SUM(units_sold)                                 AS total_units_sold,
        SUM(revenue_proxy)                              AS total_revenue_proxy,
        COUNT(DISTINCT item_id)                         AS n_active_skus,
        COUNT(DISTINCT store_id)                        AS n_stores,
        AVG(rolling_28d_avg)                            AS avg_demand_per_sku,
        SUM(CASE WHEN is_zero_sale THEN 1 ELSE 0 END)   AS zero_sale_count,
        COUNT(*)                                        AS total_observations
    FROM {{ ref('mart_demand') }}
    GROUP BY date, state_id
),

-- ── A2. Daily demand by category ─────────────────────────────────────────────
demand_by_category AS (
    SELECT
        date,
        cat_id,
        SUM(units_sold)                                 AS total_units_sold,
        SUM(revenue_proxy)                              AS total_revenue_proxy,
        COUNT(DISTINCT item_id)                         AS n_active_skus,
        AVG(rolling_28d_avg)                            AS avg_demand_per_sku
    FROM {{ ref('mart_demand') }}
    GROUP BY date, cat_id
),

-- ── A3. Monthly demand by state (for 12-month trend chart) ───────────────────
demand_monthly_state AS (
    SELECT
        DATE_TRUNC('month', date)                       AS month,
        state_id,
        SUM(units_sold)                                 AS total_units_sold,
        SUM(revenue_proxy)                              AS total_revenue_proxy,
        COUNT(DISTINCT item_id)                         AS n_active_skus,
        AVG(rolling_28d_avg)                            AS avg_demand_per_sku
    FROM {{ ref('mart_demand') }}
    GROUP BY DATE_TRUNC('month', date), state_id
),

-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION B — Forecast accuracy aggregates
-- Uses only the best model (lgbm_global) and the latest fold for each series.
-- ══════════════════════════════════════════════════════════════════════════════

-- ── B1. Identify latest fold per model per series ────────────────────────────
latest_fold AS (
    SELECT
        model_id,
        MAX(fold_id)                                    AS max_fold_id
    FROM {{ ref('mart_forecast') }}
    WHERE model_id = 'lgbm_global'
    GROUP BY model_id
),

-- ── B2. Forecast metrics filtered to latest fold ─────────────────────────────
forecast_latest AS (
    SELECT
        f.date,
        f.store_id,
        f.item_id,
        f.state_id,
        f.cat_id,
        f.model_id,
        f.fold_id,
        f.abs_error,
        f.sq_error,
        f.smape,
        f.is_covered_80,
        f.interval_width
    FROM {{ ref('mart_forecast') }} f
    INNER JOIN latest_fold lf
        ON  f.model_id = lf.model_id
        AND f.fold_id  = lf.max_fold_id
),

-- ── B3. Accuracy by state ─────────────────────────────────────────────────────
accuracy_by_state AS (
    SELECT
        date,
        state_id,
        AVG(abs_error)                                  AS mae,
        SQRT(AVG(sq_error))                             AS rmse,
        AVG(smape)                                      AS smape_avg,
        AVG(CAST(is_covered_80 AS FLOAT)) * 100         AS coverage_80_pct,
        AVG(interval_width)                             AS avg_interval_width,
        COUNT(*)                                        AS n_observations
    FROM forecast_latest
    GROUP BY date, state_id
),

-- ── B4. Accuracy by category ──────────────────────────────────────────────────
accuracy_by_category AS (
    SELECT
        date,
        cat_id,
        AVG(abs_error)                                  AS mae,
        SQRT(AVG(sq_error))                             AS rmse,
        AVG(smape)                                      AS smape_avg,
        AVG(CAST(is_covered_80 AS FLOAT)) * 100         AS coverage_80_pct
    FROM forecast_latest
    GROUP BY date, cat_id
),

-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION C — Inventory aggregates (SYNTHETIC)
-- ══════════════════════════════════════════════════════════════════════════════

-- ── C1. Inventory KPIs by state ───────────────────────────────────────────────
inventory_by_state AS (
    SELECT
        date,
        state_id,
        AVG(fill_rate_28d)                              AS fill_rate_avg,
        AVG(stockout_prob_91d)                          AS stockout_prob_avg,
        SUM(CAST(is_stockout  AS INTEGER))              AS n_stockout_skus,
        SUM(CAST(is_overstock AS INTEGER))              AS n_overstock_skus,
        COUNT(*)                                        AS n_total_skus,
        AVG(days_of_supply)                             AS avg_days_of_supply,
        SUM(avg_inventory_28d)                          AS total_inventory_units,
        SUM(stockout_event_lost_revenue)                AS total_lost_revenue_proxy,
        SUM(CASE WHEN risk_tier IN ('STOCKOUT','HIGH_RISK') THEN 1 ELSE 0 END)
                                                        AS n_skus_at_risk
    FROM {{ ref('mart_inventory') }}
    GROUP BY date, state_id
),

-- ── C2. Inventory KPIs by category ───────────────────────────────────────────
inventory_by_category AS (
    SELECT
        date,
        cat_id,
        AVG(fill_rate_28d)                              AS fill_rate_avg,
        AVG(stockout_prob_91d)                          AS stockout_prob_avg,
        SUM(CAST(is_stockout  AS INTEGER))              AS n_stockout_skus,
        AVG(days_of_supply)                             AS avg_days_of_supply,
        SUM(stockout_event_lost_revenue)                AS total_lost_revenue_proxy,
        COUNT(*)                                        AS n_total_skus
    FROM {{ ref('mart_inventory') }}
    GROUP BY date, cat_id
),

-- ══════════════════════════════════════════════════════════════════════════════
-- SECTION D — Final stitched outputs
-- Three separate materialisations; downstream SQL or BI tool selects the grain.
-- ══════════════════════════════════════════════════════════════════════════════

-- ── D1. Executive KPIs by date × state ───────────────────────────────────────
exec_by_state AS (
    SELECT
        d.date,
        d.state_id,

        -- Demand (REAL)
        d.total_units_sold,
        d.total_revenue_proxy,
        d.n_active_skus,
        d.n_stores,
        d.avg_demand_per_sku,
        ROUND(d.zero_sale_count * 100.0 / NULLIF(d.total_observations, 0), 2)
                                                        AS pct_zero_sales,

        -- Forecast quality (LightGBM latest fold)
        a.mae                                           AS forecast_mae,
        a.rmse                                          AS forecast_rmse,
        a.smape_avg                                     AS forecast_smape,
        a.coverage_80_pct                               AS forecast_coverage_80,
        a.avg_interval_width,

        -- Inventory (SYNTHETIC)
        i.fill_rate_avg,
        ROUND(i.stockout_prob_avg * 100, 2)             AS stockout_rate_pct,
        i.n_stockout_skus,
        i.n_overstock_skus,
        i.avg_days_of_supply,
        i.total_inventory_units,
        i.total_lost_revenue_proxy,
        i.n_skus_at_risk,
        ROUND(i.n_skus_at_risk * 100.0 / NULLIF(i.n_total_skus, 0), 2)
                                                        AS pct_skus_at_risk,

        -- Combined health score [0-100]: higher = better
        -- Weights: 40% fill rate + 30% forecast accuracy + 30% zero-sale rate
        ROUND(
            LEAST(100,
                GREATEST(0,
                    40 * COALESCE(i.fill_rate_avg, 0)
                    + 30 * (1 - LEAST(a.smape_avg / 100.0, 1))
                    + 30 * (1 - COALESCE(d.zero_sale_count * 1.0
                                         / NULLIF(d.total_observations, 0), 0))
                )
            ) * 100
        , 1)                                            AS health_score,

        CURRENT_TIMESTAMP                               AS _dbt_loaded_at

    FROM demand_by_state      d
    LEFT JOIN accuracy_by_state  a ON d.date = a.date AND d.state_id = a.state_id
    LEFT JOIN inventory_by_state i ON d.date = i.date AND d.state_id = i.state_id
),

-- ── D2. Executive KPIs by date × category ────────────────────────────────────
exec_by_category AS (
    SELECT
        d.date,
        d.cat_id,

        -- Demand (REAL)
        d.total_units_sold,
        d.total_revenue_proxy,
        d.n_active_skus,
        d.avg_demand_per_sku,

        -- Forecast
        a.mae                                           AS forecast_mae,
        a.coverage_80_pct                               AS forecast_coverage_80,

        -- Inventory (SYNTHETIC)
        i.fill_rate_avg,
        ROUND(i.stockout_prob_avg * 100, 2)             AS stockout_rate_pct,
        i.n_stockout_skus,
        i.avg_days_of_supply,
        i.total_lost_revenue_proxy,

        CURRENT_TIMESTAMP                               AS _dbt_loaded_at

    FROM demand_by_category      d
    LEFT JOIN accuracy_by_category  a ON d.date = a.date AND d.cat_id = a.cat_id
    LEFT JOIN inventory_by_category i ON d.date = i.date AND d.cat_id = i.cat_id
),

-- ── D3. Monthly trend by state (12-month revenue + fill rate chart) ───────────
exec_monthly_state AS (
    SELECT
        dm.month,
        dm.state_id,
        dm.total_units_sold,
        dm.total_revenue_proxy,
        dm.n_active_skus,

        -- For monthly trend chart: average the daily fill rate over the month
        AVG(i.fill_rate_avg)                            AS fill_rate_monthly_avg,
        AVG(i.stockout_prob_avg)                        AS stockout_prob_monthly_avg,
        AVG(a.mae)                                      AS mae_monthly_avg,

        CURRENT_TIMESTAMP                               AS _dbt_loaded_at

    FROM demand_monthly_state dm
    -- Join on first-of-month date to the state grain
    LEFT JOIN inventory_by_state i
        ON DATE_TRUNC('month', i.date) = dm.month
        AND i.state_id = dm.state_id
    LEFT JOIN accuracy_by_state a
        ON DATE_TRUNC('month', a.date) = dm.month
        AND a.state_id = dm.state_id
    GROUP BY dm.month, dm.state_id, dm.total_units_sold, dm.total_revenue_proxy, dm.n_active_skus
)

-- ── Primary output: by state (most used by the dashboard) ────────────────────
-- Switch the final SELECT to exec_by_category or exec_monthly_state
-- when building the category-mix or trend sub-charts.
SELECT * FROM exec_by_state

-- Uncomment to expose additional grains (use dbt selectors / aliases):
-- SELECT * FROM exec_by_category
-- SELECT * FROM exec_monthly_state
