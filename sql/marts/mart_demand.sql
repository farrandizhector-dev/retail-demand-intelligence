-- =============================================================================
-- mart_demand.sql
-- Description : Historical demand enriched with product classifications,
--               store context, event context, and rolling metrics.
--               Primary read surface for the Forecast Lab and Product
--               Drilldown pages of the dashboard.
-- Grain       : one row per (date × store_id × item_id) — daily
-- Source      : {{ ref('stg_sales') }}
--               {{ ref('stg_calendar') }}
--               {{ ref('stg_prices') }}
--               {{ source('gold', 'dim_product') }}   (SCD Type 2, is_current)
--               {{ source('gold', 'dim_store') }}
-- Key metrics : units_sold, revenue_proxy, rolling_7d_avg, rolling_28d_avg,
--               yoy_change, zero_streak, pct_zero_last_28d
-- =============================================================================

WITH

-- ── 1. Base sales with calendar context ─────────────────────────────────────
sales AS (
    SELECT
        s.date,
        s.state_id,
        s.store_id,
        s.cat_id,
        s.dept_id,
        s.item_id,
        s.units_sold,
        s.is_zero_sale,
        s.days_since_last_sale,
        s.sell_price,
        s.revenue_proxy,

        -- Calendar fields joined once here to avoid repetition downstream
        c.year,
        c.quarter,
        c.month,
        c.week_of_year,
        c.day_of_week_iso,
        c.is_weekend,
        c.is_month_start,
        c.is_month_end,
        c.is_quarter_end,
        c.snap_ca,
        c.snap_tx,
        c.snap_wi,
        c.has_event,
        c.primary_event_name,
        c.primary_event_type,
        c.event_type_encoded,
        c.days_to_next_event,
        c.days_since_last_event,

        -- Store-level SNAP flag (each series belongs to one state)
        CASE s.state_id
            WHEN 'CA' THEN c.snap_ca
            WHEN 'TX' THEN c.snap_tx
            WHEN 'WI' THEN c.snap_wi
            ELSE FALSE
        END                                             AS snap_active

    FROM {{ ref('stg_sales') }} s
    INNER JOIN {{ ref('stg_calendar') }} c
        ON s.date = c.date
),

-- ── 2. Product dimension — current version only (SCD Type 2) ────────────────
-- is_current = TRUE gives us the latest classification for each SKU.
-- For historical analysis we accept that old records may show the latest class;
-- point-in-time joins are used in mart_executive where classification history matters.
dim_prod AS (
    SELECT
        item_id,
        abc_class,
        xyz_class,
        demand_class,
        avg_daily_demand,
        pct_zero_days,
        adi,
        cv_squared
    FROM {{ source('gold', 'dim_product') }}
    WHERE is_current = TRUE
),

-- ── 3. Store dimension ───────────────────────────────────────────────────────
dim_store AS (
    SELECT
        store_id,
        state_id         AS store_state,
        store_tier
    FROM {{ source('gold', 'dim_store') }}
),

-- ── 4. Joined base ───────────────────────────────────────────────────────────
joined AS (
    SELECT
        s.*,
        COALESCE(dp.abc_class,    'Unknown')            AS abc_class,
        COALESCE(dp.xyz_class,    'Unknown')            AS xyz_class,
        COALESCE(dp.demand_class, 'Unknown')            AS demand_class,
        dp.avg_daily_demand,
        dp.pct_zero_days,
        dp.adi,
        dp.cv_squared,
        ds.store_tier
    FROM sales s
    LEFT JOIN dim_prod  dp ON s.item_id  = dp.item_id
    LEFT JOIN dim_store ds ON s.store_id = ds.store_id
),

-- ── 5. Rolling demand metrics ────────────────────────────────────────────────
-- Computed per series (store × item), ordered by date.
-- These are the same rolling features used in the feature store; having them
-- in the mart allows analysts to explore without re-running the Python pipeline.
rolling AS (
    SELECT
        *,

        -- Short-term average
        AVG(units_sold)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            )                                           AS rolling_7d_avg,

        -- Medium-term average (demand baseline)
        AVG(units_sold)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
            )                                           AS rolling_28d_avg,

        -- Quarterly average (trend baseline)
        AVG(units_sold)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN 90 PRECEDING AND CURRENT ROW
            )                                           AS rolling_91d_avg,

        -- Revenue rolling (for executive KPIs)
        SUM(revenue_proxy)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
            )                                           AS rolling_28d_revenue,

        -- Zero-sale percentage in last 28 days (intermittency signal)
        AVG(CAST(is_zero_sale AS INTEGER))
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
            )                                           AS pct_zero_last_28d,

        -- Consecutive zero-sale streak (current run length)
        SUM(CAST(is_zero_sale AS INTEGER))
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
            - MAX(
                CASE WHEN NOT is_zero_sale THEN
                    SUM(CAST(is_zero_sale AS INTEGER))
                        OVER (
                            PARTITION BY store_id, item_id
                            ORDER BY date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        )
                ELSE NULL END
              )
              OVER (
                  PARTITION BY store_id, item_id
                  ORDER BY date
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
              )                                         AS zero_streak

    FROM joined
),

-- ── 6. Year-over-year change ─────────────────────────────────────────────────
-- Requires 365-row lag; NULL for the first year of each series.
yoy AS (
    SELECT
        *,
        units_sold - LAG(units_sold, 365)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
            )                                           AS yoy_units_delta,

        CASE
            WHEN LAG(units_sold, 365) OVER (
                     PARTITION BY store_id, item_id ORDER BY date
                 ) > 0
            THEN (units_sold
                  - LAG(units_sold, 365) OVER (
                        PARTITION BY store_id, item_id ORDER BY date
                    )
                 )
                 / LAG(units_sold, 365) OVER (
                       PARTITION BY store_id, item_id ORDER BY date
                   ) * 100
            ELSE NULL
        END                                             AS yoy_change_pct
    FROM rolling
),

-- ── 7. Final column ordering ─────────────────────────────────────────────────
final AS (
    SELECT
        -- Natural keys
        date,
        store_id,
        item_id,

        -- Hierarchy
        state_id,
        cat_id,
        dept_id,
        abc_class,
        xyz_class,
        demand_class,
        store_tier,

        -- Calendar context
        year,
        quarter,
        month,
        week_of_year,
        day_of_week_iso,
        is_weekend,
        snap_active,
        has_event,
        primary_event_name,
        primary_event_type,
        event_type_encoded,
        days_to_next_event,
        days_since_last_event,

        -- Core demand measures
        units_sold,
        is_zero_sale,
        days_since_last_sale,
        zero_streak,

        -- Price
        sell_price,
        revenue_proxy,
        rolling_28d_revenue,

        -- Rolling demand
        rolling_7d_avg,
        rolling_28d_avg,
        rolling_91d_avg,
        pct_zero_last_28d,

        -- YoY
        yoy_units_delta,
        yoy_change_pct,

        -- Classification signals
        avg_daily_demand,
        pct_zero_days,
        adi,
        cv_squared,

        CURRENT_TIMESTAMP AS _dbt_loaded_at
    FROM yoy
)

SELECT * FROM final
