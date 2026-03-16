-- =============================================================================
-- stg_sales.sql
-- Description : Staging layer for raw M5 sales data.
--               Casts types, renames columns to warehouse convention,
--               derives is_zero_sale and days_since_last_sale.
-- Grain       : one row per (date, store_id, item_id) — daily
-- Source      : {{ source('bronze', 'sales_long') }}
--               (output of bronze_writer: data/bronze/sales_long.parquet)
-- Downstream  : int_sales_enriched → fact_sales_daily
-- =============================================================================

WITH

-- ── 1. Raw cast ───────────────────────────────────────────────────────────────
-- Enforce types from bronze; handle any nulls that slipped through ingest.
raw AS (
    SELECT
        CAST(date        AS DATE)    AS date,
        CAST(store_id    AS VARCHAR) AS store_id,
        CAST(item_id     AS VARCHAR) AS item_id,
        CAST(units_sold  AS INTEGER) AS units_sold,

        -- Derived state / dept / category from structured ID conventions:
        --   item_id  = CAT_DEPT_NNN   e.g. HOBBIES_1_001
        --   store_id = STATE_NUM       e.g. CA_1
        SPLIT_PART(store_id, '_', 1)                            AS state_id,
        SPLIT_PART(item_id, '_', 1)                             AS cat_id,
        SPLIT_PART(item_id, '_', 1) || '_' || SPLIT_PART(item_id, '_', 2)
                                                                AS dept_id

    FROM {{ source('bronze', 'sales_long') }}
    WHERE date IS NOT NULL
      AND store_id IS NOT NULL
      AND item_id IS NOT NULL
      AND units_sold >= 0    -- guard against bad ingest; negatives are invalid
),

-- ── 2. Zero-sale flag ─────────────────────────────────────────────────────────
with_zero_flag AS (
    SELECT
        *,
        (units_sold = 0) AS is_zero_sale
    FROM raw
),

-- ── 3. Days since last sale (intermittency signal) ────────────────────────────
-- Uses a window function over ordered dates per series.
-- NULL on the very first sale of each series is acceptable.
with_days_since AS (
    SELECT
        *,
        date - MAX(CASE WHEN units_sold > 0 THEN date END)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )                                                    AS days_since_last_sale
    FROM with_zero_flag
),

-- ── 4. Revenue proxy ─────────────────────────────────────────────────────────
-- Joined with prices at the staging layer to avoid repeating the join downstream.
-- sell_price may be NULL for some weeks (product not on shelf) — kept as NULL
-- so downstream can choose to exclude or forward-fill.
final AS (
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
        p.sell_price,
        CASE
            WHEN p.sell_price IS NOT NULL
            THEN s.units_sold * p.sell_price
            ELSE NULL
        END                                                      AS revenue_proxy,

        -- Data lineage tag — required by contract (spec §8.2)
        'REAL'                                                   AS data_source_tag,
        CURRENT_TIMESTAMP                                        AS _dbt_loaded_at

    FROM with_days_since s
    LEFT JOIN {{ ref('stg_prices') }} p
        ON  s.item_id  = p.item_id
        AND s.store_id = p.store_id
        AND s.date     = p.date
)

SELECT * FROM final
