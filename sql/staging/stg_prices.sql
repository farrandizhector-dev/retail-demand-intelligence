-- =============================================================================
-- stg_prices.sql
-- Description : Staging layer for M5 sell_prices.csv.
--               Source is weekly (store × item × wm_yr_wk).
--               This model:
--                 1. Casts and validates the raw weekly prices.
--                 2. Joins to the date spine to explode weekly → daily.
--                 3. Forward-fills gaps (items not listed every week).
--                 4. Derives price-change features (WoW delta, rolling ratio,
--                    promo proxy, price momentum).
-- Grain       : one row per (date, store_id, item_id) — daily after explosion
-- Source      : {{ source('bronze', 'sell_prices') }}  (weekly)
--               {{ ref('stg_calendar') }}              (date spine + wm_yr_wk)
-- Downstream  : stg_sales (revenue_proxy), fact_price_daily, feature store
-- Note        : sell_price can be legitimately NULL for weeks when an item
--               was not on the shelf. NULL is preserved; forward-fill is
--               explicit and bounded to 8 weeks (56 days) max gap.
-- =============================================================================

WITH

-- ── 1. Raw weekly prices — cast & validate ───────────────────────────────────
raw_weekly AS (
    SELECT
        CAST(store_id   AS VARCHAR) AS store_id,
        CAST(item_id    AS VARCHAR) AS item_id,
        CAST(wm_yr_wk   AS INTEGER) AS wm_yr_wk,
        CAST(sell_price AS FLOAT)   AS sell_price

    FROM {{ source('bronze', 'sell_prices') }}
    WHERE sell_price IS NOT NULL
      AND sell_price > 0
),

-- ── 2. Date spine from calendar ──────────────────────────────────────────────
-- We need one row per date for every store×item combination.
-- The calendar provides the wm_yr_wk → date mapping.
date_spine AS (
    SELECT DISTINCT
        date,
        wm_yr_wk
    FROM {{ ref('stg_calendar') }}
),

-- ── 3. All store×item×wm_yr_wk combos that appear in prices ─────────────────
-- Cross-joining with the date spine gives us the weekly granularity first,
-- then we expand to daily in the next CTE.
weekly_with_dates AS (
    SELECT
        ds.date,
        ds.wm_yr_wk,
        rw.store_id,
        rw.item_id,
        rw.sell_price
    FROM date_spine ds
    INNER JOIN raw_weekly rw
        ON ds.wm_yr_wk = rw.wm_yr_wk
),

-- ── 4. Forward-fill within each series ───────────────────────────────────────
-- For dates where a price exists it's used as-is.
-- For dates where price is NULL (item off-shelf for that week), we carry
-- forward the last known price — bounded to 56 days (8 Walmart weeks).
-- This mirrors the Polars ffill logic in src/transform/pipeline.py.
filled AS (
    SELECT
        date,
        wm_yr_wk,
        store_id,
        item_id,

        -- Last non-null price observed on or before this date for this series
        LAST_VALUE(sell_price IGNORE NULLS)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )                                           AS sell_price,

        -- Keep original to distinguish filled vs observed
        sell_price                                      AS sell_price_raw,
        sell_price IS NULL                              AS was_price_missing
    FROM weekly_with_dates
),

-- ── 5. Week-over-week price delta ─────────────────────────────────────────────
-- Compares current price to the price exactly 7 days ago (prior Walmart week).
price_delta AS (
    SELECT
        *,
        sell_price
            - LAG(sell_price, 7)
              OVER (
                  PARTITION BY store_id, item_id
                  ORDER BY date
              )                                         AS price_delta_wow,

        -- % change vs prior week (NULL-safe)
        CASE
            WHEN LAG(sell_price, 7) OVER (
                    PARTITION BY store_id, item_id ORDER BY date
                 ) > 0
            THEN (sell_price
                  - LAG(sell_price, 7) OVER (
                        PARTITION BY store_id, item_id ORDER BY date
                    )
                 )
                 / LAG(sell_price, 7) OVER (
                       PARTITION BY store_id, item_id ORDER BY date
                   )
            ELSE NULL
        END                                             AS price_change_pct_wow
    FROM filled
),

-- ── 6. Rolling price baseline (28-day mean) ──────────────────────────────────
-- Used to detect temporary promotions (spec §8.1 Fam-4: is_promo_proxy).
rolling_stats AS (
    SELECT
        *,
        AVG(sell_price)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
            )                                           AS price_rolling_mean_28d,

        -- Price ratio: current vs rolling mean
        -- < 0.85 → likely promotional price (>15% below rolling)
        CASE
            WHEN AVG(sell_price) OVER (
                     PARTITION BY store_id, item_id
                     ORDER BY date
                     ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
                 ) > 0
            THEN sell_price
                 / AVG(sell_price) OVER (
                       PARTITION BY store_id, item_id
                       ORDER BY date
                       ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
                   )
            ELSE NULL
        END                                             AS price_ratio_vs_rolling_28,

        -- 4-week price momentum (current vs price 28d ago)
        CASE
            WHEN LAG(sell_price, 28) OVER (
                     PARTITION BY store_id, item_id ORDER BY date
                 ) > 0
            THEN (sell_price
                  - LAG(sell_price, 28) OVER (
                        PARTITION BY store_id, item_id ORDER BY date
                    )
                 )
                 / LAG(sell_price, 28) OVER (
                       PARTITION BY store_id, item_id ORDER BY date
                   )
            ELSE NULL
        END                                             AS price_momentum_4w,

        -- Count distinct prices in the last 52 weeks (price stability)
        COUNT(DISTINCT sell_price)
            OVER (
                PARTITION BY store_id, item_id
                ORDER BY date
                ROWS BETWEEN 364 PRECEDING AND CURRENT ROW
            )                                           AS price_nunique_last_52w
    FROM price_delta
),

-- ── 7. Promo proxy flag ──────────────────────────────────────────────────────
-- Inferred: price dropped >15% vs rolling 28d mean → likely promotion.
-- Not a true label (M5 does not provide promo flags) — documented as inferred.
with_promo AS (
    SELECT
        *,
        (price_ratio_vs_rolling_28 < 0.85)             AS is_promo_proxy,
        (ABS(COALESCE(price_change_pct_wow, 0)) > 0.10) AS is_price_drop_gt_10pct
    FROM rolling_stats
),

-- ── 8. Relative price within department ─────────────────────────────────────
-- How does this item's price compare to the dept mean on this date?
-- Captures cross-item elasticity signal.
final AS (
    SELECT
        wp.date,
        wp.wm_yr_wk,
        wp.store_id,
        wp.item_id,
        SPLIT_PART(wp.item_id, '_', 1) || '_' || SPLIT_PART(wp.item_id, '_', 2)
                                                        AS dept_id,
        wp.sell_price,
        wp.sell_price_raw,
        wp.was_price_missing,
        wp.price_delta_wow,
        wp.price_change_pct_wow,
        wp.price_rolling_mean_28d,
        wp.price_ratio_vs_rolling_28,
        wp.price_momentum_4w,
        wp.price_nunique_last_52w,
        wp.is_promo_proxy,
        wp.is_price_drop_gt_10pct,

        -- Relative price: item price / dept avg price on same date
        CASE
            WHEN AVG(wp.sell_price) OVER (
                     PARTITION BY wp.store_id,
                         SPLIT_PART(wp.item_id,'_',1)||'_'||SPLIT_PART(wp.item_id,'_',2),
                         wp.date
                 ) > 0
            THEN wp.sell_price
                 / AVG(wp.sell_price) OVER (
                       PARTITION BY wp.store_id,
                           SPLIT_PART(wp.item_id,'_',1)||'_'||SPLIT_PART(wp.item_id,'_',2),
                           wp.date
                   )
            ELSE NULL
        END                                             AS relative_price_in_dept,

        CURRENT_TIMESTAMP                               AS _dbt_loaded_at
    FROM with_promo wp
)

SELECT * FROM final
