-- =============================================================================
-- stg_calendar.sql
-- Description : Staging layer for M5 calendar.csv.
--               Adds computed temporal fields, encodes event types,
--               and builds SNAP flags per state.
--               This becomes the source of truth for dim_date.
-- Grain       : one row per calendar date (1,969 days: 2011-01-29 → 2016-06-19)
-- Source      : {{ source('bronze', 'calendar') }}
--               (data/bronze/calendar.parquet)
-- Downstream  : dim_date, int_sales_enriched, stg_sales (via event context)
-- =============================================================================

WITH

-- ── 1. Base calendar cast ─────────────────────────────────────────────────────
base AS (
    SELECT
        CAST(date        AS DATE)    AS date,
        CAST(wm_yr_wk    AS INTEGER) AS wm_yr_wk,   -- Walmart week key (used for price join)
        CAST(weekday     AS INTEGER) AS weekday,     -- 0=Monday … 6=Sunday (M5 convention)
        CAST(wday        AS INTEGER) AS wday,        -- 1=Saturday … 7=Friday (M5 original)
        CAST(month       AS INTEGER) AS month,
        CAST(year        AS INTEGER) AS year,
        COALESCE(event_name_1, '')   AS event_name_1,
        COALESCE(event_type_1, '')   AS event_type_1,
        COALESCE(event_name_2, '')   AS event_name_2,
        COALESCE(event_type_2, '')   AS event_type_2,
        CAST(snap_CA     AS BOOLEAN) AS snap_ca,
        CAST(snap_TX     AS BOOLEAN) AS snap_tx,
        CAST(snap_WI     AS BOOLEAN) AS snap_wi,
        CAST(d           AS VARCHAR) AS d_col        -- "d_1" … "d_1969" original column label
    FROM {{ source('bronze', 'calendar') }}
),

-- ── 2. Standard temporal decomposition ───────────────────────────────────────
-- Adds fields that are expensive to recompute in every downstream query.
temporal AS (
    SELECT
        *,
        -- ISO day-of-week: 1=Monday, 7=Sunday
        EXTRACT(ISODOW FROM date)                       AS day_of_week_iso,
        EXTRACT(DAY    FROM date)                       AS day_of_month,
        EXTRACT(DOY    FROM date)                       AS day_of_year,
        EXTRACT(WEEK   FROM date)                       AS week_of_year,
        EXTRACT(QUARTER FROM date)                      AS quarter,

        -- Boolean convenience flags
        EXTRACT(ISODOW FROM date) >= 6                  AS is_weekend,
        EXTRACT(DAY    FROM date) = 1                   AS is_month_start,
        date = DATE_TRUNC('month', date)
              + INTERVAL '1 month' - INTERVAL '1 day'  AS is_month_end,
        EXTRACT(MONTH  FROM date) IN (3, 6, 9, 12)
            AND date = DATE_TRUNC('month', date)
              + INTERVAL '1 month' - INTERVAL '1 day'  AS is_quarter_end,
        EXTRACT(MONTH  FROM date) = 12
            AND EXTRACT(DAY FROM date) = 31             AS is_year_end,

        -- Sinusoidal encoding of week position (for ML use)
        SIN(2 * 3.14159265 * EXTRACT(DOY FROM date) / 365.25) AS doy_sin,
        COS(2 * 3.14159265 * EXTRACT(DOY FROM date) / 365.25) AS doy_cos,
        SIN(2 * 3.14159265 * EXTRACT(ISODOW FROM date) / 7)   AS dow_sin,
        COS(2 * 3.14159265 * EXTRACT(ISODOW FROM date) / 7)   AS dow_cos
    FROM base
),

-- ── 3. Event consolidation ───────────────────────────────────────────────────
-- M5 calendar has up to 2 events per day; we unify into a single event flag
-- and a standardised event_type enum for simpler downstream filtering.
events AS (
    SELECT
        *,

        -- At least one event on this date?
        (event_name_1 <> '' OR event_name_2 <> '') AS has_event,

        -- Priority: event_1 takes precedence; fall back to event_2
        CASE
            WHEN event_name_1 <> '' THEN event_name_1
            WHEN event_name_2 <> '' THEN event_name_2
            ELSE NULL
        END                                         AS primary_event_name,

        CASE
            WHEN event_type_1 <> '' THEN event_type_1
            WHEN event_type_2 <> '' THEN event_type_2
            ELSE NULL
        END                                         AS primary_event_type,

        -- Integer-encoded event type for ML (NULL → 0)
        CASE primary_event_type
            WHEN 'Cultural'  THEN 1
            WHEN 'National'  THEN 2
            WHEN 'Religious' THEN 3
            WHEN 'Sporting'  THEN 4
            ELSE 0
        END                                         AS event_type_encoded
    FROM temporal
),

-- ── 4. Days-to / days-since nearest event (proximity features) ───────────────
-- These are derived here once and reused in the feature store (spec §8.1 Fam-3).
event_proximity AS (
    SELECT
        e.*,

        -- Days until the NEXT event (forward-looking within dataset window)
        MIN(CASE WHEN has_event THEN date END)
            OVER (
                ORDER BY date
                ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
            ) - date                                AS days_to_next_event,

        -- Days since the LAST event (backward-looking)
        date - MAX(CASE WHEN has_event THEN date END)
            OVER (
                ORDER BY date
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )                                       AS days_since_last_event,

        -- SNAP benefit days (by state) — days until next SNAP active day
        MIN(CASE WHEN snap_ca THEN date END)
            OVER (ORDER BY date ROWS BETWEEN CURRENT ROW AND 30 FOLLOWING)
            - date                                  AS days_to_next_snap_ca,
        MIN(CASE WHEN snap_tx THEN date END)
            OVER (ORDER BY date ROWS BETWEEN CURRENT ROW AND 30 FOLLOWING)
            - date                                  AS days_to_next_snap_tx,
        MIN(CASE WHEN snap_wi THEN date END)
            OVER (ORDER BY date ROWS BETWEEN CURRENT ROW AND 30 FOLLOWING)
            - date                                  AS days_to_next_snap_wi
    FROM events
),

-- ── 5. Final select — clean column ordering ──────────────────────────────────
final AS (
    SELECT
        -- Keys
        date,
        wm_yr_wk,
        d_col,

        -- Temporal decomposition
        year,
        quarter,
        month,
        week_of_year,
        day_of_week_iso,
        day_of_month,
        day_of_year,
        is_weekend,
        is_month_start,
        is_month_end,
        is_quarter_end,
        is_year_end,

        -- ML-ready encodings
        doy_sin,
        doy_cos,
        dow_sin,
        dow_cos,

        -- Events
        has_event,
        primary_event_name,
        primary_event_type,
        event_type_encoded,
        event_name_1,
        event_type_1,
        event_name_2,
        event_type_2,
        days_to_next_event,
        days_since_last_event,

        -- SNAP flags
        snap_ca,
        snap_tx,
        snap_wi,
        days_to_next_snap_ca,
        days_to_next_snap_tx,
        days_to_next_snap_wi,

        CURRENT_TIMESTAMP AS _dbt_loaded_at
    FROM event_proximity
)

SELECT * FROM final
