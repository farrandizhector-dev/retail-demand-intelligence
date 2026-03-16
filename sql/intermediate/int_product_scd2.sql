-- int_product_scd2.sql
-- dbt snapshot for dim_product — SCD Type 2 tracking abc_class, xyz_class, demand_class
-- spec §7.2: trigger on changes to classification columns every 90 days
{% snapshot dim_product_snapshot %}
    {{
        config(
            target_schema='snapshots',
            unique_key='item_id',
            strategy='check',
            check_cols=['abc_class', 'xyz_class', 'demand_class'],
            invalidate_hard_deletes=True,
        )
    }}
    SELECT
        item_id,
        dept_id,
        cat_id,
        item_name,
        abc_class,
        xyz_class,
        demand_class,
        avg_daily_demand,
        pct_zero_days,
        adi,
        cv_squared
    FROM {{ ref('int_demand_classification') }}
{% endsnapshot %}
