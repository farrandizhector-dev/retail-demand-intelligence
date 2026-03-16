"""Price and elasticity features for demand forecasting.

Derives price-based signals by joining the silver daily prices table with
the sales DataFrame.

LEAKAGE PROTOCOL (rule 1 — section 8.2):
  All price features use ``sell_price`` on or before the current date
  (daily-grain forward-filled prices are available contemporaneously).
  Week-over-week delta uses ``price at t`` vs ``price at t-7``, which
  is safe as both dates are in the past at prediction time.
"""

from __future__ import annotations

from datetime import date

import polars as pl


def add_price_features(
    df: pl.DataFrame,
    prices_df: pl.DataFrame,
    *,
    date_col: str = "date",
    id_col: str = "id",
    item_col: str = "item_id",
    store_col: str = "store_id",
    dept_col: str = "dept_id",
    price_col: str = "sell_price",
    cutoff_date: date | None = None,
) -> pl.DataFrame:
    """Join price features onto the sales DataFrame.

    Features computed:
    - ``sell_price``: current daily sell price (forward-filled weekly → daily).
    - ``price_delta_wow``: price this week vs same day 7 weeks prior (Δ absolute).
    - ``price_ratio_vs_rolling_28``: current price / 28-day rolling mean price.
    - ``price_nunique_last_52w``: number of distinct price points in last 52 weeks.
    - ``is_price_drop_gt_10pct``: True if price dropped > 10 % vs prior week.
    - ``relative_price_in_dept``: price / mean(price in same department, same day).

    Parameters
    ----------
    df:
        Long-format sales DataFrame sorted by ``(id, date)``.
    prices_df:
        Silver daily prices: ``(store_id, item_id, date, sell_price)``.
    date_col, id_col, item_col, store_col, dept_col, price_col:
        Column name overrides.
    cutoff_date:
        Only price data up to this date is used (leakage guard).

    Returns
    -------
    pl.DataFrame
        Input DataFrame with price feature columns appended.
    """
    if cutoff_date:
        prices_df = prices_df.filter(pl.col(date_col) <= cutoff_date)

    # --- Join raw price ---
    price_slim = prices_df.select([store_col, item_col, date_col, price_col])
    df = df.join(price_slim, on=[store_col, item_col, date_col], how="left")

    # Sort for rolling computations
    df = df.sort([id_col, date_col])

    # --- price_delta_wow: shift(7) within each series ---
    df = df.with_columns(
        (pl.col(price_col) - pl.col(price_col).shift(7).over(id_col))
        .alias("price_delta_wow")
    )

    # --- price_ratio_vs_rolling_28 ---
    rolling_mean_28 = (
        pl.col(price_col)
        .rolling_mean(window_size=28, min_samples=1)
        .over(id_col)
    )
    df = df.with_columns(
        (pl.col(price_col) / rolling_mean_28.replace(0.0, None))
        .fill_null(1.0)
        .alias("price_ratio_vs_rolling_28")
    )

    # --- price_nunique_last_52w: rolling count of distinct prices (52 wks = 364 days) ---
    # Approximation: std of prices over last 364 days (if std=0, price never changed)
    # True nunique in rolling is not natively supported; use a proxy
    df = df.with_columns(
        pl.col(price_col)
        .rolling_std(window_size=364, min_samples=1)
        .over(id_col)
        .fill_null(0.0)
        .alias("price_nunique_last_52w")   # proxy: price std over last year
    )

    # --- is_price_drop_gt_10pct ---
    price_7_ago = pl.col(price_col).shift(7).over(id_col)
    df = df.with_columns(
        pl.when(
            (pl.col(price_col) < price_7_ago * 0.90) & price_7_ago.is_not_null()
        )
        .then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("is_price_drop_gt_10pct")
    )

    # --- relative_price_in_dept: price / mean(dept price on same day) ---
    dept_mean = (
        df
        .group_by([dept_col, date_col])
        .agg(pl.col(price_col).mean().alias("_dept_mean_price"))
    )
    df = df.join(dept_mean, on=[dept_col, date_col], how="left")
    df = df.with_columns(
        (pl.col(price_col) / pl.col("_dept_mean_price").replace(0.0, None))
        .fill_null(1.0)
        .alias("relative_price_in_dept")
    )
    df = df.drop("_dept_mean_price")

    return df
