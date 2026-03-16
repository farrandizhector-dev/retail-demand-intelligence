"""Reorder Point (ROP) calculation.

Formula (spec section 10.1):
    ROP = Σ_{t=1}^{LT} ŷ_p50(t) + SS

When a daily p50 forecast is unavailable, the expected demand during
lead time is approximated as:
    ROP ≈ avg_daily_demand × lead_time_mean + SS

Output is always clipped to ≥ 0.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Scalar / array functions
# ---------------------------------------------------------------------------


def reorder_point(
    forecast_p50_during_lt: float,
    safety_stock: float,
) -> float:
    """ROP = expected demand during LT + safety stock.

    Parameters
    ----------
    forecast_p50_during_lt:
        Cumulative p50 forecast over the lead time window (units).
    safety_stock:
        Pre-computed safety stock (units).

    Returns
    -------
    float — reorder point in units (≥ 0).
    """
    return float(max(0.0, forecast_p50_during_lt + safety_stock))


def reorder_point_from_arrays(
    p50_forecasts: Sequence[float],
    safety_stock: float,
    lead_time_mean: float,
) -> float:
    """Compute ROP from daily p50 forecast array.

    Takes the first ``round(lead_time_mean)`` days of ``p50_forecasts``,
    sums them, and adds ``safety_stock``.

    Parameters
    ----------
    p50_forecasts:
        Ordered daily p50 forecast values (horizon ≥ lead_time_mean).
    safety_stock:
        Pre-computed safety stock (units).
    lead_time_mean:
        Mean lead time in days.

    Returns
    -------
    float — reorder point in units.
    """
    n = max(1, round(lead_time_mean))
    p50_arr = np.asarray(p50_forecasts[:n], dtype=np.float64)
    expected_demand = float(p50_arr.sum())
    return reorder_point(expected_demand, safety_stock)


def reorder_point_avg_demand(
    avg_daily_demand: float,
    lead_time_mean: float,
    safety_stock: float,
) -> float:
    """ROP using average demand proxy (when no forecast available).

    ROP = avg_daily_demand × lead_time_mean + SS

    Parameters
    ----------
    avg_daily_demand:
        Historical average daily demand (units/day).
    lead_time_mean:
        Mean lead time in days.
    safety_stock:
        Pre-computed safety stock (units).

    Returns
    -------
    float — reorder point in units.
    """
    expected_demand = avg_daily_demand * max(0.0, lead_time_mean)
    return reorder_point(expected_demand, safety_stock)


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------


def compute_reorder_points_batch(
    params_df,       # pl.DataFrame (synthetic params)
    ss_df,           # pl.DataFrame (safety stock)
    forecast_df=None,  # pl.DataFrame | None
    id_col: str = "id",
) -> "pl.DataFrame":
    """Compute ROP for each SKU-store.

    Parameters
    ----------
    params_df:
        Synthetic parameters (id, avg_daily_demand, lead_time_mean, …).
    ss_df:
        Safety stock per SKU-store (id, safety_stock).
    forecast_df:
        Optional forecast DataFrame (id, date, forecast_p50).
    id_col:
        Series identifier column.

    Returns
    -------
    pl.DataFrame with columns: id, reorder_point.
    """
    import polars as pl

    ss_map: dict[str, float] = {
        row[id_col]: float(row["safety_stock"])
        for row in ss_df.iter_rows(named=True)
    }

    rows: list[dict] = []

    for row in params_df.iter_rows(named=True):
        sid = row[id_col]
        lt = float(row.get("lead_time_mean", 7.0))
        avg_d = float(row.get("avg_daily_demand", 1.0) or 1.0)
        ss = ss_map.get(sid, 0.0)

        rop: float
        if forecast_df is not None:
            series_fc = forecast_df.filter(pl.col(id_col) == sid)
            if len(series_fc) >= 1:
                fc_sorted = series_fc.sort("date")
                p50_vals = fc_sorted["forecast_p50"].to_list()
                rop = reorder_point_from_arrays(p50_vals, ss, lt)
            else:
                rop = reorder_point_avg_demand(avg_d, lt, ss)
        else:
            rop = reorder_point_avg_demand(avg_d, lt, ss)

        rows.append({"id": sid, "reorder_point": rop})

    return pl.DataFrame(rows)
