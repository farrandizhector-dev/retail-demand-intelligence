"""Safety stock calculation for demand forecasting-driven inventory.

Two methods (spec section 10.1):

Classic (Normal approximation):
    SS = z(SL) × σ_forecast × √(LT)

Quantile (preferred when quantile forecasts are available):
    SS = forecast_p90_during_LT − forecast_p50_during_LT

where "during_LT" means the cumulative forecast over ``lead_time_mean`` days.

Both methods clip safety stock to ≥ 0 (never negative).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# z-score helper
# ---------------------------------------------------------------------------


def z_score(service_level: float) -> float:
    """Return the standard Normal z-score for a given service level.

    Parameters
    ----------
    service_level:
        Desired service level in (0, 1), e.g. 0.95.

    Returns
    -------
    float — z-score such that P(Z ≤ z) = service_level.

    Examples
    --------
    >>> round(z_score(0.95), 3)
    1.645
    >>> round(z_score(0.98), 3)
    2.054
    """
    if not (0.0 < service_level < 1.0):
        raise ValueError(f"service_level must be in (0, 1), got {service_level}")
    return float(norm.ppf(service_level))


# ---------------------------------------------------------------------------
# Classic method
# ---------------------------------------------------------------------------


def safety_stock_classic(
    service_level: float,
    sigma_forecast: float,
    lead_time_mean: float,
) -> float:
    """Classic safety stock: SS = z(SL) × σ_forecast × √(LT).

    Parameters
    ----------
    service_level:
        Target service level (e.g. 0.95).
    sigma_forecast:
        Standard deviation of daily demand forecast error (or demand std).
    lead_time_mean:
        Mean lead time in days.

    Returns
    -------
    float — safety stock in units (≥ 0).
    """
    if lead_time_mean <= 0:
        return 0.0
    z = z_score(service_level)
    ss = z * sigma_forecast * np.sqrt(lead_time_mean)
    return float(max(0.0, ss))


# ---------------------------------------------------------------------------
# Quantile method (preferred)
# ---------------------------------------------------------------------------


def safety_stock_quantile(
    forecast_p50_during_lt: float,
    forecast_p90_during_lt: float,
) -> float:
    """Quantile safety stock: SS = forecast_p90_during_LT − forecast_p50_during_LT.

    Parameters
    ----------
    forecast_p50_during_lt:
        Cumulative p50 forecast over the lead time window (sum of daily p50).
    forecast_p90_during_lt:
        Cumulative p90 forecast over the lead time window (sum of daily p90).

    Returns
    -------
    float — safety stock in units (≥ 0).
    """
    ss = forecast_p90_during_lt - forecast_p50_during_lt
    return float(max(0.0, ss))


def safety_stock_quantile_from_arrays(
    p50_forecasts: Sequence[float],
    p90_forecasts: Sequence[float],
    lead_time_mean: float,
) -> float:
    """Compute quantile SS from daily forecast arrays truncated to LT days.

    Takes the first ``round(lead_time_mean)`` days of each forecast array,
    sums them, and applies the quantile formula.

    Parameters
    ----------
    p50_forecasts:
        Daily p50 forecasts (horizon must be ≥ lead_time_mean).
    p90_forecasts:
        Daily p90 forecasts.
    lead_time_mean:
        Mean lead time in days.

    Returns
    -------
    float — safety stock in units.
    """
    n = max(1, round(lead_time_mean))
    p50_arr = np.asarray(p50_forecasts[:n], dtype=np.float64)
    p90_arr = np.asarray(p90_forecasts[:n], dtype=np.float64)
    return safety_stock_quantile(float(p50_arr.sum()), float(p90_arr.sum()))


# ---------------------------------------------------------------------------
# Batch computation over a DataFrame
# ---------------------------------------------------------------------------


def compute_safety_stock_batch(
    params_df,  # pl.DataFrame
    forecast_df=None,  # pl.DataFrame | None
    method: str = "quantile",
    id_col: str = "id",
) -> "pl.DataFrame":
    """Compute safety stock for each SKU-store.

    Parameters
    ----------
    params_df:
        Synthetic parameters DataFrame (output of ``generate_synthetic_params``).
        Required columns: id, service_level_target, lead_time_mean.
        For the classic method also needs: demand_std (or avg_daily_demand).
    forecast_df:
        Optional forecast DataFrame with columns: id, date, forecast_p50, forecast_p90.
        Required for method='quantile'.
    method:
        'quantile' (preferred) or 'classic'.

    Returns
    -------
    pl.DataFrame with columns: id, safety_stock, ss_method.
    """
    import polars as pl

    rows: list[dict] = []

    for row in params_df.iter_rows(named=True):
        sid = row[id_col]
        sl = float(row.get("service_level_target", 0.95))
        lt = float(row.get("lead_time_mean", 7.0))

        ss: float
        if method == "quantile" and forecast_df is not None:
            series_fc = forecast_df.filter(pl.col(id_col) == sid)
            if len(series_fc) >= 1:
                n_lt = max(1, round(lt))
                fc_sorted = series_fc.sort("date").head(n_lt)
                p50_sum = float(fc_sorted["forecast_p50"].sum())
                p90_sum = float(fc_sorted["forecast_p90"].sum())
                ss = safety_stock_quantile(p50_sum, p90_sum)
            else:
                # fallback to classic with avg_demand estimate
                avg_d = float(row.get("avg_daily_demand", 1.0) or 1.0)
                ss = safety_stock_classic(sl, avg_d * 0.5, lt)
        else:
            # Classic method
            avg_d = float(row.get("avg_daily_demand", 1.0) or 1.0)
            sigma = float(row.get("demand_std", avg_d * 0.5) or avg_d * 0.5)
            ss = safety_stock_classic(sl, sigma, lt)

        rows.append({"id": sid, "safety_stock": ss, "ss_method": method})

    return pl.DataFrame(rows)
