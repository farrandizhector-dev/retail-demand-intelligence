"""Interaction and target-encoding features for demand forecasting.

Implements two types of features:

1. **Target encoding** (leave-one-out, LOO):
   Each categorical level is replaced by the mean target within that group,
   excluding the current row.  LOO + Gaussian noise regularisation prevents
   overfitting on small groups.

   LEAKAGE PROTOCOL (rule 5 — section 8.2):
     The encoder MUST be fitted on training data only.  The ``fit_transform``
     path (used during training) uses LOO within the training fold.  The
     ``transform`` path (used on validation/test) applies the fitted group
     means without LOO (the held-out rows are not in the fit set).

2. **Explicit interaction terms**:
   - ``price_x_is_weekend``: sell_price × is_weekend
   - ``snap_x_dept``: snap_active × dept_id (one-hot style, simplified as binary × dept encoding)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl


# ---------------------------------------------------------------------------
# Target encoding
# ---------------------------------------------------------------------------


class LeaveOneOutEncoder:
    """Leave-one-out target encoder with Gaussian noise regularisation.

    Parameters
    ----------
    smoothing:
        Additive smoothing count: larger values pull group means toward the
        global mean (prevents overfitting on small groups).
    noise_std:
        Standard deviation of Gaussian noise added during ``fit_transform``
        to discourage exact memorisation.
    random_seed:
        Seed for reproducible noise generation.
    """

    def __init__(
        self,
        smoothing: float = 10.0,
        noise_std: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        self.smoothing = smoothing
        self.noise_std = noise_std
        self.random_seed = random_seed
        self._group_stats: dict[str, dict[str, tuple[float, int]]] = {}
        self._global_mean: float = 0.0

    def fit(
        self,
        df: pl.DataFrame,
        group_cols: list[str],
        target_col: str = "sales",
    ) -> "LeaveOneOutEncoder":
        """Compute group means from a training DataFrame.

        Parameters
        ----------
        df:
            Training data with ``group_cols`` and ``target_col``.
        group_cols:
            Columns identifying each encoding group (e.g. ``["item_id"]``).
        target_col:
            Column to encode (sales).
        """
        self._global_mean = float(df[target_col].mean() or 0.0)

        for col in group_cols:
            group_agg = (
                df
                .group_by(col)
                .agg(
                    pl.col(target_col).sum().alias("_sum"),
                    pl.col(target_col).count().alias("_n"),
                )
            )
            self._group_stats[col] = {
                str(row[col]): (float(row["_sum"]), int(row["_n"]))
                for row in group_agg.iter_rows(named=True)
            }
        return self

    def fit_transform(
        self,
        df: pl.DataFrame,
        group_cols: list[str],
        target_col: str = "sales",
    ) -> pl.DataFrame:
        """Fit on ``df`` and transform in-place using LOO (for training).

        Each row's encoded value = (group_sum − row_value) / (group_n − 1),
        blended with the global mean using additive smoothing.

        Parameters
        ----------
        df:
            Training DataFrame.
        group_cols:
            Columns to encode.
        target_col:
            Target column.

        Returns
        -------
        pl.DataFrame
            ``df`` with ``target_enc_{col}`` columns added.
        """
        self.fit(df, group_cols, target_col)

        rng = np.random.default_rng(self.random_seed)
        for col in group_cols:
            enc_values: list[float] = []
            for row in df.iter_rows(named=True):
                key = str(row[col])
                y = float(row[target_col])
                stats = self._group_stats[col].get(key, (0.0, 0))
                group_sum, group_n = stats
                if group_n > 1:
                    loo_mean = (group_sum - y) / (group_n - 1)
                    n_eff = group_n - 1
                else:
                    loo_mean = self._global_mean
                    n_eff = 0
                # Smooth toward global mean
                blended = (n_eff * loo_mean + self.smoothing * self._global_mean) / (
                    n_eff + self.smoothing
                )
                noise = float(rng.normal(0, self.noise_std))
                enc_values.append(blended + noise)

            df = df.with_columns(
                pl.Series(name=f"target_enc_{col}", values=enc_values, dtype=pl.Float64)
            )
        return df

    def transform(
        self,
        df: pl.DataFrame,
        group_cols: list[str],
    ) -> pl.DataFrame:
        """Apply fitted group means to a held-out DataFrame (no LOO).

        Parameters
        ----------
        df:
            Validation or test DataFrame.
        group_cols:
            Columns to encode (must match those used in ``fit``).

        Returns
        -------
        pl.DataFrame
            ``df`` with ``target_enc_{col}`` columns added.
        """
        for col in group_cols:
            enc_map = {
                k: (s / n if n > 0 else self._global_mean)
                for k, (s, n) in self._group_stats.get(col, {}).items()
            }
            df = df.with_columns(
                pl.col(col)
                .cast(pl.Utf8)
                .replace_strict(enc_map, default=str(self._global_mean))
                .cast(pl.Float64)
                .fill_null(self._global_mean)
                .alias(f"target_enc_{col}")
            )
        return df


# ---------------------------------------------------------------------------
# Explicit interaction features
# ---------------------------------------------------------------------------


def add_interaction_features(
    df: pl.DataFrame,
    *,
    price_col: str = "sell_price",
    is_weekend_col: str = "is_weekend",
    snap_col: str = "snap_active",
    dept_col: str = "dept_id",
) -> pl.DataFrame:
    """Add explicit interaction terms to ``df``.

    Features added:
    - ``price_x_is_weekend``: sell_price × is_weekend (0/1).
    - ``snap_x_dept``: snap_active × mean-encoded dept_id (proxy for
        snap impact by department).

    Parameters
    ----------
    df:
        Sales DataFrame with price, calendar and snap features already joined.

    Returns
    -------
    pl.DataFrame
        ``df`` with interaction columns appended.
    """
    exprs: list[pl.Expr] = []

    # price × is_weekend
    if price_col in df.columns and is_weekend_col in df.columns:
        exprs.append(
            (
                pl.col(price_col).fill_null(0.0)
                * pl.col(is_weekend_col).cast(pl.Float64)
            ).alias("price_x_is_weekend")
        )
    else:
        exprs.append(pl.lit(0.0).alias("price_x_is_weekend"))

    # snap × dept (binary × categorical → encoded as snap × dept_index)
    if snap_col in df.columns and dept_col in df.columns:
        # Map dept_id to a float index for the interaction term
        dept_vals = df[dept_col].unique().sort().to_list()
        dept_map = {d: float(i + 1) for i, d in enumerate(dept_vals)}
        exprs.append(
            (
                pl.col(snap_col).cast(pl.Float64)
                * pl.col(dept_col).cast(pl.Utf8).replace(dept_map).cast(pl.Float64)
            ).alias("snap_x_dept")
        )
    else:
        exprs.append(pl.lit(0.0).alias("snap_x_dept"))

    return df.with_columns(exprs)
