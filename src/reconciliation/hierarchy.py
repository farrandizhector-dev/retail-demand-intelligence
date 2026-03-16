"""M5 hierarchy matrix builder for hierarchical forecast reconciliation.

Constructs the summing matrix S and aggregation tags for the 12-level M5
hierarchy (42,840 total series; 30,490 SKU-Store bottom series).

Hierarchy levels
----------------
 1. Total                  —     1 series
 2. State                  —     3 series   (CA, TX, WI)
 3. Store                  —    10 series
 4. Category               —     3 series   (FOODS, HOUSEHOLD, HOBBIES)
 5. Department             —     7 series
 6. State × Category       —     9 series
 7. State × Department     —    21 series
 8. Store × Category       —    30 series
 9. Store × Department     —    70 series
10. Product (item)         — 3,049 series
11. Product × State        — 9,147 series
12. SKU-Store (bottom)     —30,490 series
                                ───────
Total                            42,840 series

The S matrix maps each bottom-level (SKU-Store) series to all aggregate
levels.  S[i, j] = 1 iff bottom series j contributes to aggregate series i.

Memory note
-----------
The full dense S matrix (42,840 × 30,490 × 8 B ≈ 10 GB) is never
materialised.  ``build_hierarchy_matrix`` uses scipy sparse internally and
returns a pandas DataFrame with ``pd.SparseDtype("float64", 0)`` columns,
keeping the footprint proportional to non-zeros (~366 K entries ≈ 3 MB).

For MinTrace reconciliation, use ``build_sub_hierarchy`` which partitions the
catalog into groups of ≤ 1,000 series — well within dense MinT budget.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

LEVEL_NAMES: list[str] = [
    "Total",
    "State",
    "Store",
    "Category",
    "Department",
    "State/Category",
    "State/Department",
    "Store/Category",
    "Store/Department",
    "Product",
    "Product/State",
    "SKU-Store",
]

# Separator for composite identifiers (e.g. "CA/FOODS")
_SEP = "/"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class HierarchyMatrix(NamedTuple):
    """Hierarchy artefacts consumed by ``HierarchicalReconciliation``."""

    S_df: pd.DataFrame
    """Summing matrix.

    - ``unique_id`` column (first column): all 42,840 series IDs.
    - Remaining columns (one per bottom series): 0/1 membership.
    - Column dtype: ``pd.SparseDtype("float64", 0)`` for memory efficiency.
    """

    tags: dict[str, np.ndarray]
    """Aggregation tags.

    Keys = level names (from ``LEVEL_NAMES``).
    Values = 1-D array of unique_id strings at that level.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _bottom_id(item_id: str, store_id: str) -> str:
    """Canonical SKU-Store identifier used as bottom-level series ID."""
    return f"{item_id}_{store_id}"


def _level_id(*parts: str) -> str:
    return _SEP.join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_hierarchy_matrix(catalog_df: pd.DataFrame) -> HierarchyMatrix:
    """Build S matrix and tags from the M5 SKU-Store catalog.

    Parameters
    ----------
    catalog_df:
        One row per unique SKU-Store combination.  Required columns:
        ``item_id``, ``dept_id``, ``cat_id``, ``store_id``, ``state_id``.

    Returns
    -------
    HierarchyMatrix
        Named tuple ``(S_df, tags)``.

    Raises
    ------
    ValueError
        If any required column is missing from ``catalog_df``.

    Notes
    -----
    Bottom-level unique_ids follow the convention ``{item_id}_{store_id}``
    (e.g. ``HOBBIES_1_001_CA_1``), consistent with the M5 dataset standard.
    Aggregate-level unique_ids use ``/`` as the multi-level separator
    (e.g. ``CA/FOODS``).
    """
    required = {"item_id", "dept_id", "cat_id", "store_id", "state_id"}
    missing = required - set(catalog_df.columns)
    if missing:
        raise ValueError(f"catalog_df missing required columns: {sorted(missing)}")

    df = (
        catalog_df[list(required)]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 1. Bottom-level IDs (SKU-Store)
    # ------------------------------------------------------------------
    bottom_ids: list[str] = [
        _bottom_id(row.item_id, row.store_id)
        for row in df.itertuples(index=False)
    ]
    bottom_arr = np.array(bottom_ids)
    n_bottom = len(bottom_arr)

    # ------------------------------------------------------------------
    # 2. Build per-level uid → list[col_index] mapping
    # ------------------------------------------------------------------
    LevelSpec = tuple[str, list[tuple[str, list[int]]]]  # (name, [(uid, [col_idxs])])

    def _build_level(
        name: str,
        key_fn,
    ) -> tuple[str, dict[str, list[int]]]:
        uid_map: dict[str, list[int]] = {}
        for i, row in enumerate(df.itertuples(index=False)):
            key = key_fn(row)
            uid_map.setdefault(key, []).append(i)
        return name, uid_map

    raw_levels: list[tuple[str, dict[str, list[int]]]] = [
        _build_level("Total",            lambda r: "Total"),
        _build_level("State",            lambda r: r.state_id),
        _build_level("Store",            lambda r: r.store_id),
        _build_level("Category",         lambda r: r.cat_id),
        _build_level("Department",       lambda r: r.dept_id),
        _build_level("State/Category",   lambda r: _level_id(r.state_id, r.cat_id)),
        _build_level("State/Department", lambda r: _level_id(r.state_id, r.dept_id)),
        _build_level("Store/Category",   lambda r: _level_id(r.store_id, r.cat_id)),
        _build_level("Store/Department", lambda r: _level_id(r.store_id, r.dept_id)),
        _build_level("Product",          lambda r: r.item_id),
        _build_level("Product/State",    lambda r: _level_id(r.item_id, r.state_id)),
        _build_level("SKU-Store",        lambda r: _bottom_id(r.item_id, r.store_id)),
    ]

    # ------------------------------------------------------------------
    # 3. Collect tags and S-matrix coordinates
    # ------------------------------------------------------------------
    tags: dict[str, np.ndarray] = {}
    all_uids: list[str] = []
    coo_rows: list[int] = []
    coo_cols: list[int] = []

    row_idx = 0
    for level_name, uid_map in raw_levels:
        sorted_uids = sorted(uid_map.keys())
        tags[level_name] = np.array(sorted_uids)
        for uid in sorted_uids:
            all_uids.append(uid)
            for col_j in uid_map[uid]:
                coo_rows.append(row_idx)
                coo_cols.append(col_j)
            row_idx += 1

    n_total = len(all_uids)

    # ------------------------------------------------------------------
    # 4. Build scipy sparse → pandas SparseDtype S_df
    # ------------------------------------------------------------------
    S_sparse = sp.csr_matrix(
        (np.ones(len(coo_rows), dtype=np.float64), (coo_rows, coo_cols)),
        shape=(n_total, n_bottom),
    )

    # from_spmatrix avoids full densification
    S_df_indexed = pd.DataFrame.sparse.from_spmatrix(
        S_sparse,
        index=all_uids,
        columns=bottom_arr,
    )
    S_df_indexed.index.name = "unique_id"
    S_df = S_df_indexed.reset_index()

    logger.info(
        "Hierarchy matrix: %d total series × %d bottom series, nnz=%d",
        n_total,
        n_bottom,
        S_sparse.nnz,
    )

    return HierarchyMatrix(S_df=S_df, tags=tags)


def build_sub_hierarchy(
    catalog_df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> dict[tuple, HierarchyMatrix]:
    """Build one HierarchyMatrix per partition for sub-hierarchy MinTrace.

    Partitioning by ``("store_id", "dept_id")`` yields 70 groups each
    containing ~435 bottom-level series — well within the dense MinT budget.

    Parameters
    ----------
    catalog_df:
        Full M5 catalog (must have the 5 standard hierarchy columns).
    group_cols:
        Columns to partition on.  Defaults to ``["store_id", "dept_id"]``.

    Returns
    -------
    dict
        Maps ``(group_key, ...)`` tuples → ``HierarchyMatrix``.
    """
    if group_cols is None:
        group_cols = ["store_id", "dept_id"]

    sub: dict[tuple, HierarchyMatrix] = {}
    for keys, grp in catalog_df.groupby(group_cols):
        key = keys if isinstance(keys, tuple) else (keys,)
        sub[key] = build_hierarchy_matrix(grp.reset_index(drop=True))

    logger.info(
        "Sub-hierarchies: %d groups (partition=%s)",
        len(sub),
        group_cols,
    )
    return sub


def get_level_for_series(uid: str, tags: dict[str, np.ndarray]) -> str | None:
    """Return the hierarchy level name for *uid*, or ``None`` if not found."""
    for level, arr in tags.items():
        if uid in arr:
            return level
    return None
