"""SCD Type 2 manager for dim_product dimension (spec §7.2).

Tracks changes in abc_class, xyz_class, demand_class over time.
Each reclassification cycle (every 90 days) may close old rows
and insert new ones with updated surrogate keys and version numbers.

dim_product schema:
  item_key     : int   (surrogate PK, auto-increment)
  item_id      : str   (business key, e.g. "HOBBIES_1_001")
  dept_id      : str
  cat_id       : str
  abc_class    : str   (A | B | C)
  xyz_class    : str   (X | Y | Z)
  demand_class : str   (smooth | erratic | intermittent | lumpy)
  valid_from   : date
  valid_to     : date  (9999-12-31 if current)
  is_current   : bool
  version      : int   (1-based)
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

# Sentinel date for open-ended validity
SCD_MAX_DATE: date = date(9999, 12, 31)

# Columns tracked for change detection
TRACKED_COLS: list[str] = ["abc_class", "xyz_class", "demand_class"]

# All SCD metadata columns (must exist in dim_product_df)
SCD_COLS: list[str] = ["valid_from", "valid_to", "is_current", "version", "item_key"]

# Optional passthrough columns from new_classification_df
_OPTIONAL_EXTRAS: list[str] = [
    "dept_id",
    "cat_id",
    "avg_daily_demand",
    "pct_zero_days",
    "adi",
    "cv_squared",
]


def detect_classification_changes(
    current_df: pl.DataFrame,
    previous_df: pl.DataFrame,
) -> pl.DataFrame:
    """Detect items whose classification changed vs the current SCD snapshot.

    Parameters
    ----------
    current_df:
        Fresh classification — columns: [item_id, abc_class, xyz_class, demand_class].
    previous_df:
        Full dim_product table (all SCD versions). May include non-current rows;
        we filter to is_current=True before comparing.

    Returns
    -------
    pl.DataFrame
        Rows from current_df where ANY tracked column differs from the is_current row
        in previous_df, PLUS new items (not in previous_df).
        Columns: [item_id, abc_class, xyz_class, demand_class,
                  old_abc_class, old_xyz_class, old_demand_class]
        old_* columns are null for brand-new items.
    """
    # Filter previous to only current rows
    if "is_current" in previous_df.columns and len(previous_df) > 0:
        prev_current = previous_df.filter(pl.col("is_current"))
    else:
        prev_current = previous_df

    # Select only the columns we need from previous
    prev_cols = ["item_id"] + TRACKED_COLS
    available_prev_cols = [c for c in prev_cols if c in prev_current.columns]
    prev_slim = prev_current.select(available_prev_cols)

    # Rename tracked cols in previous to old_*
    rename_map = {col: f"old_{col}" for col in TRACKED_COLS if col in prev_slim.columns}
    prev_renamed = prev_slim.rename(rename_map)

    # Left join: current ← previous (to detect changes and new items)
    joined = current_df.join(prev_renamed, on="item_id", how="left")

    # Build change condition: ANY tracked col differs OR item is new (old_* is null)
    change_conditions = []
    for col in TRACKED_COLS:
        old_col = f"old_{col}"
        if old_col in joined.columns:
            # Changed: old value is not null AND differs from new value
            changed = (pl.col(old_col).is_not_null()) & (pl.col(col) != pl.col(old_col))
            # New: old value is null (item didn't exist before)
            is_new = pl.col(old_col).is_null()
            change_conditions.append(changed | is_new)

    if not change_conditions:
        # No tracked columns found — return empty with expected schema
        return pl.DataFrame(
            schema={
                "item_id": pl.Utf8,
                "abc_class": pl.Utf8,
                "xyz_class": pl.Utf8,
                "demand_class": pl.Utf8,
                "old_abc_class": pl.Utf8,
                "old_xyz_class": pl.Utf8,
                "old_demand_class": pl.Utf8,
            }
        )

    # Combine with OR — any change triggers
    combined_condition = change_conditions[0]
    for cond in change_conditions[1:]:
        combined_condition = combined_condition | cond

    # Deduplicate: if we already have all old_* cols from join, just filter
    result = joined.filter(combined_condition)

    # Ensure old_* columns exist (add nulls if missing)
    for col in TRACKED_COLS:
        old_col = f"old_{col}"
        if old_col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias(old_col))

    # Select final columns
    output_cols = ["item_id"] + TRACKED_COLS + [f"old_{c}" for c in TRACKED_COLS]
    return result.select(output_cols)


def apply_scd_type2(
    dim_product_df: pl.DataFrame,
    new_classification_df: pl.DataFrame,
    effective_date: date,
) -> pl.DataFrame:
    """Apply SCD Type 2 logic to the dim_product dimension.

    Parameters
    ----------
    dim_product_df:
        Full dimension table (all SCD versions). May be empty (first load).
    new_classification_df:
        Fresh classification — required columns: [item_id, abc_class, xyz_class,
        demand_class]. Optional extras: dept_id, cat_id, avg_daily_demand,
        pct_zero_days, adi, cv_squared.
    effective_date:
        Date of reclassification (becomes valid_from for new rows).

    Returns
    -------
    pl.DataFrame
        Updated full dimension table.
    """
    is_empty = len(dim_product_df) == 0

    if is_empty:
        return _first_load(new_classification_df, effective_date)

    return _incremental_update(dim_product_df, new_classification_df, effective_date)


def _build_new_rows(
    items_df: pl.DataFrame,
    start_key: int,
    version_map: dict[str, int],
    effective_date: date,
) -> pl.DataFrame:
    """Build new SCD rows for a set of items."""
    n = len(items_df)
    if n == 0:
        return pl.DataFrame()

    # Assign sequential surrogate keys
    keys = list(range(start_key, start_key + n))

    # Build version list
    versions = [version_map.get(row["item_id"], 1) for row in items_df.iter_rows(named=True)]

    result = items_df.with_columns([
        pl.Series("item_key", keys, dtype=pl.Int64),
        pl.Series("version", versions, dtype=pl.Int64),
        pl.lit(effective_date).cast(pl.Date).alias("valid_from"),
        pl.lit(SCD_MAX_DATE).cast(pl.Date).alias("valid_to"),
        pl.lit(True).alias("is_current"),
    ])

    return result


def _first_load(
    new_classification_df: pl.DataFrame,
    effective_date: date,
) -> pl.DataFrame:
    """Handle first load: all items are new, version=1, keys 1..N."""
    n = len(new_classification_df)
    keys = list(range(1, n + 1))
    versions = [1] * n

    result = new_classification_df.with_columns([
        pl.Series("item_key", keys, dtype=pl.Int64),
        pl.Series("version", versions, dtype=pl.Int64),
        pl.lit(effective_date).cast(pl.Date).alias("valid_from"),
        pl.lit(SCD_MAX_DATE).cast(pl.Date).alias("valid_to"),
        pl.lit(True).alias("is_current"),
    ])

    # Add optional extra columns as null if not present
    for col in _OPTIONAL_EXTRAS:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))

    return result


def _incremental_update(
    dim_product_df: pl.DataFrame,
    new_classification_df: pl.DataFrame,
    effective_date: date,
) -> pl.DataFrame:
    """Handle incremental update with SCD Type 2 logic."""
    # Step 1: Detect changed/new items
    changes = detect_classification_changes(new_classification_df, dim_product_df)
    changed_item_ids = changes["item_id"].to_list() if len(changes) > 0 else []

    if not changed_item_ids:
        # Nothing changed — return as-is (but ensure is_current rows reflect new classification)
        return dim_product_df

    # Step 2: Close old current rows for changed items
    close_date = effective_date - timedelta(days=1)

    # Split dim into: rows for changed items (current) vs everything else
    changed_set = set(changed_item_ids)

    # All rows for changed items that are currently active
    rows_to_close = dim_product_df.filter(
        pl.col("item_id").is_in(changed_item_ids) & pl.col("is_current")
    )
    # Rows not affected (unchanged current + all historical)
    unchanged_rows = dim_product_df.filter(
        ~(pl.col("item_id").is_in(changed_item_ids) & pl.col("is_current"))
    )

    # Close old rows
    closed_rows = rows_to_close.with_columns([
        pl.lit(close_date).cast(pl.Date).alias("valid_to"),
        pl.lit(False).alias("is_current"),
    ])

    # Step 3: Determine next surrogate key
    max_key = dim_product_df["item_key"].max()
    next_key = (max_key or 0) + 1

    # Step 4: Build version map (version = old_version + 1 for existing, 1 for brand new)
    # Get current versions of changed items
    version_map: dict[str, int] = {}
    if len(rows_to_close) > 0:
        for row in rows_to_close.iter_rows(named=True):
            version_map[row["item_id"]] = row["version"] + 1

    # Items in new_classification_df that changed or are new
    new_rows_df = new_classification_df.filter(pl.col("item_id").is_in(changed_item_ids))

    # Build new rows
    new_rows = _build_new_rows(new_rows_df, next_key, version_map, effective_date)

    # Add optional extras as null if not present
    for col in _OPTIONAL_EXTRAS:
        if col not in new_rows.columns:
            new_rows = new_rows.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))

    # Step 5: Concatenate everything
    # Align schemas before concat
    all_frames = [unchanged_rows, closed_rows, new_rows]

    # Get union of all columns
    all_cols: set[str] = set()
    for frame in all_frames:
        all_cols.update(frame.columns)

    aligned = []
    for frame in all_frames:
        if len(frame) == 0:
            continue
        for col in all_cols:
            if col not in frame.columns:
                # Infer dtype from other frames
                dtype = pl.Utf8
                for other in all_frames:
                    if col in other.columns and len(other) > 0:
                        dtype = other[col].dtype
                        break
                frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        aligned.append(frame.select(sorted(all_cols)))

    if not aligned:
        return dim_product_df

    return pl.concat(aligned)


def save_dim_product(dim_product_df: pl.DataFrame, output_path: Path) -> Path:
    """Write dim_product DataFrame to parquet.

    Parameters
    ----------
    dim_product_df:
        Full dimension table to persist.
    output_path:
        Target file path (.parquet).

    Returns
    -------
    Path
        The output_path that was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dim_product_df.write_parquet(output_path)
    logger.info("Saved dim_product (%d rows) to %s", len(dim_product_df), output_path)
    return output_path
