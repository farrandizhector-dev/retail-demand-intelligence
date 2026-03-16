"""Unit tests for SCD Type 2 manager (spec §7.2)."""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from src.classification.scd_manager import (
    SCD_MAX_DATE,
    TRACKED_COLS,
    apply_scd_type2,
    detect_classification_changes,
    save_dim_product,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_current_df(items: list[dict]) -> pl.DataFrame:
    """Build a fresh classification DataFrame."""
    return pl.DataFrame(items)


def _make_dim_df(items: list[dict]) -> pl.DataFrame:
    """Build an existing dim_product DataFrame with SCD columns."""
    return pl.DataFrame(items)


def _base_item(
    item_id="ITEM_001",
    abc="A",
    xyz="X",
    demand="smooth",
    version=1,
    is_current=True,
    valid_from=date(2026, 1, 1),
    valid_to=None,
    item_key=1,
):
    return {
        "item_key": item_key,
        "item_id": item_id,
        "abc_class": abc,
        "xyz_class": xyz,
        "demand_class": demand,
        "valid_from": valid_from,
        "valid_to": valid_to or SCD_MAX_DATE,
        "is_current": is_current,
        "version": version,
    }


EFFECTIVE_DATE = date(2026, 4, 1)


# ---------------------------------------------------------------------------
# detect_classification_changes
# ---------------------------------------------------------------------------


class TestDetectClassificationChanges:
    def test_no_change_returns_empty(self):
        current = _make_current_df([{"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"}])
        previous = _make_dim_df([_base_item("A", "A", "X", "smooth")])
        changes = detect_classification_changes(current, previous)
        assert len(changes) == 0

    def test_abc_class_change_detected(self):
        current = _make_current_df([{"item_id": "A", "abc_class": "B", "xyz_class": "X", "demand_class": "smooth"}])
        previous = _make_dim_df([_base_item("A", "A", "X", "smooth")])
        changes = detect_classification_changes(current, previous)
        assert len(changes) == 1
        assert changes["item_id"][0] == "A"

    def test_xyz_class_change_detected(self):
        current = _make_current_df([{"item_id": "A", "abc_class": "A", "xyz_class": "Y", "demand_class": "smooth"}])
        previous = _make_dim_df([_base_item("A", "A", "X", "smooth")])
        changes = detect_classification_changes(current, previous)
        assert len(changes) == 1

    def test_demand_class_change_detected(self):
        current = _make_current_df([{"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "erratic"}])
        previous = _make_dim_df([_base_item("A", "A", "X", "smooth")])
        changes = detect_classification_changes(current, previous)
        assert len(changes) == 1

    def test_new_item_included(self):
        """Item in current but not previous is treated as new → included."""
        current = _make_current_df([{"item_id": "NEW", "abc_class": "C", "xyz_class": "Z", "demand_class": "lumpy"}])
        previous = _make_dim_df([_base_item("EXISTING", "A", "X", "smooth")])
        changes = detect_classification_changes(current, previous)
        assert any(changes["item_id"] == "NEW")

    def test_old_cols_null_for_new_item(self):
        current = _make_current_df([{"item_id": "NEW", "abc_class": "C", "xyz_class": "Z", "demand_class": "lumpy"}])
        previous = _make_dim_df([_base_item("OTHER", "A", "X", "smooth")])
        changes = detect_classification_changes(current, previous)
        new_row = changes.filter(pl.col("item_id") == "NEW")
        assert new_row["old_abc_class"][0] is None

    def test_multiple_items_partial_changes(self):
        current = _make_current_df([
            {"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"},
            {"item_id": "B", "abc_class": "C", "xyz_class": "Z", "demand_class": "lumpy"},
        ])
        previous = _make_dim_df([
            _base_item("A", "A", "X", "smooth", item_key=1),
            _base_item("B", "B", "Y", "erratic", item_key=2),
        ])
        changes = detect_classification_changes(current, previous)
        assert len(changes) == 1
        assert changes["item_id"][0] == "B"

    def test_uses_only_current_rows_from_previous(self):
        """Non-current (historical) rows in previous should not affect comparison."""
        current = _make_current_df([{"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"}])
        previous = _make_dim_df([
            _base_item("A", "B", "Y", "erratic", is_current=False, version=1, item_key=1),
            _base_item("A", "A", "X", "smooth", is_current=True, version=2, item_key=2),
        ])
        changes = detect_classification_changes(current, previous)
        assert len(changes) == 0


# ---------------------------------------------------------------------------
# apply_scd_type2 — first load
# ---------------------------------------------------------------------------


class TestApplyScdType2FirstLoad:
    def test_empty_dim_produces_all_current(self):
        dim = pl.DataFrame(schema={
            "item_key": pl.Int64, "item_id": pl.Utf8,
            "abc_class": pl.Utf8, "xyz_class": pl.Utf8, "demand_class": pl.Utf8,
            "valid_from": pl.Date, "valid_to": pl.Date,
            "is_current": pl.Boolean, "version": pl.Int64,
        })
        new_clf = _make_current_df([
            {"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"},
            {"item_id": "B", "abc_class": "B", "xyz_class": "Y", "demand_class": "erratic"},
        ])
        result = apply_scd_type2(dim, new_clf, EFFECTIVE_DATE)
        assert len(result) == 2
        assert result["is_current"].all()

    def test_first_load_version_is_1(self):
        dim = pl.DataFrame(schema={
            "item_key": pl.Int64, "item_id": pl.Utf8,
            "abc_class": pl.Utf8, "xyz_class": pl.Utf8, "demand_class": pl.Utf8,
            "valid_from": pl.Date, "valid_to": pl.Date,
            "is_current": pl.Boolean, "version": pl.Int64,
        })
        new_clf = _make_current_df([{"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"}])
        result = apply_scd_type2(dim, new_clf, EFFECTIVE_DATE)
        assert result["version"][0] == 1

    def test_first_load_valid_to_is_max(self):
        dim = pl.DataFrame(schema={
            "item_key": pl.Int64, "item_id": pl.Utf8,
            "abc_class": pl.Utf8, "xyz_class": pl.Utf8, "demand_class": pl.Utf8,
            "valid_from": pl.Date, "valid_to": pl.Date,
            "is_current": pl.Boolean, "version": pl.Int64,
        })
        new_clf = _make_current_df([{"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"}])
        result = apply_scd_type2(dim, new_clf, EFFECTIVE_DATE)
        assert result["valid_to"][0] == SCD_MAX_DATE

    def test_first_load_valid_from_is_effective_date(self):
        dim = pl.DataFrame(schema={
            "item_key": pl.Int64, "item_id": pl.Utf8,
            "abc_class": pl.Utf8, "xyz_class": pl.Utf8, "demand_class": pl.Utf8,
            "valid_from": pl.Date, "valid_to": pl.Date,
            "is_current": pl.Boolean, "version": pl.Int64,
        })
        new_clf = _make_current_df([{"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"}])
        result = apply_scd_type2(dim, new_clf, EFFECTIVE_DATE)
        assert result["valid_from"][0] == EFFECTIVE_DATE

    def test_first_load_item_keys_sequential(self):
        dim = pl.DataFrame(schema={
            "item_key": pl.Int64, "item_id": pl.Utf8,
            "abc_class": pl.Utf8, "xyz_class": pl.Utf8, "demand_class": pl.Utf8,
            "valid_from": pl.Date, "valid_to": pl.Date,
            "is_current": pl.Boolean, "version": pl.Int64,
        })
        new_clf = _make_current_df([
            {"item_id": "A", "abc_class": "A", "xyz_class": "X", "demand_class": "smooth"},
            {"item_id": "B", "abc_class": "B", "xyz_class": "Y", "demand_class": "erratic"},
        ])
        result = apply_scd_type2(dim, new_clf, EFFECTIVE_DATE)
        keys = sorted(result["item_key"].to_list())
        assert keys == [1, 2]


# ---------------------------------------------------------------------------
# apply_scd_type2 — with changes
# ---------------------------------------------------------------------------


class TestApplyScdType2WithChanges:
    def _initial_dim(self):
        return _make_dim_df([_base_item("A", "A", "X", "smooth", item_key=1)])

    def _new_clf_changed(self):
        return _make_current_df([{"item_id": "A", "abc_class": "B", "xyz_class": "X", "demand_class": "smooth"}])

    def test_old_row_is_closed(self):
        result = apply_scd_type2(self._initial_dim(), self._new_clf_changed(), EFFECTIVE_DATE)
        old_rows = result.filter((pl.col("item_id") == "A") & ~pl.col("is_current"))
        assert len(old_rows) == 1

    def test_old_row_valid_to_is_day_before_effective(self):
        result = apply_scd_type2(self._initial_dim(), self._new_clf_changed(), EFFECTIVE_DATE)
        old_rows = result.filter((pl.col("item_id") == "A") & ~pl.col("is_current"))
        expected_valid_to = EFFECTIVE_DATE - timedelta(days=1)
        assert old_rows["valid_to"][0] == expected_valid_to

    def test_new_row_is_current(self):
        result = apply_scd_type2(self._initial_dim(), self._new_clf_changed(), EFFECTIVE_DATE)
        new_rows = result.filter((pl.col("item_id") == "A") & pl.col("is_current"))
        assert len(new_rows) == 1

    def test_new_row_version_incremented(self):
        result = apply_scd_type2(self._initial_dim(), self._new_clf_changed(), EFFECTIVE_DATE)
        new_rows = result.filter((pl.col("item_id") == "A") & pl.col("is_current"))
        assert new_rows["version"][0] == 2

    def test_new_row_has_updated_classification(self):
        result = apply_scd_type2(self._initial_dim(), self._new_clf_changed(), EFFECTIVE_DATE)
        new_rows = result.filter((pl.col("item_id") == "A") & pl.col("is_current"))
        assert new_rows["abc_class"][0] == "B"

    def test_new_row_item_key_is_new(self):
        result = apply_scd_type2(self._initial_dim(), self._new_clf_changed(), EFFECTIVE_DATE)
        new_rows = result.filter((pl.col("item_id") == "A") & pl.col("is_current"))
        assert new_rows["item_key"][0] != 1  # Must be a new surrogate key

    def test_total_row_count(self):
        """One unchanged + one old closed + one new = 2 rows for changed item."""
        result = apply_scd_type2(self._initial_dim(), self._new_clf_changed(), EFFECTIVE_DATE)
        assert len(result) == 2  # old (closed) + new (current)

    def test_no_change_item_untouched(self):
        dim = _make_dim_df([
            _base_item("A", "A", "X", "smooth", item_key=1),
            _base_item("B", "B", "Y", "erratic", item_key=2),
        ])
        new_clf = _make_current_df([
            {"item_id": "A", "abc_class": "C", "xyz_class": "X", "demand_class": "smooth"},  # changed
            {"item_id": "B", "abc_class": "B", "xyz_class": "Y", "demand_class": "erratic"},  # unchanged
        ])
        result = apply_scd_type2(dim, new_clf, EFFECTIVE_DATE)
        # B should still be current with version 1
        b_current = result.filter((pl.col("item_id") == "B") & pl.col("is_current"))
        assert len(b_current) == 1
        assert b_current["version"][0] == 1

    def test_exactly_one_current_per_item(self):
        dim = _make_dim_df([_base_item("A", "A", "X", "smooth", item_key=1)])
        new_clf = _make_current_df([{"item_id": "A", "abc_class": "B", "xyz_class": "X", "demand_class": "smooth"}])
        result = apply_scd_type2(dim, new_clf, EFFECTIVE_DATE)
        current_count = result.filter(pl.col("is_current")).group_by("item_id").len()
        assert (current_count["len"] == 1).all()


# ---------------------------------------------------------------------------
# save_dim_product
# ---------------------------------------------------------------------------


class TestSaveDimProduct:
    def test_saves_parquet(self, tmp_path):
        dim = _make_dim_df([_base_item()])
        output_path = tmp_path / "dim_product_scd.parquet"
        result = save_dim_product(dim, output_path)
        assert result.exists()

    def test_readable_back(self, tmp_path):
        dim = _make_dim_df([_base_item()])
        output_path = tmp_path / "dim_product_scd.parquet"
        save_dim_product(dim, output_path)
        loaded = pl.read_parquet(output_path)
        assert len(loaded) == 1
        assert loaded["item_id"][0] == "ITEM_001"

    def test_creates_parent_dirs(self, tmp_path):
        dim = _make_dim_df([_base_item()])
        output_path = tmp_path / "nested" / "deep" / "dim.parquet"
        save_dim_product(dim, output_path)
        assert output_path.exists()
