"""Unit tests for src/reconciliation/hierarchy.py.

Tests verify:
- S matrix dimensions match expected hierarchy counts
- tags keys match LEVEL_NAMES
- Bottom-level IDs follow {item_id}_{store_id} convention
- S matrix is binary and correctly encodes membership
- Sub-hierarchy partitioning produces expected number of groups
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from src.reconciliation.hierarchy import (
    LEVEL_NAMES,
    HierarchyMatrix,
    _bottom_id,
    _level_id,
    build_hierarchy_matrix,
    build_sub_hierarchy,
    get_level_for_series,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_catalog() -> pd.DataFrame:
    """3 items × 2 stores × 2 states — 6 SKU-Store bottom series."""
    rows = [
        # item_id       dept_id      cat_id     store_id   state_id
        ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "TX_1", "TX"),
    ]
    return pd.DataFrame(rows, columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"])


@pytest.fixture
def small_catalog() -> pd.DataFrame:
    """10 items × 3 stores × 2 states — 30 SKU-Store bottom series."""
    items = [
        ("FOODS_1_001", "FOODS_1", "FOODS"),
        ("FOODS_1_002", "FOODS_1", "FOODS"),
        ("FOODS_2_001", "FOODS_2", "FOODS"),
        ("FOODS_2_002", "FOODS_2", "FOODS"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES"),
        ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES"),
        ("HOBBIES_2_001", "HOBBIES_2", "HOBBIES"),
        ("HOUSEHOLD_1_001", "HOUSEHOLD_1", "HOUSEHOLD"),
        ("HOUSEHOLD_1_002", "HOUSEHOLD_1", "HOUSEHOLD"),
        ("HOUSEHOLD_2_001", "HOUSEHOLD_2", "HOUSEHOLD"),
    ]
    stores = [("CA_1", "CA"), ("TX_1", "TX"), ("WI_1", "WI")]
    rows = []
    for item, dept, cat in items:
        for store, state in stores:
            rows.append((item, dept, cat, store, state))
    return pd.DataFrame(rows, columns=["item_id", "dept_id", "cat_id", "store_id", "state_id"])


# ---------------------------------------------------------------------------
# build_hierarchy_matrix
# ---------------------------------------------------------------------------


class TestBuildHierarchyMatrix:
    def test_returns_hierarchy_matrix_namedtuple(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        assert isinstance(hm, HierarchyMatrix)

    def test_s_df_has_unique_id_column(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        assert "unique_id" in hm.S_df.columns

    def test_s_df_shape_tiny_catalog(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        n_bottom = 6  # 3 items × 2 stores
        # Total rows: sum across all 12 levels
        # Level counts for tiny_catalog:
        # Total=1, State=2(CA,TX), Store=2(CA_1,TX_1), Category=2(FOODS,HOBBIES)
        # Department=2(FOODS_1,HOBBIES_1), State/Cat=4, State/Dept=4
        # Store/Cat=4, Store/Dept=4, Product=3, Product/State=6, SKU-Store=6
        assert hm.S_df.shape[1] == n_bottom + 1  # +1 for unique_id column

    def test_s_df_n_rows_equals_total_series(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        # Count distinct series at each level
        cat = tiny_catalog.drop_duplicates()
        expected = (
            1  # Total
            + cat["state_id"].nunique()              # State
            + cat["store_id"].nunique()              # Store
            + cat["cat_id"].nunique()                # Category
            + cat["dept_id"].nunique()               # Department
            + cat[["state_id", "cat_id"]].drop_duplicates().shape[0]
            + cat[["state_id", "dept_id"]].drop_duplicates().shape[0]
            + cat[["store_id", "cat_id"]].drop_duplicates().shape[0]
            + cat[["store_id", "dept_id"]].drop_duplicates().shape[0]
            + cat["item_id"].nunique()               # Product
            + cat[["item_id", "state_id"]].drop_duplicates().shape[0]
            + cat[["item_id", "store_id"]].drop_duplicates().shape[0]  # SKU-Store
        )
        n_rows = len(hm.S_df)
        assert n_rows == expected, f"Expected {expected} rows, got {n_rows}"

    def test_tags_has_all_12_levels(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        assert set(hm.tags.keys()) == set(LEVEL_NAMES)

    def test_tags_total_has_one_entry(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        assert list(hm.tags["Total"]) == ["Total"]

    def test_bottom_ids_follow_convention(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        bottom_ids = set(hm.tags["SKU-Store"])
        expected = {
            "FOODS_1_001_CA_1", "FOODS_1_001_TX_1",
            "FOODS_1_002_CA_1", "FOODS_1_002_TX_1",
            "HOBBIES_1_001_CA_1", "HOBBIES_1_001_TX_1",
        }
        assert bottom_ids == expected

    def test_s_matrix_is_binary(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        # Exclude unique_id column
        data_cols = [c for c in hm.S_df.columns if c != "unique_id"]
        vals = hm.S_df[data_cols].values
        # SparseDtype — convert to dense for assertion
        if hasattr(vals, "todense"):
            vals = vals.todense()
        unique_vals = set(np.unique(vals.astype(float)))
        assert unique_vals <= {0.0, 1.0}, f"Unexpected values: {unique_vals}"

    def test_total_row_is_all_ones(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        total_row = hm.S_df[hm.S_df["unique_id"] == "Total"]
        assert not total_row.empty
        data = total_row.drop(columns="unique_id").values[0]
        assert np.all(data == 1.0), "Total row should be all 1s"

    def test_bottom_row_is_identity(self, tiny_catalog):
        """Each bottom series row should have exactly one 1 (the diagonal)."""
        hm = build_hierarchy_matrix(tiny_catalog)
        bottom_ids = set(hm.tags["SKU-Store"])
        bottom_rows = hm.S_df[hm.S_df["unique_id"].isin(bottom_ids)]
        for _, row in bottom_rows.iterrows():
            data = row.drop("unique_id").values.astype(float)
            assert data.sum() == 1.0, f"Bottom row {row['unique_id']} should have exactly one 1"
            assert data.max() == 1.0

    def test_state_row_membership(self, tiny_catalog):
        """CA row should cover FOODS_1_001_CA_1, FOODS_1_002_CA_1, HOBBIES_1_001_CA_1."""
        hm = build_hierarchy_matrix(tiny_catalog)
        ca_row = hm.S_df[hm.S_df["unique_id"] == "CA"]
        assert not ca_row.empty
        data = ca_row.drop(columns="unique_id").squeeze()
        # Columns with 1 should be the CA bottom series
        ca_series = {col for col in data.index if data[col] == 1.0}
        expected_ca = {"FOODS_1_001_CA_1", "FOODS_1_002_CA_1", "HOBBIES_1_001_CA_1"}
        assert ca_series == expected_ca

    def test_missing_column_raises(self):
        bad_df = pd.DataFrame({"item_id": ["A"], "store_id": ["S1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_hierarchy_matrix(bad_df)

    def test_small_catalog_dimensions(self, small_catalog):
        """10 items × 3 stores × 3 states → 30 bottom series."""
        hm = build_hierarchy_matrix(small_catalog)
        n_bottom = len(small_catalog)  # 30
        assert hm.S_df.shape[1] == n_bottom + 1  # +1 for unique_id
        assert len(hm.tags["SKU-Store"]) == n_bottom


# ---------------------------------------------------------------------------
# build_sub_hierarchy
# ---------------------------------------------------------------------------


class TestBuildSubHierarchy:
    def test_default_partition_store_dept(self, small_catalog):
        subs = build_sub_hierarchy(small_catalog)
        # 3 stores × 4 departments = 12 non-empty groups
        n_stores = small_catalog["store_id"].nunique()
        n_depts = small_catalog["dept_id"].nunique()
        assert len(subs) == n_stores * n_depts

    def test_each_sub_is_hierarchy_matrix(self, small_catalog):
        subs = build_sub_hierarchy(small_catalog)
        for key, hm in subs.items():
            assert isinstance(hm, HierarchyMatrix)

    def test_custom_partition_state(self, small_catalog):
        subs = build_sub_hierarchy(small_catalog, group_cols=["state_id"])
        n_states = small_catalog["state_id"].nunique()
        assert len(subs) == n_states

    def test_sub_bottom_ids_disjoint(self, small_catalog):
        """Bottom-level series in different sub-hierarchies must not overlap."""
        subs = build_sub_hierarchy(small_catalog, group_cols=["store_id"])
        all_bottom: list[set] = []
        for hm in subs.values():
            bottom = set(hm.tags["SKU-Store"])
            all_bottom.append(bottom)
        # Check pairwise disjointness
        for i, s1 in enumerate(all_bottom):
            for j, s2 in enumerate(all_bottom):
                if i != j:
                    overlap = s1 & s2
                    assert not overlap, f"Groups {i} and {j} share series: {overlap}"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_bottom_id_format(self):
        assert _bottom_id("FOODS_1_001", "CA_1") == "FOODS_1_001_CA_1"

    def test_level_id_separator(self):
        assert _level_id("CA", "FOODS") == "CA/FOODS"
        assert _level_id("CA_1", "FOODS_1", "001") == "CA_1/FOODS_1/001"

    def test_get_level_for_series_known(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        assert get_level_for_series("Total", hm.tags) == "Total"
        assert get_level_for_series("CA", hm.tags) == "State"
        assert get_level_for_series("FOODS_1_001_CA_1", hm.tags) == "SKU-Store"

    def test_get_level_for_series_unknown(self, tiny_catalog):
        hm = build_hierarchy_matrix(tiny_catalog)
        assert get_level_for_series("NONEXISTENT_SERIES", hm.tags) is None
