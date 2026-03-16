"""Unit tests for src/evaluation/shap_analysis.py.

Tests verify:
- compute_shap_values returns correct shape
- generate_shap_summary returns top-N features with correct format
- all SHAP mean_abs values are >= 0
- shap_by_segment returns correct keys
- export_shap_for_frontend writes valid JSON files
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.evaluation.shap_analysis import (
    TOP_N_FEATURES,
    compute_shap_values,
    export_shap_for_frontend,
    generate_shap_summary,
    shap_by_segment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_lgbm_model():
    """Train a tiny LightGBM model on synthetic data."""
    rng = np.random.default_rng(42)
    n, p = 200, 10
    X = pd.DataFrame(rng.normal(0, 1, (n, p)), columns=[f"feat_{i}" for i in range(p)])
    y = rng.normal(0, 1, n)
    ds = lgb.Dataset(X, y)
    model = lgb.train(
        {"objective": "regression", "verbose": -1, "num_leaves": 8},
        ds,
        num_boost_round=20,
    )
    return model, X


# ---------------------------------------------------------------------------
# compute_shap_values
# ---------------------------------------------------------------------------


class TestComputeShapValues:
    def test_output_shape(self, tiny_lgbm_model):
        model, X = tiny_lgbm_model
        sv = compute_shap_values(model, X)
        assert sv.shape == X.shape

    def test_output_is_float(self, tiny_lgbm_model):
        model, X = tiny_lgbm_model
        sv = compute_shap_values(model, X)
        assert sv.dtype == np.float64

    def test_sample_subset(self, tiny_lgbm_model):
        model, X = tiny_lgbm_model
        sv = compute_shap_values(model, X.iloc[:50])
        assert sv.shape[0] == 50
        assert sv.shape[1] == X.shape[1]

    def test_non_nan(self, tiny_lgbm_model):
        model, X = tiny_lgbm_model
        sv = compute_shap_values(model, X)
        assert not np.isnan(sv).any()


# ---------------------------------------------------------------------------
# generate_shap_summary
# ---------------------------------------------------------------------------


class TestGenerateShapSummary:
    def _make_shap(self, n=100, p=15):
        rng = np.random.default_rng(0)
        return rng.normal(0, 1, (n, p)), [f"f{i}" for i in range(p)]

    def test_top_features_length_default(self):
        sv, names = self._make_shap(p=50)
        result = generate_shap_summary(sv, names)
        assert len(result["top_features"]) == min(TOP_N_FEATURES, 50)

    def test_top_features_length_custom(self):
        sv, names = self._make_shap(p=10)
        result = generate_shap_summary(sv, names, top_n=5)
        assert len(result["top_features"]) == 5

    def test_top_n_capped_by_n_features(self):
        sv, names = self._make_shap(p=8)
        result = generate_shap_summary(sv, names, top_n=30)
        assert len(result["top_features"]) == 8  # capped at 8

    def test_all_mean_abs_non_negative(self):
        sv, names = self._make_shap()
        result = generate_shap_summary(sv, names)
        for item in result["top_features"]:
            assert item["mean_abs_shap"] >= 0.0

    def test_features_sorted_descending(self):
        sv, names = self._make_shap()
        result = generate_shap_summary(sv, names)
        vals = [item["mean_abs_shap"] for item in result["top_features"]]
        assert vals == sorted(vals, reverse=True)

    def test_rank_starts_at_1(self):
        sv, names = self._make_shap()
        result = generate_shap_summary(sv, names)
        ranks = [item["rank"] for item in result["top_features"]]
        assert ranks[0] == 1
        assert ranks == list(range(1, len(ranks) + 1))

    def test_all_features_in_summary(self):
        sv, names = self._make_shap(p=10)
        result = generate_shap_summary(sv, names)
        assert set(result["all_features"].keys()) == set(names)

    def test_n_samples_correct(self):
        sv, names = self._make_shap(n=77)
        result = generate_shap_summary(sv, names)
        assert result["n_samples"] == 77

    def test_n_features_correct(self):
        sv, names = self._make_shap(n=50, p=12)
        result = generate_shap_summary(sv, names)
        assert result["n_features"] == 12

    def test_mismatched_feature_names_raises(self):
        sv, _ = self._make_shap(n=20, p=5)
        with pytest.raises(ValueError, match="feature_names length"):
            generate_shap_summary(sv, ["a", "b"])  # only 2 names for 5 features


# ---------------------------------------------------------------------------
# shap_by_segment
# ---------------------------------------------------------------------------


class TestShapBySegment:
    def _make_shap(self, n=120, p=5):
        rng = np.random.default_rng(1)
        sv = rng.normal(0, 1, (n, p))
        names = [f"feat_{i}" for i in range(p)]
        return sv, names

    def test_returns_dict_keyed_by_segment(self):
        sv, names = self._make_shap()
        segs = pd.Series(["FOODS"] * 40 + ["HOUSEHOLD"] * 40 + ["HOBBIES"] * 40)
        result = shap_by_segment(sv, names, segs)
        assert set(result.keys()) == {"FOODS", "HOUSEHOLD", "HOBBIES"}

    def test_each_segment_has_features(self):
        sv, names = self._make_shap()
        segs = pd.Series(["A"] * 60 + ["B"] * 60)
        result = shap_by_segment(sv, names, segs)
        for seg_list in result.values():
            assert len(seg_list) > 0
            for item in seg_list:
                assert "feature" in item
                assert "mean_abs_shap" in item
                assert item["mean_abs_shap"] >= 0.0

    def test_top_n_respected(self):
        sv, names = self._make_shap(p=10)
        segs = pd.Series(["X"] * 120)
        result = shap_by_segment(sv, names, segs, top_n=3)
        assert len(result["X"]) == 3

    def test_single_segment(self):
        sv, names = self._make_shap()
        segs = pd.Series(["ALL"] * 120)
        result = shap_by_segment(sv, names, segs)
        assert "ALL" in result


# ---------------------------------------------------------------------------
# export_shap_for_frontend
# ---------------------------------------------------------------------------


class TestExportShapForFrontend:
    def _make_summary(self, p=15):
        rng = np.random.default_rng(2)
        sv = rng.normal(0, 1, (100, p))
        names = [f"f{i}" for i in range(p)]
        return generate_shap_summary(sv, names)

    def test_shap_summary_json_written(self, tmp_path):
        summary = self._make_summary()
        paths = export_shap_for_frontend(summary, None, None, tmp_path)
        assert "shap_summary" in paths
        assert (tmp_path / "shap_summary.json").exists()

    def test_shap_summary_valid_json(self, tmp_path):
        summary = self._make_summary()
        export_shap_for_frontend(summary, None, None, tmp_path)
        with open(tmp_path / "shap_summary.json") as f:
            data = json.load(f)
        assert "top_features" in data
        assert "n_samples" in data

    def test_shap_by_segment_written_when_provided(self, tmp_path):
        summary = self._make_summary()
        by_cat = {"FOODS": [{"feature": "f0", "mean_abs_shap": 0.5}]}
        paths = export_shap_for_frontend(summary, by_cat, None, tmp_path)
        assert "shap_by_segment" in paths
        assert (tmp_path / "shap_by_segment.json").exists()

    def test_by_segment_not_written_when_both_none(self, tmp_path):
        summary = self._make_summary()
        paths = export_shap_for_frontend(summary, None, None, tmp_path)
        assert "shap_by_segment" not in paths

    def test_shap_summary_size_under_50kb(self, tmp_path):
        """Top-30 features JSON should be < 50 KB (spec §13.4)."""
        summary = self._make_summary(p=100)
        export_shap_for_frontend(summary, None, None, tmp_path)
        size_kb = (tmp_path / "shap_summary.json").stat().st_size / 1024
        assert size_kb < 50, f"shap_summary.json is {size_kb:.1f} KB, exceeds 50 KB budget"

    def test_all_mean_abs_non_negative_in_json(self, tmp_path):
        summary = self._make_summary()
        export_shap_for_frontend(summary, None, None, tmp_path)
        with open(tmp_path / "shap_summary.json") as f:
            data = json.load(f)
        for feat in data["top_features"]:
            assert feat["mean_abs_shap"] >= 0.0
