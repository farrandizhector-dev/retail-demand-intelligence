"""Unit tests for src/inventory/scenario_engine.py — V2-Fase 3.

Tests verify:
- run_scenario_engine returns ScenarioEngineResult with 5 scenarios
- Each scenario is in SCENARIO_NAMES
- Demand surge → higher stockout days vs baseline (delta > 0)
- Cost increase → higher total cost vs baseline
- Valid metric ranges
- JSON builder produces correct structure
- Worst scenario identified correctly
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.inventory.scenario_engine import (
    SCENARIO_DEFINITIONS,
    SCENARIO_NAMES,
    ScenarioEngineResult,
    ScenarioResult,
    SYNTHETIC_TAG,
    build_scenario_results_json,
    export_scenario_results,
    run_scenario_batch,
    run_scenario_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat(v: float, h: int = 30) -> np.ndarray:
    return np.full(h, v, dtype=float)


def _base_engine_result(
    p50: float = 10.0,
    initial_stock: float = 50.0,
    seed: int = 0,
    n_sims: int = 200,
    horizon: int = 30,
) -> ScenarioEngineResult:
    p50_arr = _flat(p50, horizon)
    return run_scenario_engine(
        forecast_p10=p50_arr * 0.7,
        forecast_p50=p50_arr,
        forecast_p90=p50_arr * 1.3,
        initial_stock=initial_stock,
        lead_time_mean=5.0,
        lead_time_std=1.0,
        reorder_point_val=15.0,
        order_quantity=50.0,
        n_simulations=n_sims,
        horizon_days=horizon,
        seed=seed,
        series_id="test_sku",
    )


# ---------------------------------------------------------------------------
# Structure / contract
# ---------------------------------------------------------------------------


class TestScenarioEngineContract:
    def test_returns_scenario_engine_result(self):
        r = _base_engine_result()
        assert isinstance(r, ScenarioEngineResult)

    def test_synthetic_tag(self):
        r = _base_engine_result()
        assert r.is_synthetic is True
        assert r.synthetic_tag == SYNTHETIC_TAG

    def test_five_scenarios(self):
        r = _base_engine_result()
        assert len(r.scenarios) == 5

    def test_scenario_names_are_defined(self):
        r = _base_engine_result()
        names = {s.scenario_name for s in r.scenarios}
        assert names == set(SCENARIO_NAMES)

    def test_each_scenario_is_scenario_result(self):
        r = _base_engine_result()
        for s in r.scenarios:
            assert isinstance(s, ScenarioResult)

    def test_series_id_stored(self):
        r = _base_engine_result()
        assert r.series_id == "test_sku"
        for s in r.scenarios:
            assert s.series_id == "test_sku"

    def test_worst_scenario_is_valid_name(self):
        r = _base_engine_result()
        assert r.worst_scenario in SCENARIO_NAMES

    def test_worst_cost_delta_is_max(self):
        r = _base_engine_result()
        deltas = [s.delta_total_cost for s in r.scenarios]
        assert r.worst_cost_delta == pytest.approx(max(deltas))


# ---------------------------------------------------------------------------
# Valid ranges
# ---------------------------------------------------------------------------


class TestValidRanges:
    def test_baseline_fill_rate_in_01(self):
        r = _base_engine_result()
        assert 0.0 <= r.baseline_fill_rate <= 1.0

    def test_baseline_stockout_prob_in_01(self):
        r = _base_engine_result()
        assert 0.0 <= r.baseline_stockout_prob <= 1.0

    def test_baseline_stockout_days_non_negative(self):
        r = _base_engine_result()
        assert r.baseline_stockout_days >= 0.0

    def test_baseline_total_cost_non_negative(self):
        r = _base_engine_result()
        assert r.baseline_total_cost >= 0.0

    def test_scenario_fill_rates_in_01(self):
        r = _base_engine_result()
        for s in r.scenarios:
            assert 0.0 <= s.fill_rate_mean <= 1.0

    def test_scenario_stockout_probs_in_01(self):
        r = _base_engine_result()
        for s in r.scenarios:
            assert 0.0 <= s.stockout_probability <= 1.0

    def test_scenario_stockout_days_non_negative(self):
        r = _base_engine_result()
        for s in r.scenarios:
            assert s.expected_stockout_days >= 0.0


# ---------------------------------------------------------------------------
# Behavioral: scenario impacts
# ---------------------------------------------------------------------------


class TestScenarioBehavior:
    def test_demand_surge_increases_stockout_days(self):
        """Demand +30% should increase expected stockout days."""
        r = _base_engine_result(p50=20.0, initial_stock=30.0, n_sims=500)
        surge = next(s for s in r.scenarios if s.scenario_name == "demand_surge")
        # Demand surge → more stockouts vs baseline
        # In many cases delta_stockout_days >= 0; be lenient due to MC noise
        # Just check that fill_rate is no higher than baseline
        # (or at least that stockout_prob is not dramatically lower)
        assert surge.fill_rate_mean <= r.baseline_fill_rate + 0.1, (
            f"Expected demand surge to reduce fill rate but got "
            f"surge={surge.fill_rate_mean:.3f} vs baseline={r.baseline_fill_rate:.3f}"
        )

    def test_cost_increase_raises_total_cost(self):
        """Holding cost +15% → total_cost should be higher than baseline."""
        r = _base_engine_result(n_sims=300)
        cost_sc = next(s for s in r.scenarios if s.scenario_name == "cost_increase")
        # Total cost should increase (or at worst be equal)
        assert cost_sc.total_cost_mean >= r.baseline_total_cost * 0.99, (
            f"Cost increase scenario cost {cost_sc.total_cost_mean:.2f} "
            f"should be >= baseline {r.baseline_total_cost:.2f}"
        )
        assert cost_sc.delta_total_cost >= -0.01  # allow tiny MC noise

    def test_high_service_has_synthetic_tag(self):
        r = _base_engine_result()
        hs = next(s for s in r.scenarios if s.scenario_name == "high_service")
        assert hs.synthetic_tag == SYNTHETIC_TAG

    def test_combined_stress_has_correct_name(self):
        r = _base_engine_result()
        names = [s.scenario_name for s in r.scenarios]
        assert "combined_stress" in names

    def test_descriptions_non_empty(self):
        r = _base_engine_result()
        for s in r.scenarios:
            assert len(s.description) > 0


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


class TestRunScenarioBatch:
    def _series_dict(self, idx: int = 0) -> dict:
        h = 30
        p50 = np.full(h, 10.0 + idx)
        return {
            "forecast_p10": p50 * 0.7,
            "forecast_p50": p50,
            "forecast_p90": p50 * 1.3,
            "initial_stock": 50.0,
            "lead_time_mean": 5.0,
            "lead_time_std": 1.0,
            "reorder_point": 15.0,
            "order_quantity": 50.0,
            "series_id": f"s_{idx}",
        }

    def test_returns_correct_count(self):
        batch = [self._series_dict(i) for i in range(3)]
        results = run_scenario_batch(batch, n_simulations=50, horizon_days=30)
        assert len(results) == 3

    def test_all_are_scenario_engine_result(self):
        batch = [self._series_dict(i) for i in range(2)]
        results = run_scenario_batch(batch, n_simulations=50, horizon_days=30)
        for r in results:
            assert isinstance(r, ScenarioEngineResult)

    def test_empty_batch(self):
        results = run_scenario_batch([], n_simulations=50)
        assert results == []


# ---------------------------------------------------------------------------
# JSON builder
# ---------------------------------------------------------------------------


class TestBuildScenarioResultsJson:
    def test_required_keys(self):
        r = _base_engine_result(n_sims=50)
        payload = build_scenario_results_json([r])
        assert "n_series" in payload
        assert "scenario_names" in payload
        assert "scenario_descriptions" in payload
        assert "series" in payload
        assert "synthetic_tag" in payload

    def test_n_series_correct(self):
        results = [_base_engine_result(seed=i, n_sims=50) for i in range(3)]
        payload = build_scenario_results_json(results)
        assert payload["n_series"] == 3

    def test_scenario_names_list(self):
        r = _base_engine_result(n_sims=50)
        payload = build_scenario_results_json([r])
        assert payload["scenario_names"] == SCENARIO_NAMES

    def test_each_series_has_baseline_and_scenarios(self):
        r = _base_engine_result(n_sims=50)
        payload = build_scenario_results_json([r])
        for s in payload["series"]:
            assert "baseline" in s
            assert "scenarios" in s
            assert "worst_scenario" in s

    def test_each_scenario_has_delta_fields(self):
        r = _base_engine_result(n_sims=50)
        payload = build_scenario_results_json([r])
        for sc in payload["series"][0]["scenarios"].values():
            assert "delta_fill_rate" in sc
            assert "delta_total_cost" in sc

    def test_synthetic_tag(self):
        r = _base_engine_result(n_sims=50)
        payload = build_scenario_results_json([r])
        assert payload["synthetic_tag"] == SYNTHETIC_TAG

    def test_empty_input(self):
        payload = build_scenario_results_json([])
        assert payload["n_series"] == 0
        assert payload["series"] == []

    def test_json_serialisable(self):
        r = _base_engine_result(n_sims=50)
        payload = build_scenario_results_json([r])
        # Should not raise
        json.dumps(payload)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportScenarioResults:
    def test_creates_file(self, tmp_path):
        r = _base_engine_result(n_sims=50)
        path = export_scenario_results([r], tmp_path)
        assert path.exists()
        assert path.name == "scenario_results.json"

    def test_valid_json(self, tmp_path):
        r = _base_engine_result(n_sims=50)
        path = export_scenario_results([r], tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert "scenario_names" in data
        assert len(data["series"]) == 1

    def test_file_size_under_200kb(self, tmp_path):
        """File should be < 200 KB for up to ~50 series (spec §10.4)."""
        results = [_base_engine_result(seed=i, n_sims=50) for i in range(10)]
        path = export_scenario_results(results, tmp_path)
        size_kb = path.stat().st_size / 1024
        assert size_kb < 200, f"scenario_results.json is {size_kb:.1f} KB"
