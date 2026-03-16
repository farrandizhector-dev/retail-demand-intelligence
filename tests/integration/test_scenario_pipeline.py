"""Integration tests for Monte Carlo + Policy + Scenario pipeline — V2-Fase 3.

Smoke tests that verify the end-to-end flow:
- All 5 scenarios produce valid JSON output
- Policy comparison + scenario engine can be chained
- scenario_results.json < 200 KB for batch of series
- policy_comparison.parquet is written correctly
- serving_exporter integration with policy_comparison_summary + scenario_results
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.inventory.policy_comparator import (
    POLICY_NAMES,
    build_policy_comparison_summary,
    compare_policies,
    run_policy_comparison_batch,
)
from src.inventory.scenario_engine import (
    SCENARIO_NAMES,
    build_scenario_results_json,
    export_scenario_results,
    run_scenario_engine,
    run_scenario_batch,
)
from src.inventory.simulator import simulate_inventory_mc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sku(p50: float = 10.0, horizon: int = 30) -> dict:
    p50_arr = np.full(horizon, p50)
    return {
        "forecast_p10": p50_arr * 0.7,
        "forecast_p50": p50_arr,
        "forecast_p90": p50_arr * 1.3,
        "initial_stock": 50.0,
        "lead_time_mean": 5.0,
        "lead_time_std": 1.0,
        "reorder_point": 15.0,
        "order_quantity": 50.0,
    }


# ---------------------------------------------------------------------------
# All 5 scenarios — basic smoke test
# ---------------------------------------------------------------------------


class TestAllFiveScenariosValid:
    def test_all_scenarios_present(self):
        sku = _make_sku()
        result = run_scenario_engine(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=sku["initial_stock"],
            lead_time_mean=sku["lead_time_mean"],
            lead_time_std=sku["lead_time_std"],
            reorder_point_val=sku["reorder_point"],
            order_quantity=sku["order_quantity"],
            n_simulations=200, horizon_days=30,
        )
        names = {s.scenario_name for s in result.scenarios}
        assert names == set(SCENARIO_NAMES)

    def test_all_scenarios_have_delta_fields(self):
        sku = _make_sku()
        result = run_scenario_engine(
            **{k: sku[k] for k in ["forecast_p10", "forecast_p50", "forecast_p90",
                                    "initial_stock", "lead_time_mean", "lead_time_std"]},
            reorder_point_val=sku["reorder_point"],
            order_quantity=sku["order_quantity"],
            n_simulations=200, horizon_days=30,
        )
        for s in result.scenarios:
            assert hasattr(s, "delta_fill_rate")
            assert hasattr(s, "delta_total_cost")
            assert hasattr(s, "delta_stockout_days")

    def test_scenario_results_json_all_scenarios(self, tmp_path):
        sku = _make_sku()
        result = run_scenario_engine(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=sku["initial_stock"],
            lead_time_mean=sku["lead_time_mean"],
            lead_time_std=sku["lead_time_std"],
            reorder_point_val=sku["reorder_point"],
            order_quantity=sku["order_quantity"],
            n_simulations=200, horizon_days=30,
        )
        path = export_scenario_results([result], tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert len(data["series"][0]["scenarios"]) == 5
        assert set(data["series"][0]["scenarios"].keys()) == set(SCENARIO_NAMES)


# ---------------------------------------------------------------------------
# scenario_results.json < 200 KB for batch
# ---------------------------------------------------------------------------


class TestScenarioJsonSizeLimit:
    def test_10_series_under_200kb(self, tmp_path):
        skus = [dict(_make_sku(p50=5.0 + i), series_id=f"s_{i}") for i in range(10)]
        results = run_scenario_batch(skus, n_simulations=100, horizon_days=30)
        path = export_scenario_results(results, tmp_path)
        size_kb = path.stat().st_size / 1024
        assert size_kb < 200, f"scenario_results.json is {size_kb:.1f} KB for 10 series"

    def test_50_series_under_200kb(self, tmp_path):
        skus = [dict(_make_sku(p50=5.0 + i * 0.1), series_id=f"s_{i}") for i in range(50)]
        results = run_scenario_batch(skus, n_simulations=50, horizon_days=30)
        path = export_scenario_results(results, tmp_path)
        size_kb = path.stat().st_size / 1024
        assert size_kb < 200, f"scenario_results.json is {size_kb:.1f} KB for 50 series"


# ---------------------------------------------------------------------------
# Policy comparison pipeline
# ---------------------------------------------------------------------------


class TestPolicyComparisonPipeline:
    def test_all_4_policies_evaluated(self):
        sku = _make_sku()
        result = compare_policies(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=sku["initial_stock"],
            lead_time_mean=sku["lead_time_mean"],
            lead_time_std=sku["lead_time_std"],
            n_simulations=200, horizon_days=30,
        )
        assert len(result.policy_names) == 4
        assert set(result.policy_names) == set(POLICY_NAMES)

    def test_policy_comparison_parquet_written(self, tmp_path):
        skus = [dict(_make_sku(p50=5.0 + i), series_id=f"sku_{i}") for i in range(3)]
        run_policy_comparison_batch(
            skus, n_simulations=100, horizon_days=30, output_dir=tmp_path
        )
        assert (tmp_path / "policy_comparison.parquet").exists()

    def test_summary_json_valid(self):
        sku = _make_sku()
        result = compare_policies(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=sku["initial_stock"],
            lead_time_mean=sku["lead_time_mean"],
            lead_time_std=sku["lead_time_std"],
            n_simulations=200, horizon_days=30,
            series_id="sku_001",
        )
        summary = build_policy_comparison_summary([result])
        # Verify JSON serialisable
        text = json.dumps(summary)
        data = json.loads(text)
        assert data["n_series"] == 1
        assert len(data["series"][0]["policies"]) == 4


# ---------------------------------------------------------------------------
# serving_exporter integration
# ---------------------------------------------------------------------------


class TestServingExporterIntegration:
    def test_policy_comparison_summary_in_serving_assets(self, tmp_path):
        from src.export.serving_exporter import export_serving_assets

        sku = _make_sku()
        result = compare_policies(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=50.0,
            lead_time_mean=5.0,
            lead_time_std=1.0,
            n_simulations=100, horizon_days=30,
        )
        summary = build_policy_comparison_summary([result])

        written = export_serving_assets(
            output_dir=tmp_path / "serving",
            is_synthetic=True,
            policy_comparison_summary=summary,
        )
        assert "policy_comparison_summary" in written
        assert (tmp_path / "serving" / "policy_comparison_summary.json").exists()

    def test_scenario_results_in_serving_assets(self, tmp_path):
        from src.export.serving_exporter import export_serving_assets

        sku = _make_sku()
        result = run_scenario_engine(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=50.0,
            lead_time_mean=5.0,
            lead_time_std=1.0,
            reorder_point_val=15.0,
            order_quantity=50.0,
            n_simulations=100, horizon_days=30,
        )
        scenario_payload = build_scenario_results_json([result])

        written = export_serving_assets(
            output_dir=tmp_path / "serving2",
            is_synthetic=True,
            scenario_results=scenario_payload,
        )
        assert "scenario_results" in written
        assert (tmp_path / "serving2" / "scenario_results.json").exists()

    def test_manifest_includes_policy_and_scenario(self, tmp_path):
        from src.export.serving_exporter import export_serving_assets

        sku = _make_sku()
        result = compare_policies(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=50.0,
            lead_time_mean=5.0,
            lead_time_std=1.0,
            n_simulations=100, horizon_days=30,
        )
        policy_summary = build_policy_comparison_summary([result])

        sc_result = run_scenario_engine(
            forecast_p10=sku["forecast_p10"],
            forecast_p50=sku["forecast_p50"],
            forecast_p90=sku["forecast_p90"],
            initial_stock=50.0,
            lead_time_mean=5.0,
            lead_time_std=1.0,
            reorder_point_val=15.0,
            order_quantity=50.0,
            n_simulations=100, horizon_days=30,
        )
        scenario_payload = build_scenario_results_json([sc_result])

        written = export_serving_assets(
            output_dir=tmp_path / "serving3",
            is_synthetic=True,
            policy_comparison_summary=policy_summary,
            scenario_results=scenario_payload,
        )

        manifest_path = written["asset_manifest"]
        with open(manifest_path) as f:
            manifest = json.load(f)
        asset_names = [a["name"] for a in manifest["assets"]]
        assert "policy_comparison_summary.json" in asset_names
        assert "scenario_results.json" in asset_names
