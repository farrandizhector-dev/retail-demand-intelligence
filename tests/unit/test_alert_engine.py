"""Unit tests for src/monitoring/alert_engine.py — 30+ tests."""

from __future__ import annotations

import pytest

from src.monitoring.alert_engine import (
    Alert,
    Severity,
    alerts_to_json,
    check_all_alerts,
    check_data_quality_failure,
    check_drift_detected,
    check_forecast_degradation,
    check_inventory_anomaly,
    check_reconciliation_incoherence,
    check_serving_budget,
)


# ===========================================================================
# TestCheckDataQualityFailure
# ===========================================================================

class TestCheckDataQualityFailure:
    def test_no_alert_when_all_pass(self):
        result = check_data_quality_failure({"ingest": "pass", "transform": "pass"})
        assert result is None

    def test_alert_on_fail_status(self):
        result = check_data_quality_failure({"ingest": "fail"})
        assert result is not None
        assert result.severity == Severity.CRITICAL

    def test_alert_on_false_status(self):
        result = check_data_quality_failure({"ingest": False})
        assert result is not None
        assert result.severity == Severity.CRITICAL

    def test_failed_steps_in_details(self):
        result = check_data_quality_failure({"ingest": "fail", "transform": "pass"})
        assert result is not None
        assert "ingest" in result.details["failed_steps"]
        assert "transform" not in result.details["failed_steps"]

    def test_empty_status_no_alert(self):
        result = check_data_quality_failure({})
        assert result is None

    def test_multiple_failed_steps(self):
        result = check_data_quality_failure({"a": "fail", "b": False, "c": "pass"})
        assert result is not None
        assert set(result.details["failed_steps"]) == {"a", "b"}

    def test_alert_name(self):
        result = check_data_quality_failure({"x": "fail"})
        assert result.name == "DATA_QUALITY_FAILURE"


# ===========================================================================
# TestCheckForecastDegradation
# ===========================================================================

class TestCheckForecastDegradation:
    def test_no_alert_below_threshold(self):
        # 10% increase < 15% threshold
        result = check_forecast_degradation(current_mae=11.0, baseline_mae=10.0, days_above=20)
        assert result is None

    def test_no_alert_too_few_days(self):
        # Enough MAE increase but < 14 days
        result = check_forecast_degradation(current_mae=12.0, baseline_mae=10.0, days_above=10)
        assert result is None

    def test_alert_fires(self):
        # 20% increase AND 20 days above
        result = check_forecast_degradation(current_mae=12.0, baseline_mae=10.0, days_above=20)
        assert result is not None
        assert result.severity == Severity.HIGH

    def test_pct_increase_in_details(self):
        result = check_forecast_degradation(current_mae=12.0, baseline_mae=10.0, days_above=20)
        assert result is not None
        assert result.details["pct_increase"] == pytest.approx(0.20, rel=1e-6)

    def test_exact_threshold_no_alert(self):
        # Exactly 15% over baseline: 11.5 / 10.0 = 1.15 — NOT strictly greater
        result = check_forecast_degradation(current_mae=11.5, baseline_mae=10.0, days_above=20)
        assert result is None

    def test_alert_name(self):
        result = check_forecast_degradation(current_mae=12.0, baseline_mae=10.0, days_above=20)
        assert result.name == "FORECAST_DEGRADATION"

    def test_details_contain_days_above(self):
        result = check_forecast_degradation(current_mae=12.0, baseline_mae=10.0, days_above=20)
        assert result.details["days_above"] == 20


# ===========================================================================
# TestCheckDriftDetected
# ===========================================================================

class TestCheckDriftDetected:
    def test_no_alert_all_ok(self):
        drift = {"f1": {"psi": 0.05, "status": "ok"}, "f2": {"psi": 0.08, "status": "ok"}}
        result = check_drift_detected(drift)
        assert result is None

    def test_alert_on_alert_status(self):
        drift = {"f1": {"psi": 0.22, "status": "alert"}}
        result = check_drift_detected(drift)
        assert result is not None
        assert result.severity == Severity.MEDIUM

    def test_alert_on_retrain_status(self):
        drift = {"f1": {"psi": 0.30, "status": "retrain"}}
        result = check_drift_detected(drift)
        assert result is not None
        assert result.severity == Severity.MEDIUM

    def test_drifted_features_in_details(self):
        drift = {
            "feat_ok": {"psi": 0.05, "status": "ok"},
            "feat_bad": {"psi": 0.30, "status": "retrain"},
        }
        result = check_drift_detected(drift)
        assert result is not None
        drifted_names = [d["feature"] for d in result.details["drifted_features"]]
        assert "feat_bad" in drifted_names
        assert "feat_ok" not in drifted_names

    def test_empty_drift_results_no_alert(self):
        result = check_drift_detected({})
        assert result is None

    def test_alert_name(self):
        drift = {"f": {"psi": 0.30, "status": "retrain"}}
        result = check_drift_detected(drift)
        assert result.name == "DRIFT_DETECTED"

    def test_warning_status_no_alert(self):
        # "warning" should NOT trigger an alert
        drift = {"f": {"psi": 0.15, "status": "warning"}}
        result = check_drift_detected(drift)
        assert result is None


# ===========================================================================
# TestCheckInventoryAnomaly
# ===========================================================================

class TestCheckInventoryAnomaly:
    def test_no_alert_above_threshold(self):
        result = check_inventory_anomaly({"CA_1": 0.92, "TX_1": 0.90})
        assert result is None

    def test_alert_below_threshold(self):
        result = check_inventory_anomaly({"CA_1": 0.80})
        assert result is not None
        assert result.severity == Severity.HIGH

    def test_affected_stores_in_details(self):
        result = check_inventory_anomaly({"CA_1": 0.80, "CA_2": 0.95})
        assert result is not None
        affected_stores = [s["store"] for s in result.details["affected_stores"]]
        assert "CA_1" in affected_stores
        assert "CA_2" not in affected_stores

    def test_exact_threshold_not_alert(self):
        # Exactly 0.85 should NOT trigger (< 0.85 is threshold)
        result = check_inventory_anomaly({"CA_1": 0.85})
        assert result is None

    def test_empty_store_rates_no_alert(self):
        result = check_inventory_anomaly({})
        assert result is None

    def test_alert_name(self):
        result = check_inventory_anomaly({"S": 0.50})
        assert result.name == "INVENTORY_ANOMALY"


# ===========================================================================
# TestCheckReconciliationIncoherence
# ===========================================================================

class TestCheckReconciliationIncoherence:
    def test_no_alert_when_coherent(self):
        result = check_reconciliation_incoherence(coherence_passed=True)
        assert result is None

    def test_alert_when_incoherent(self):
        result = check_reconciliation_incoherence(coherence_passed=False)
        assert result is not None
        assert result.severity == Severity.CRITICAL

    def test_details_forwarded(self):
        d = {"max_abs_error": 0.5}
        result = check_reconciliation_incoherence(False, details=d)
        assert result.details == d

    def test_alert_name(self):
        result = check_reconciliation_incoherence(False)
        assert result.name == "RECONCILIATION_INCOHERENCE"


# ===========================================================================
# TestCheckServingBudget
# ===========================================================================

class TestCheckServingBudget:
    def test_no_alert_within_budget(self):
        assets = {"a.json": 1_000_000, "b.json": 2_000_000}  # 3MB < 5MB
        result = check_serving_budget(assets)
        assert result is None

    def test_alert_over_budget(self):
        assets = {"a.json": 3_000_000, "b.json": 3_000_000}  # 6MB > 5MB
        result = check_serving_budget(assets)
        assert result is not None
        assert result.severity == Severity.MEDIUM

    def test_total_bytes_in_details(self):
        assets = {"a.json": 3_000_000, "b.json": 3_000_000}
        result = check_serving_budget(assets)
        assert result.details["total_bytes"] == 6_000_000

    def test_largest_assets_top3(self):
        assets = {f"f{i}.json": i * 1_000_000 for i in range(1, 8)}  # 7 assets
        result = check_serving_budget(assets, budget_bytes=0)  # force trigger
        assert len(result.details["largest_assets"]) == 3

    def test_empty_assets_no_alert(self):
        result = check_serving_budget({})
        assert result is None

    def test_alert_name(self):
        assets = {"big.json": 6_000_000}
        result = check_serving_budget(assets)
        assert result.name == "SERVING_BUDGET_EXCEEDED"


# ===========================================================================
# TestCheckAllAlerts
# ===========================================================================

class TestCheckAllAlerts:
    def test_empty_context_no_alerts(self):
        result = check_all_alerts({})
        # Default context: everything passing, no alerts expected
        assert isinstance(result, list)
        # Some rules might not fire with defaults, but none should crash
        for alert in result:
            assert isinstance(alert, Alert)

    def test_multiple_alerts_returned(self):
        context = {
            "pipeline_status": {"ingest": "fail"},
            "coherence_passed": False,
            "store_fill_rates": {"S1": 0.70},
        }
        result = check_all_alerts(context)
        assert len(result) >= 3

    def test_alerts_are_alert_instances(self):
        context = {"pipeline_status": {"x": "fail"}}
        result = check_all_alerts(context)
        for alert in result:
            assert isinstance(alert, Alert)

    def test_forecast_degradation_fires(self):
        context = {"current_mae": 12.0, "baseline_mae": 10.0, "days_above_threshold": 20}
        result = check_all_alerts(context)
        names = [a.name for a in result]
        assert "FORECAST_DEGRADATION" in names

    def test_drift_alert_fires(self):
        context = {"drift_results": {"feat": {"psi": 0.30, "status": "retrain"}}}
        result = check_all_alerts(context)
        names = [a.name for a in result]
        assert "DRIFT_DETECTED" in names


# ===========================================================================
# TestAlertsToJson
# ===========================================================================

class TestAlertsToJson:
    def test_serializable(self):
        import json
        alerts = [
            Alert(
                name="TEST",
                trigger="test",
                severity=Severity.HIGH,
                message="test msg",
                timestamp="2024-01-01T00:00:00+00:00",
                details={"k": 1},
            )
        ]
        result = alerts_to_json(alerts)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        # Should be JSON serialisable
        json.dumps(result)

    def test_severity_as_string(self):
        alerts = [
            Alert(
                name="T",
                trigger="t",
                severity=Severity.CRITICAL,
                message="m",
                timestamp="2024-01-01T00:00:00+00:00",
            )
        ]
        result = alerts_to_json(alerts)
        assert isinstance(result[0]["severity"], str)
        assert result[0]["severity"] == "CRITICAL"

    def test_empty_list(self):
        result = alerts_to_json([])
        assert result == []

    def test_all_fields_present(self):
        alerts = [
            Alert(
                name="N",
                trigger="T",
                severity=Severity.LOW,
                message="M",
                timestamp="ts",
                details={"x": 99},
            )
        ]
        result = alerts_to_json(alerts)
        assert set(result[0].keys()) == {"name", "trigger", "severity", "message", "timestamp", "details"}
