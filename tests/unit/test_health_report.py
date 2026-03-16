"""Unit tests for src/monitoring/health_report_generator.py — 20+ tests."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.monitoring.alert_engine import Alert, Severity
from src.monitoring.health_report_generator import (
    HealthReport,
    build_recommendations,
    determine_overall_status,
    generate_health_report,
    save_health_report,
)


# ===========================================================================
# Fixtures / helpers
# ===========================================================================

def _make_alert(name: str, severity: Severity) -> Alert:
    return Alert(
        name=name,
        trigger="test_trigger",
        severity=severity,
        message=f"Test alert: {name}",
        timestamp="2024-01-01T00:00:00+00:00",
        details={},
    )


def _minimal_report(**overrides) -> HealthReport:
    defaults = dict(
        pipeline_status={"steps": {}, "all_passing": True},
        quality_scores={},
        model_metrics={},
        drift_results={},
        alerts=[],
        serving_manifest={},
    )
    defaults.update(overrides)
    return generate_health_report(**defaults)


# ===========================================================================
# TestDetermineOverallStatus
# ===========================================================================

class TestDetermineOverallStatus:
    def test_healthy_no_alerts(self):
        assert determine_overall_status([]) == "HEALTHY"

    def test_degraded_high_alert(self):
        alerts = [_make_alert("FORECAST_DEGRADATION", Severity.HIGH)]
        assert determine_overall_status(alerts) == "DEGRADED"

    def test_critical_on_critical_alert(self):
        alerts = [_make_alert("DATA_QUALITY_FAILURE", Severity.CRITICAL)]
        assert determine_overall_status(alerts) == "CRITICAL"

    def test_critical_beats_degraded(self):
        alerts = [
            _make_alert("FORECAST_DEGRADATION", Severity.HIGH),
            _make_alert("DATA_QUALITY_FAILURE", Severity.CRITICAL),
        ]
        assert determine_overall_status(alerts) == "CRITICAL"

    def test_low_alert_is_healthy(self):
        alerts = [_make_alert("SOME_MINOR", Severity.LOW)]
        assert determine_overall_status(alerts) == "HEALTHY"

    def test_medium_alert_is_healthy(self):
        alerts = [_make_alert("DRIFT_DETECTED", Severity.MEDIUM)]
        assert determine_overall_status(alerts) == "HEALTHY"


# ===========================================================================
# TestBuildRecommendations
# ===========================================================================

class TestBuildRecommendations:
    def test_no_alerts_default_recommendation(self):
        recs = build_recommendations([], {}, {})
        assert len(recs) == 1
        assert "operating normally" in recs[0].lower()

    def test_degradation_recommendation(self):
        alert = Alert(
            name="FORECAST_DEGRADATION",
            trigger="mae",
            severity=Severity.HIGH,
            message="m",
            timestamp="t",
            details={"days_above": 21},
        )
        recs = build_recommendations([alert], {}, {})
        assert any("retrain" in r.lower() for r in recs)

    def test_drift_recommendation(self):
        alert = Alert(
            name="DRIFT_DETECTED",
            trigger="psi",
            severity=Severity.MEDIUM,
            message="m",
            timestamp="t",
            details={"drifted_features": [{"feature": "lag_7", "psi": 0.30, "status": "retrain"}]},
        )
        recs = build_recommendations([alert], {}, {})
        assert any("drift" in r.lower() for r in recs)

    def test_data_quality_recommendation(self):
        alert = Alert(
            name="DATA_QUALITY_FAILURE",
            trigger="step",
            severity=Severity.CRITICAL,
            message="m",
            timestamp="t",
            details={"failed_steps": ["ingest"]},
        )
        recs = build_recommendations([alert], {}, {})
        assert any("pipeline" in r.lower() or "investigate" in r.lower() for r in recs)

    def test_inventory_recommendation(self):
        alert = Alert(
            name="INVENTORY_ANOMALY",
            trigger="fill",
            severity=Severity.HIGH,
            message="m",
            timestamp="t",
            details={"affected_stores": [{"store": "CA_1", "fill_rate": 0.80}]},
        )
        recs = build_recommendations([alert], {}, {})
        assert any("inventory" in r.lower() for r in recs)

    def test_reconciliation_recommendation(self):
        alert = Alert(
            name="RECONCILIATION_INCOHERENCE",
            trigger="check",
            severity=Severity.CRITICAL,
            message="m",
            timestamp="t",
            details={},
        )
        recs = build_recommendations([alert], {}, {})
        assert any("reconciliation" in r.lower() or "matrix" in r.lower() for r in recs)

    def test_serving_budget_recommendation(self):
        alert = Alert(
            name="SERVING_BUDGET_EXCEEDED",
            trigger="budget",
            severity=Severity.MEDIUM,
            message="m",
            timestamp="t",
            details={"total_bytes": 6 * 1024 * 1024},
        )
        recs = build_recommendations([alert], {}, {})
        assert any("serving" in r.lower() or "bundle" in r.lower() or "mb" in r.lower() for r in recs)


# ===========================================================================
# TestGenerateHealthReport
# ===========================================================================

class TestGenerateHealthReport:
    def test_has_all_required_fields(self):
        report = _minimal_report()
        assert hasattr(report, "generated_at")
        assert hasattr(report, "overall_status")
        assert hasattr(report, "pipeline_status")
        assert hasattr(report, "data_quality")
        assert hasattr(report, "model_health")
        assert hasattr(report, "drift_indicators")
        assert hasattr(report, "active_alerts")
        assert hasattr(report, "serving_health")
        assert hasattr(report, "recommendations")

    def test_generated_at_is_iso_string(self):
        report = _minimal_report()
        # Should be parseable as datetime
        dt = datetime.fromisoformat(report.generated_at)
        assert dt is not None

    def test_overall_status_values(self):
        report = _minimal_report()
        assert report.overall_status in ("HEALTHY", "DEGRADED", "CRITICAL")

    def test_active_alerts_is_list(self):
        report = _minimal_report()
        assert isinstance(report.active_alerts, list)

    def test_recommendations_is_list(self):
        report = _minimal_report()
        assert isinstance(report.recommendations, list)

    def test_serving_health_has_budget_info(self):
        report = _minimal_report()
        assert "total_bytes" in report.serving_health
        assert "pct_used" in report.serving_health

    def test_critical_status_with_critical_alert(self):
        report = generate_health_report(
            pipeline_status={},
            quality_scores={},
            model_metrics={},
            drift_results={},
            alerts=[_make_alert("DATA_QUALITY_FAILURE", Severity.CRITICAL)],
            serving_manifest={},
        )
        assert report.overall_status == "CRITICAL"

    def test_pipeline_all_passing_true(self):
        report = generate_health_report(
            pipeline_status={"ingest": "pass", "transform": "pass"},
            quality_scores={},
            model_metrics={},
            drift_results={},
            alerts=[],
            serving_manifest={},
        )
        assert report.pipeline_status["all_passing"] is True

    def test_pipeline_all_passing_false_on_fail(self):
        report = generate_health_report(
            pipeline_status={"ingest": "fail"},
            quality_scores={},
            model_metrics={},
            drift_results={},
            alerts=[],
            serving_manifest={},
        )
        assert report.pipeline_status["all_passing"] is False

    def test_data_quality_mean_score(self):
        report = generate_health_report(
            pipeline_status={},
            quality_scores={"bronze": 0.90, "silver": 0.80},
            model_metrics={},
            drift_results={},
            alerts=[],
            serving_manifest={},
        )
        assert report.data_quality["mean_score"] == pytest.approx(0.85)

    def test_serving_health_pct_used(self):
        # 2.5MB out of 5MB = 50%
        report = generate_health_report(
            pipeline_status={},
            quality_scores={},
            model_metrics={},
            drift_results={},
            alerts=[],
            serving_manifest={"a.json": {"size_bytes": 2_621_440}},
        )
        assert 0.0 < report.serving_health["pct_used"] <= 1.0


# ===========================================================================
# TestSaveHealthReport
# ===========================================================================

class TestSaveHealthReport:
    def test_saves_to_path(self, tmp_path):
        report = _minimal_report()
        out = tmp_path / "health_report.json"
        save_health_report(report, out)
        assert out.exists()

    def test_valid_json(self, tmp_path):
        report = _minimal_report()
        out = tmp_path / "report.json"
        save_health_report(report, out)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_returns_path(self, tmp_path):
        report = _minimal_report()
        out = tmp_path / "r.json"
        returned = save_health_report(report, out)
        assert returned == out

    def test_creates_parent_dirs(self, tmp_path):
        report = _minimal_report()
        out = tmp_path / "nested" / "deep" / "report.json"
        save_health_report(report, out)
        assert out.exists()

    def test_json_has_overall_status(self, tmp_path):
        report = _minimal_report()
        out = tmp_path / "r.json"
        save_health_report(report, out)
        data = json.loads(out.read_text())
        assert "overall_status" in data
