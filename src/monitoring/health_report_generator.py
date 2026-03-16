"""Health report generator — produces monitoring/health_report.json (spec §11.4)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.monitoring.alert_engine import Alert, Severity, alerts_to_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    generated_at: str
    overall_status: str  # "HEALTHY" | "DEGRADED" | "CRITICAL"
    pipeline_status: dict[str, Any]
    data_quality: dict[str, Any]
    model_health: dict[str, Any]
    drift_indicators: dict[str, Any]
    active_alerts: list[dict]
    serving_health: dict[str, Any]
    recommendations: list[str]


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def determine_overall_status(alerts: list[Alert]) -> str:
    """Derive overall system status from active alerts.

    Returns "CRITICAL" if any alert has CRITICAL severity,
    "DEGRADED" if any alert has HIGH severity,
    "HEALTHY" otherwise.
    """
    severities = {a.severity for a in alerts}
    if Severity.CRITICAL in severities:
        return "CRITICAL"
    if Severity.HIGH in severities:
        return "DEGRADED"
    return "HEALTHY"


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

def build_recommendations(
    alerts: list[Alert],
    drift_results: dict[str, Any],
    model_health: dict[str, Any],
) -> list[str]:
    """Build human-readable recommendation strings from active alerts."""
    if not alerts:
        return ["System is operating normally. Continue weekly monitoring schedule."]

    recs: list[str] = []

    for alert in alerts:
        if alert.name == "DATA_QUALITY_FAILURE":
            steps = alert.details.get("failed_steps", [])
            recs.append(
                f"Block pipeline and investigate failed steps: {steps}"
            )

        elif alert.name == "FORECAST_DEGRADATION":
            days = alert.details.get("days_above", 0)
            recs.append(
                f"Consider retraining: MAE has exceeded baseline for {days} days"
            )

        elif alert.name == "DRIFT_DETECTED":
            drifted = alert.details.get("drifted_features", [])
            feat_names = [d.get("feature", "") for d in drifted]
            recs.append(
                f"Investigate feature drift in: {feat_names}. Consider data pipeline review."
            )

        elif alert.name == "INVENTORY_ANOMALY":
            affected = alert.details.get("affected_stores", [])
            store_ids = [s.get("store", "") for s in affected]
            recs.append(
                f"Review inventory policy for affected stores: {store_ids}"
            )

        elif alert.name == "RECONCILIATION_INCOHERENCE":
            recs.append(
                "Re-run reconciliation pipeline and verify S matrix integrity"
            )

        elif alert.name == "SERVING_BUDGET_EXCEEDED":
            total_bytes = alert.details.get("total_bytes", 0)
            total_mb = total_bytes / (1024 * 1024)
            recs.append(
                f"Reduce JSON serving bundle: currently {total_mb:.2f}MB vs 5MB budget"
            )

    return recs if recs else ["System is operating normally. Continue weekly monitoring schedule."]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_health_report(
    pipeline_status: dict[str, Any],
    quality_scores: dict[str, float],
    model_metrics: dict[str, Any],
    drift_results: dict[str, Any],
    alerts: list[Alert],
    serving_manifest: dict[str, Any],
) -> HealthReport:
    """Generate a structured HealthReport.

    Parameters
    ----------
    pipeline_status:
        Mapping of step_name → "pass" | "fail".
    quality_scores:
        Mapping of layer_name → quality score (float 0-1).
    model_metrics:
        Dict with "current" and "baseline" sub-dicts of metric values.
    drift_results:
        Dict with keys "sales_drift", "feature_drift", "zero_inflation",
        "price_regime" (each containing their respective sub-dicts).
    alerts:
        List of Alert objects from alert_engine.check_all_alerts.
    serving_manifest:
        Dict of {filename: {size_bytes, ...}} or {filename: size_bytes}.

    Returns
    -------
    HealthReport dataclass.
    """
    # --- pipeline status block ---
    all_passing = all(
        v in ("pass", True) for v in pipeline_status.values()
    ) if pipeline_status else True
    pipeline_block = {"steps": pipeline_status, "all_passing": all_passing}

    # --- data quality block ---
    scores_list = list(quality_scores.values())
    mean_score = float(sum(scores_list) / len(scores_list)) if scores_list else 0.0
    quality_block = {"scores": quality_scores, "mean_score": mean_score}

    # --- model health block ---
    decay_detected = any(a.name == "FORECAST_DEGRADATION" for a in alerts)
    model_block = {"metrics": model_metrics, "decay_detected": decay_detected}

    # --- serving health block ---
    def _extract_size(v: Any) -> int:
        if isinstance(v, dict):
            return int(v.get("size_bytes", 0))
        return int(v)

    total_bytes = sum(_extract_size(v) for v in serving_manifest.values())
    budget_bytes = 5 * 1024 * 1024
    pct_used = float(total_bytes / budget_bytes) if budget_bytes > 0 else 0.0
    serving_block = {
        "total_bytes": total_bytes,
        "budget_bytes": budget_bytes,
        "pct_used": pct_used,
        "within_budget": total_bytes <= budget_bytes,
    }

    return HealthReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        overall_status=determine_overall_status(alerts),
        pipeline_status=pipeline_block,
        data_quality=quality_block,
        model_health=model_block,
        drift_indicators=drift_results,
        active_alerts=alerts_to_json(alerts),
        serving_health=serving_block,
        recommendations=build_recommendations(alerts, drift_results, model_metrics),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_health_report(report: HealthReport, output_path: Path) -> Path:
    """Serialise a HealthReport to JSON and write to output_path.

    Parameters
    ----------
    report:
        HealthReport dataclass instance.
    output_path:
        Destination file path (parent directories created automatically).

    Returns
    -------
    output_path (for chaining / confirmation).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": report.generated_at,
        "overall_status": report.overall_status,
        "pipeline_status": report.pipeline_status,
        "data_quality": report.data_quality,
        "model_health": report.model_health,
        "drift_indicators": report.drift_indicators,
        "active_alerts": report.active_alerts,
        "serving_health": report.serving_health,
        "recommendations": report.recommendations,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Health report saved to %s", output_path)
    return output_path
