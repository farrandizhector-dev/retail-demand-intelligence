"""Alert engine with 6 rules (spec §11.3).

Severity: CRITICAL > HIGH > MEDIUM > LOW
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class Alert:
    name: str
    trigger: str
    severity: Severity
    message: str
    timestamp: str  # ISO format
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Rule 1 — Data quality failure
# ---------------------------------------------------------------------------

def check_data_quality_failure(pipeline_status: dict[str, Any]) -> Alert | None:
    """Fire CRITICAL alert if any pipeline step has failed."""
    failed_steps = [
        step for step, status in pipeline_status.items()
        if status == "fail" or status is False
    ]
    if not failed_steps:
        return None

    return Alert(
        name="DATA_QUALITY_FAILURE",
        trigger="pipeline_step_failed",
        severity=Severity.CRITICAL,
        message=f"Data quality failure detected in steps: {failed_steps}",
        timestamp=_now_iso(),
        details={"failed_steps": failed_steps},
    )


# ---------------------------------------------------------------------------
# Rule 2 — Forecast degradation
# ---------------------------------------------------------------------------

def check_forecast_degradation(
    current_mae: float,
    baseline_mae: float,
    days_above: int,
) -> Alert | None:
    """Fire HIGH alert if MAE exceeds 115% of baseline for >= 14 days."""
    if current_mae > baseline_mae * 1.15 and days_above >= 14:
        pct_increase = (current_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0.0
        return Alert(
            name="FORECAST_DEGRADATION",
            trigger="mae_above_threshold",
            severity=Severity.HIGH,
            message=(
                f"Model MAE has exceeded baseline by {pct_increase:.1%} "
                f"for {days_above} consecutive days"
            ),
            timestamp=_now_iso(),
            details={
                "current_mae": float(current_mae),
                "baseline_mae": float(baseline_mae),
                "days_above": int(days_above),
                "pct_increase": float(pct_increase),
            },
        )
    return None


# ---------------------------------------------------------------------------
# Rule 3 — Feature drift detected
# ---------------------------------------------------------------------------

def check_drift_detected(drift_results: dict[str, dict[str, Any]]) -> Alert | None:
    """Fire MEDIUM alert if any feature has status 'alert' or 'retrain'."""
    drifted = [
        {"feature": feat, "psi": info.get("psi", 0.0), "status": info.get("status", "ok")}
        for feat, info in drift_results.items()
        if info.get("status") in ("alert", "retrain")
    ]
    if not drifted:
        return None

    feature_names = [d["feature"] for d in drifted]
    return Alert(
        name="DRIFT_DETECTED",
        trigger="psi_threshold_exceeded",
        severity=Severity.MEDIUM,
        message=f"Feature drift detected in: {feature_names}",
        timestamp=_now_iso(),
        details={"drifted_features": drifted},
    )


# ---------------------------------------------------------------------------
# Rule 4 — Inventory anomaly
# ---------------------------------------------------------------------------

def check_inventory_anomaly(store_fill_rates: dict[str, float]) -> Alert | None:
    """Fire HIGH alert if any store fill rate is below 85%."""
    affected = [
        {"store": store, "fill_rate": float(rate)}
        for store, rate in store_fill_rates.items()
        if rate < 0.85
    ]
    if not affected:
        return None

    store_ids = [a["store"] for a in affected]
    return Alert(
        name="INVENTORY_ANOMALY",
        trigger="fill_rate_below_threshold",
        severity=Severity.HIGH,
        message=f"Low fill rates detected in stores: {store_ids}",
        timestamp=_now_iso(),
        details={"affected_stores": affected},
    )


# ---------------------------------------------------------------------------
# Rule 5 — Reconciliation incoherence
# ---------------------------------------------------------------------------

def check_reconciliation_incoherence(
    coherence_passed: bool,
    details: dict[str, Any] | None = None,
) -> Alert | None:
    """Fire CRITICAL alert if hierarchical reconciliation coherence check fails."""
    if coherence_passed:
        return None

    return Alert(
        name="RECONCILIATION_INCOHERENCE",
        trigger="coherence_check_failed",
        severity=Severity.CRITICAL,
        message="Hierarchical reconciliation coherence check failed",
        timestamp=_now_iso(),
        details=details or {},
    )


# ---------------------------------------------------------------------------
# Rule 6 — Serving budget exceeded
# ---------------------------------------------------------------------------

def check_serving_budget(
    asset_sizes: dict[str, int],
    budget_bytes: int = 5 * 1024 * 1024,
) -> Alert | None:
    """Fire MEDIUM alert if total serving asset size exceeds budget."""
    total = sum(asset_sizes.values())
    if total <= budget_bytes:
        return None

    sorted_assets = sorted(asset_sizes.items(), key=lambda x: x[1], reverse=True)[:3]
    largest = [{"filename": name, "size_bytes": size} for name, size in sorted_assets]

    return Alert(
        name="SERVING_BUDGET_EXCEEDED",
        trigger="asset_total_over_budget",
        severity=Severity.MEDIUM,
        message=(
            f"Serving bundle {total / (1024 * 1024):.2f}MB exceeds "
            f"{budget_bytes / (1024 * 1024):.2f}MB budget"
        ),
        timestamp=_now_iso(),
        details={
            "total_bytes": int(total),
            "budget_bytes": int(budget_bytes),
            "largest_assets": largest,
        },
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def check_all_alerts(context: dict[str, Any]) -> list[Alert]:
    """Run all 6 alert rules and return non-None results.

    Context keys (all optional):
    - pipeline_status: dict (default {})
    - current_mae: float (default 0.0)
    - baseline_mae: float (default 0.0)
    - days_above_threshold: int (default 0)
    - drift_results: dict (default {})
    - store_fill_rates: dict (default {})
    - coherence_passed: bool (default True)
    - coherence_details: dict (default {})
    - asset_sizes: dict (default {})
    """
    alerts: list[Alert] = []

    checks = [
        check_data_quality_failure(context.get("pipeline_status", {})),
        check_forecast_degradation(
            current_mae=float(context.get("current_mae", 0.0)),
            baseline_mae=float(context.get("baseline_mae", 0.0)),
            days_above=int(context.get("days_above_threshold", 0)),
        ),
        check_drift_detected(context.get("drift_results", {})),
        check_inventory_anomaly(context.get("store_fill_rates", {})),
        check_reconciliation_incoherence(
            coherence_passed=bool(context.get("coherence_passed", True)),
            details=context.get("coherence_details", {}),
        ),
        check_serving_budget(context.get("asset_sizes", {})),
    ]

    for alert in checks:
        if alert is not None:
            alerts.append(alert)

    return alerts


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def alerts_to_json(alerts: list[Alert]) -> list[dict[str, Any]]:
    """Convert Alert objects to JSON-serialisable dicts."""
    return [
        {
            "name": a.name,
            "trigger": a.trigger,
            "severity": str(a.severity.value),
            "message": a.message,
            "timestamp": a.timestamp,
            "details": a.details,
        }
        for a in alerts
    ]
