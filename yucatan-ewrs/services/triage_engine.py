"""Post-impact incident triage with a Human-in-the-Loop dispatch workflow.

Incidents are scored on a 0–100 priority scale combining severity, vulnerable
population affected, infrastructure criticality, accessibility and the current
AI risk level of the municipality. A greedy allocator then suggests a brigade
for each incident (highest priority first) subject to unit availability.
The final decision always belongs to the PROCIVY operator.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INCIDENTS_PATH = ROOT / "data" / "sample_incidents.json"

WEIGHTS: dict[str, float] = {
    "severity": 0.45,
    "vulnerable": 0.20,
    "infrastructure": 0.15,
    "access": 0.15,
    "area_risk": 0.05,
}
# Life-safety rule: imminent threat to life (severity 5) always jumps the queue.
LIFE_SAFETY_BONUS = 10.0
ACCESS_SCORES: dict[str, float] = {"open": 0.2, "restricted": 0.6, "cut_off": 1.0}

# Brigade types that can stand in when the preferred one is exhausted.
ALTERNATES: dict[str, list[str]] = {
    "Medical Evacuation": ["Water Rescue", "Search & Assessment"],
    "Water Rescue": ["Medical Evacuation", "Search & Assessment"],
    "Road Clearance": ["Search & Assessment"],
    "Power Restoration": ["Shelter Logistics"],
    "Shelter Logistics": ["Search & Assessment"],
    "Search & Assessment": ["Road Clearance"],
}

STATUS_PENDING = "Pending review"
STATUS_APPROVED = "Approved"
STATUS_OVERRIDDEN = "Overridden"
UNASSIGNED = "Queue – no unit available"


def load_incident_data(path: Path = INCIDENTS_PATH) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load incident reports and the brigade roster."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return pd.DataFrame(payload["incidents"]), payload["brigades"]


def priority_band(score: float) -> str:
    """Translate a priority score into an operational band."""
    if score >= 75:
        return "P1 – Immediate"
    if score >= 55:
        return "P2 – Urgent"
    if score >= 35:
        return "P3 – Priority"
    return "P4 – Routine"


def score_incidents(incidents: pd.DataFrame, area_risk: dict[str, float] | None = None) -> pd.DataFrame:
    """Compute a 0–100 priority score for every incident.

    Args:
        incidents: Incident reports.
        area_risk: Optional mapping municipality -> AI risk score (0–100).

    Returns:
        Copy of ``incidents`` with ``priority_score`` and ``priority_band``,
        sorted by descending priority.
    """
    area_risk = area_risk or {}
    df = incidents.copy()
    # Log scale so a handful of dialysis patients is not dwarfed by a city outage.
    vuln_norm = df["vulnerable_people"].apply(lambda v: min(math.log10(v + 1) / math.log10(5000), 1.0))
    df["priority_score"] = 100 * (
        WEIGHTS["severity"] * df["severity"] / 5
        + WEIGHTS["vulnerable"] * vuln_norm
        + WEIGHTS["infrastructure"] * df["infrastructure_criticality"] / 5
        + WEIGHTS["access"] * df["access"].map(ACCESS_SCORES).fillna(0.5)
        + WEIGHTS["area_risk"] * df["municipality"].map(area_risk).fillna(50) / 100
    )
    df.loc[df["severity"] >= 5, "priority_score"] += LIFE_SAFETY_BONUS
    df["priority_score"] = df["priority_score"].clip(upper=100).round(1)
    df["priority_band"] = df["priority_score"].apply(priority_band)
    return df.sort_values(["priority_score", "severity"], ascending=False).reset_index(drop=True)


def suggest_allocation(scored: pd.DataFrame, brigades: list[dict[str, Any]]) -> pd.DataFrame:
    """Greedy AI brigade suggestion in priority order.

    Uses the required brigade type when units remain, otherwise an alternate
    type, otherwise queues the incident.
    """
    available = {b["type"]: int(b["units"]) for b in brigades}
    agencies = {b["type"]: b["agency"] for b in brigades}
    suggestions, rationales = [], []
    for row in scored.itertuples():
        wanted = row.required_brigade
        chosen, why = None, ""
        if available.get(wanted, 0) > 0:
            chosen, why = wanted, "Matches required capability"
        else:
            for alt in ALTERNATES.get(wanted, []):
                if available.get(alt, 0) > 0:
                    chosen, why = alt, f"{wanted} units exhausted → nearest alternate capability"
                    break
        if chosen:
            available[chosen] -= 1
            suggestions.append(f"{chosen} ({agencies[chosen]})")
        else:
            suggestions.append(UNASSIGNED)
            why = f"All {wanted} and alternate units committed to higher-priority incidents"
        rationales.append(why)
    out = scored.copy()
    out["ai_suggestion"] = suggestions
    out["ai_rationale"] = rationales
    return out


def brigade_options(brigades: list[dict[str, Any]]) -> list[str]:
    """Labels an operator can choose from when overriding."""
    return [f"{b['type']} ({b['agency']})" for b in brigades] + [UNASSIGNED, "Hold – request more information"]


def init_decisions(incident_ids: list[str], state: dict[str, Any]) -> None:
    """Ensure every incident has a decision record in session state."""
    decisions = state.setdefault("triage_decisions", {})
    for iid in incident_ids:
        decisions.setdefault(iid, {"status": STATUS_PENDING, "assigned": None, "note": "", "time": None})
    state.setdefault("triage_audit", [])


def record_decision(state: dict[str, Any], incident_id: str, status: str, assigned: str, note: str = "", operator: str = "PROCIVY operator") -> None:
    """Store an operator decision and append it to the audit log."""
    stamp = datetime.now().strftime("%H:%M:%S")
    state["triage_decisions"][incident_id] = {"status": status, "assigned": assigned, "note": note, "time": stamp}
    state["triage_audit"].insert(
        0, {"time": stamp, "incident": incident_id, "decision": status, "assigned": assigned, "operator": operator, "note": note}
    )


def reset_decisions(state: dict[str, Any]) -> None:
    """Clear all operator decisions and the audit log."""
    state.pop("triage_decisions", None)
    state.pop("triage_audit", None)
