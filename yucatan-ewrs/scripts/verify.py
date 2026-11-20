"""Pre-demo verification for Yucatán EWRS.

Checks that:
  1. The ML model trains, saves, reloads and predicts sane values.
  2. The fallback alert generator works with no API key (all levels, SMS <= 160).
  3. The triage engine scores and allocates incidents.
  4. ``app.py`` compiles and runs end-to-end in Streamlit's headless AppTest.

Usage:  python scripts/verify.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Guarantee the "no API key" path regardless of the local .env.
for var in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
    os.environ[var] = ""


def check(label: str, condition: bool) -> None:
    """Print a pass/fail line and abort on failure."""
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        sys.exit(1)


def verify_model() -> None:
    from models.risk_model import ARTIFACT_PATH, Scenario, load_model, predict_municipality_risks, train_model

    print("[1] ML risk model")
    art = train_model(save=True)
    check(f"trained RandomForest (R²={art['metrics']['r2']:.3f}, MAE={art['metrics']['mae']:.2f})", art["metrics"]["r2"] > 0.8)
    check("artifact saved with joblib", ARTIFACT_PATH.exists())
    art = load_model()
    df = predict_municipality_risks(Scenario(5, 450, 5.0, "North Coast"), art)
    check("8 municipalities predicted", len(df) == 8)
    check("scores within 0–100", df["risk_score"].between(0, 100).all())
    check("3 risk drivers per municipality", all(len(f) == 3 for f in df["top_factors"]))
    calm = predict_municipality_risks(Scenario(1, 60, 0.3, "East Coast"), art)
    check("Cat 5 scenario riskier than Cat 1", df["risk_score"].mean() > calm["risk_score"].mean())


def verify_alerts() -> None:
    from models.risk_model import Scenario, load_model, predict_municipality_risks
    from services.alert_generator import CHANNELS, LANGUAGES, SMS_MAX_CHARS, STATUS_FALLBACK, generate_alerts, llm_available

    print("[2] Alert generator (no API key)")
    check("no API key detected", not llm_available())
    scenario = {"category": 4, "rainfall_mm_24h": 320.0, "storm_surge_m": 3.5, "landfall_zone": "West Coast"}
    df = predict_municipality_risks(Scenario(**scenario), load_model())
    for level in ("Critical", "High", "Moderate", "Low"):
        for _, row in df.iterrows():
            profile = row.to_dict() | {"risk_level": level}
            res = generate_alerts(profile, scenario, online=True)
            assert res.status == STATUS_FALLBACK, res
            for lang in LANGUAGES:
                for ch in CHANNELS:
                    text = res.alerts[lang][ch]
                    assert text and "{" not in text, (level, lang, ch, text)
                assert len(res.alerts[lang]["sms"]) <= SMS_MAX_CHARS, res.alerts[lang]["sms"]
    check("fallback used for all 4 levels × 8 municipalities × 3 languages × 3 channels", True)
    check("all SMS <= 160 characters and placeholders filled", True)
    offline = generate_alerts(df.iloc[0].to_dict(), scenario, online=False)
    check("Offline / Demo Mode forces fallback", offline.status == STATUS_FALLBACK)


def verify_triage() -> None:
    from services.triage_engine import load_incident_data, score_incidents, suggest_allocation

    print("[3] Triage engine")
    incidents, brigades = load_incident_data()
    queue = suggest_allocation(score_incidents(incidents), brigades)
    check(f"{len(queue)} incidents scored and ranked", queue["priority_score"].is_monotonic_decreasing)
    check("every incident has an AI suggestion", queue["ai_suggestion"].notna().all())


def verify_app() -> None:
    from streamlit.testing.v1 import AppTest

    print("[4] Streamlit app")
    compile((ROOT / "app.py").read_text(encoding="utf-8"), "app.py", "exec")
    check("app.py compiles", True)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    check("app.py runs headless without exceptions", not at.exception)
    gen = next(b for b in at.button if b.label.startswith("📣"))
    gen.click().run()
    check("Generate Alert works (offline)", not at.exception)
    for t in at.toggle:
        t.set_value(True)
    at.run()
    check("outage + offline toggles render without exceptions", not at.exception)
    approve = next(b for b in at.button if b.label == "Approve")
    approve.click().run()
    check("HITL approve action works", not at.exception and len(at.session_state["triage_audit"]) == 1)


if __name__ == "__main__":
    verify_model()
    verify_alerts()
    verify_triage()
    verify_app()
    print("\nAll checks passed ✅")
