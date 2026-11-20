"""Multilingual emergency alert generation (Spanish, Yucatec Maya, English).

Alerts are produced for three channels — Cell Broadcast SMS (<=160 chars),
WhatsApp community message and a 30-second community radio script.

Generation strategy:
    1. If Online Mode is active and an API key is configured, call the
       configured LLM provider (Google Gemini or OpenAI).
    2. Otherwise — or if the call fails, times out or returns malformed
       output — fill the pre-approved templates in ``data/fallback_alerts.json``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback_alerts.json"
load_dotenv(ROOT / ".env")

LANGUAGES: dict[str, str] = {"es": "Spanish", "yua": "Yucatec Maya", "en": "English"}
CHANNELS: dict[str, str] = {
    "sms": "Cell Broadcast SMS",
    "whatsapp": "WhatsApp Community Message",
    "radio": "30-sec Community Radio Script",
}
SMS_MAX_CHARS = 160
HOTLINE = "911"
LLM_TIMEOUT_S = 20

STATUS_LIVE = "Live LLM"
STATUS_FALLBACK = "Offline Fallback"

MAYA_DISCLAIMER = (
    "Yucatec Maya outputs are machine-assisted drafts. They must be validated by native speakers "
    "and INDEMAYA (Instituto para el Desarrollo de la Cultura Maya del Estado de Yucatán) before any "
    "real-world deployment."
)


@dataclass
class AlertResult:
    """Generated alert bundle and its provenance."""

    municipality: str
    risk_level: str
    alerts: dict[str, dict[str, str]]  # {lang: {channel: text}}
    status: str  # STATUS_LIVE or STATUS_FALLBACK
    provider: str
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def get_llm_config() -> dict[str, str | None]:
    """Read LLM provider settings from the environment / ``.env``."""
    provider = (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()
    if provider == "openai":
        key, model = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
    else:
        provider = "gemini"
        key, model = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
    return {"provider": provider, "api_key": (key or "").strip() or None, "model": model}


def llm_available() -> bool:
    """True when an API key for the configured provider is present."""
    return get_llm_config()["api_key"] is not None


def load_fallback_library(path: Path = FALLBACK_PATH) -> dict[str, Any]:
    """Load the offline template library."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _short_factor(profile: dict[str, Any]) -> str:
    """Main risk driver label without the parenthesised value."""
    factors = profile.get("top_factors") or []
    if not factors:
        return "hurricane hazard"
    return re.sub(r"\s*\(.*\)$", "", factors[0]["label"])


def enforce_sms_limit(text: str, limit: int = SMS_MAX_CHARS) -> str:
    """Collapse whitespace and trim an SMS to ``limit`` characters."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip(" ,.;:") + "…"


def _template_values(profile: dict[str, Any], scenario: dict[str, Any], lang: str, labels: dict[str, Any]) -> dict[str, str]:
    """Placeholder values for template injection in a given language."""
    level = profile["risk_level"]
    return {
        "municipality": profile["name"],
        "locality": profile.get("locality", profile["name"]),
        "category": str(scenario["category"]),
        "risk_level": labels.get(lang, {}).get(level, level.upper()),
        "risk_score": f"{profile['risk_score']:.0f}",
        "rainfall": f"{scenario['rainfall_mm_24h']:.0f}",
        "surge": f"{scenario['storm_surge_m']:.1f}",
        "exposed": f"{profile.get('exposed_population', 0):,}",
        "top_factor": _short_factor(profile),
        "hotline": HOTLINE,
    }


# --------------------------------------------------------------------------- #
# Offline fallback
# --------------------------------------------------------------------------- #
def generate_fallback_alerts(profile: dict[str, Any], scenario: dict[str, Any], reason: str = "Offline / Demo Mode") -> AlertResult:
    """Fill pre-approved templates with municipality / risk values."""
    library = load_fallback_library()
    level = profile["risk_level"]
    templates = library["templates"].get(level) or library["templates"]["High"]
    labels = library.get("risk_labels", {})

    alerts: dict[str, dict[str, str]] = {}
    for lang in LANGUAGES:
        values = _template_values(profile, scenario, lang, labels)
        alerts[lang] = {ch: templates[lang][ch].format(**values) for ch in CHANNELS}
        alerts[lang]["sms"] = enforce_sms_limit(alerts[lang]["sms"])

    return AlertResult(
        municipality=profile["name"],
        risk_level=level,
        alerts=alerts,
        status=STATUS_FALLBACK,
        provider="Pre-approved templates (data/fallback_alerts.json)",
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# Live LLM
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an emergency communications officer for PROCIVY (Protección Civil de Yucatán, Mexico). "
    "Write clear, calm, actionable hurricane alerts for the public. Never invent shelter names, "
    "addresses or phone numbers other than the hotline provided. Use plain language suitable for "
    "low-literacy audiences. Respond ONLY with valid JSON."
)


def build_prompt(profile: dict[str, Any], scenario: dict[str, Any]) -> str:
    """Construct the user prompt describing the risk profile and output schema."""
    factors = "; ".join(f["label"] for f in profile.get("top_factors", [])) or "n/a"
    return f"""
Generate hurricane emergency alerts for the municipality of {profile['name']} (locality: {profile.get('locality', profile['name'])}), Yucatán, Mexico.

RISK PROFILE
- Hurricane category: {scenario['category']}
- Rainfall: {scenario['rainfall_mm_24h']:.0f} mm in 24 h
- Storm surge: {scenario['storm_surge_m']:.1f} m
- Landfall zone: {scenario['landfall_zone']}
- AI risk score: {profile['risk_score']:.0f}/100 → risk level {profile['risk_level']}
- Estimated exposed population: {profile.get('exposed_population', 0):,}
- Top risk drivers: {factors}
- Emergency hotline: {HOTLINE}

ACTION GUIDANCE BY LEVEL
- Critical: mandatory immediate evacuation to the nearest official shelter.
- High: prepare to evacuate; low-lying/coastal residents go to shelter now.
- Moderate: prepare go-bag, locate shelter, clear drains.
- Low: stay informed, review family plan.

OUTPUT: a JSON object with keys "es" (Spanish), "yua" (Yucatec Maya), "en" (English).
Each language maps to an object with keys:
- "sms": Cell Broadcast text, MAXIMUM 160 characters, starting with "PROCIVY".
- "whatsapp": community message (80–140 words), may use emojis and *bold*, numbered actions.
- "radio": 30-second radio script (~70–80 words), begin with "[ALERT TONE]" or equivalent.
For Yucatec Maya use natural Yucatec Maya; keep "PROCIVY", the hotline and numbers unchanged.
""".strip()


def _call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Call Google Gemini via ``google-genai`` and return raw text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=LLM_TIMEOUT_S * 1000))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.4,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def _call_openai(prompt: str, api_key: str, model: str) -> str:
    """Call OpenAI Chat Completions and return raw text."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=LLM_TIMEOUT_S, max_retries=1)
    response = client.chat.completions.create(
        model=model,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def parse_llm_output(raw: str) -> dict[str, dict[str, str]]:
    """Parse and validate the LLM JSON payload.

    Raises:
        ValueError: if any language/channel is missing or empty.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    data = json.loads(cleaned)
    alerts: dict[str, dict[str, str]] = {}
    for lang in LANGUAGES:
        block = data.get(lang)
        if not isinstance(block, dict):
            raise ValueError(f"Missing language block '{lang}'")
        alerts[lang] = {}
        for ch in CHANNELS:
            text = str(block.get(ch, "")).strip()
            if not text:
                raise ValueError(f"Missing '{ch}' text for '{lang}'")
            alerts[lang][ch] = text
    return alerts


def generate_llm_alerts(profile: dict[str, Any], scenario: dict[str, Any]) -> AlertResult:
    """Generate alerts with the configured LLM (raises on any failure)."""
    cfg = get_llm_config()
    if not cfg["api_key"]:
        raise RuntimeError("No API key configured")
    prompt = build_prompt(profile, scenario)
    caller = _call_openai if cfg["provider"] == "openai" else _call_gemini
    alerts = parse_llm_output(caller(prompt, cfg["api_key"], cfg["model"]))

    warnings: list[str] = []
    for lang in LANGUAGES:
        original = alerts[lang]["sms"]
        alerts[lang]["sms"] = enforce_sms_limit(original)
        if len(re.sub(r"\s+", " ", original).strip()) > SMS_MAX_CHARS:
            warnings.append(f"{LANGUAGES[lang]} SMS exceeded {SMS_MAX_CHARS} chars and was trimmed.")

    return AlertResult(
        municipality=profile["name"],
        risk_level=profile["risk_level"],
        alerts=alerts,
        status=STATUS_LIVE,
        provider=f"{cfg['provider'].title()} · {cfg['model']}",
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def generate_alerts(profile: dict[str, Any], scenario: dict[str, Any], online: bool = True) -> AlertResult:
    """Generate multilingual alerts with automatic offline fallback.

    Args:
        profile: Municipality risk profile (row from ``predict_municipality_risks``).
        scenario: Scenario dictionary (category, rainfall, surge, landfall zone).
        online: False when Offline / Demo Mode or a telecom outage is active.

    Returns:
        An :class:`AlertResult`; never raises for LLM problems.
    """
    if not online:
        return generate_fallback_alerts(profile, scenario, reason="Offline / Demo Mode active")
    if not llm_available():
        return generate_fallback_alerts(profile, scenario, reason="No LLM API key configured")
    try:
        return generate_llm_alerts(profile, scenario)
    except Exception as exc:  # network, timeout, quota, malformed JSON, missing SDK…
        detail = f"{type(exc).__name__}: {exc}"
        return generate_fallback_alerts(profile, scenario, reason=f"LLM call failed → {detail[:160]}")
