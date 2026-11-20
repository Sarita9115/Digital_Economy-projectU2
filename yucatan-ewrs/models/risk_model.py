"""Hyperlocal hurricane risk model for the Yucatán EWRS prototype.

The module builds a deterministic, physically plausible synthetic training set
(municipality static features x hurricane scenario inputs), trains a
``RandomForestRegressor`` that predicts a 0–100 risk score, persists it with
``joblib`` and exposes :func:`predict_municipality_risks` for the dashboard.

Per-municipality explanations are computed with a local occlusion analysis:
each feature is replaced by its training-set mean and the resulting drop in the
predicted score is reported as that feature's contribution.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "municipalities.json"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "risk_model.joblib"

MODEL_VERSION = "1.0.0"
RANDOM_SEED = 42
N_SAMPLES = 1500

FEATURES: list[str] = [
    "category",
    "rainfall_mm_24h",
    "storm_surge_m",
    "distance_to_track_km",
    "coastal",
    "flood_vulnerability",
    "elevation_m",
    "maya_speaking_pct",
    "shelter_ratio",
    "log_population",
]

FEATURE_LABELS: dict[str, str] = {
    "category": "Hurricane intensity",
    "rainfall_mm_24h": "Extreme rainfall",
    "storm_surge_m": "Storm surge",
    "distance_to_track_km": "Proximity to storm track",
    "coastal": "Coastal exposure",
    "flood_vulnerability": "Flood vulnerability",
    "elevation_m": "Low elevation",
    "maya_speaking_pct": "Language barrier (Maya-speaking pop.)",
    "shelter_ratio": "Limited shelter capacity",
    "log_population": "Population density",
}

RISK_LEVELS: list[tuple[float, str]] = [
    (25.0, "Low"),
    (50.0, "Moderate"),
    (75.0, "High"),
    (math.inf, "Critical"),
]

RISK_COLORS: dict[str, str] = {
    "Low": "#2E9E6A",
    "Moderate": "#E3B23C",
    "High": "#E07B39",
    "Critical": "#C8423B",
}

# Simplified storm tracks (lat, lon) for each landfall zone. The track passes
# through the landfall point; distance to the polyline drives hazard decay.
LANDFALL_TRACKS: dict[str, list[tuple[float, float]]] = {
    "North Coast": [(22.60, -88.40), (21.32, -89.25), (20.30, -90.00)],
    "West Coast": [(21.40, -91.60), (20.86, -90.42), (20.20, -89.60)],
    "East Coast": [(21.90, -86.40), (21.48, -87.62), (20.80, -88.70)],
    "Inland": [(19.70, -87.20), (20.45, -88.60), (20.95, -90.20)],
}

# Share of the exposed population expected to seek a public shelter.
SHELTER_SEEKING_RATE = 0.25


@dataclass(frozen=True)
class Scenario:
    """Hurricane scenario inputs selected in the sidebar simulator."""

    category: int
    rainfall_mm_24h: float
    storm_surge_m: float
    landfall_zone: str


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def load_municipalities(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load municipality baseline indicators and metadata.

    Returns:
        A DataFrame with one row per municipality and the metadata dictionary.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    return pd.DataFrame(payload["municipalities"]), payload.get("metadata", {})


def _to_km(lat: float, lon: float, ref_lat: float) -> tuple[float, float]:
    """Project lat/lon to a local equirectangular plane in kilometres."""
    return lon * 111.32 * math.cos(math.radians(ref_lat)), lat * 110.57


def distance_to_track_km(lat: float, lon: float, landfall_zone: str) -> float:
    """Minimum distance (km) from a point to the storm-track polyline."""
    track = LANDFALL_TRACKS[landfall_zone]
    px, py = _to_km(lat, lon, lat)
    best = math.inf
    for (lat1, lon1), (lat2, lon2) in zip(track[:-1], track[1:]):
        ax, ay = _to_km(lat1, lon1, lat)
        bx, by = _to_km(lat2, lon2, lat)
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        t = 0.0 if seg_len2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len2))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


def risk_level(score: float) -> str:
    """Map a 0–100 risk score to a categorical risk level."""
    for threshold, label in RISK_LEVELS:
        if score < threshold:
            return label
    return "Critical"


# --------------------------------------------------------------------------- #
# Synthetic ground truth
# --------------------------------------------------------------------------- #
def physical_risk_score(df: pd.DataFrame) -> np.ndarray:
    """Physically plausible 'ground-truth' risk formula used to label data.

    Combines wind, storm-surge and rainfall hazards (each decaying with
    distance to the track) with social and physical vulnerability.
    """
    cat = df["category"].to_numpy(float)
    rain = df["rainfall_mm_24h"].to_numpy(float)
    surge = df["storm_surge_m"].to_numpy(float)
    dist = df["distance_to_track_km"].to_numpy(float)
    coastal = df["coastal"].to_numpy(float)
    flood = df["flood_vulnerability"].to_numpy(float)
    elev = df["elevation_m"].to_numpy(float)
    maya = df["maya_speaking_pct"].to_numpy(float) / 100.0
    shelter = df["shelter_ratio"].to_numpy(float)
    logpop = df["log_population"].to_numpy(float)

    wind = (cat / 5.0) ** 1.6 * np.exp(-dist / 110.0)
    elev_factor = np.clip(1.0 - elev / 10.0, 0.0, 1.0)
    surge_h = coastal * np.clip(surge / 6.0, 0.0, 1.0) * np.exp(-dist / 70.0) * (0.3 + 0.7 * elev_factor)
    rain_h = np.clip(rain / 500.0, 0.0, 1.0) * (0.4 + 0.6 * flood) * np.exp(-dist / 260.0)

    hazard = np.clip(0.40 * wind + 0.35 * surge_h + 0.35 * rain_h, 0.0, 1.0)
    shelter_gap = np.clip(1.0 - shelter / 0.20, 0.0, 1.0)
    pop_density = np.clip((logpop - 8.0) / 6.0, 0.0, 1.0)
    vulnerability = 0.45 * flood + 0.20 * maya + 0.20 * shelter_gap + 0.15 * pop_density

    return np.clip(100.0 * hazard * (0.75 + 0.9 * vulnerability) * 1.25, 0.0, 100.0)


def generate_synthetic_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create a deterministic synthetic training dataset.

    Each row samples one of the 8 municipalities (with ±10% jitter on static
    indicators so the model generalises) and a random hurricane scenario.
    """
    rng = np.random.default_rng(seed)
    munis, _ = load_municipalities()
    idx = rng.integers(0, len(munis), n_samples)
    base = munis.iloc[idx].reset_index(drop=True)

    def jitter(values: pd.Series, scale: float = 0.10) -> np.ndarray:
        return values.to_numpy(float) * rng.uniform(1 - scale, 1 + scale, n_samples)

    population = jitter(base["population"])
    df = pd.DataFrame(
        {
            "category": rng.integers(1, 6, n_samples),
            "rainfall_mm_24h": rng.uniform(40, 500, n_samples).round(0),
            "storm_surge_m": rng.uniform(0.0, 6.0, n_samples).round(2),
            "distance_to_track_km": np.clip(rng.gamma(2.0, 60.0, n_samples), 0, 350).round(1),
            "coastal": base["coastal"].astype(int).to_numpy(),
            "flood_vulnerability": np.clip(jitter(base["flood_vulnerability"]), 0, 1),
            "elevation_m": np.clip(jitter(base["elevation_m"], 0.25), 0.5, None),
            "maya_speaking_pct": np.clip(jitter(base["maya_speaking_pct"]), 0, 100),
            "shelter_ratio": jitter(base["shelter_capacity"]) / population,
            "log_population": np.log(population),
        }
    )
    noise = rng.normal(0.0, 2.5, n_samples)
    df["risk_score"] = np.clip(physical_risk_score(df) + noise, 0.0, 100.0)
    return df


# --------------------------------------------------------------------------- #
# Training / persistence
# --------------------------------------------------------------------------- #
def train_model(save: bool = True) -> dict[str, Any]:
    """Train the RandomForest risk model and optionally persist it.

    Returns:
        Artifact dictionary with the model, feature list, baseline means,
        evaluation metrics and global feature importances.
    """
    df = generate_synthetic_dataset()
    X, y = df[FEATURES], df["risk_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = RandomForestRegressor(
        n_estimators=250, max_depth=14, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    artifact: dict[str, Any] = {
        "version": MODEL_VERSION,
        "model": model,
        "features": FEATURES,
        "feature_means": X_train.mean().to_dict(),
        "metrics": {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "feature_importances": dict(zip(FEATURES, model.feature_importances_.round(4).tolist())),
    }
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, ARTIFACT_PATH)
    return artifact


def load_model(retrain_if_missing: bool = True) -> dict[str, Any]:
    """Load the persisted artifact, retraining if it is missing or stale."""
    if ARTIFACT_PATH.exists():
        try:
            artifact = joblib.load(ARTIFACT_PATH)
            if artifact.get("version") == MODEL_VERSION and artifact.get("features") == FEATURES:
                return artifact
        except Exception:  # corrupted or incompatible pickle -> retrain
            pass
    if not retrain_if_missing:
        raise FileNotFoundError(f"No valid model artifact at {ARTIFACT_PATH}")
    return train_model(save=True)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_feature_frame(scenario: Scenario, munis: pd.DataFrame) -> pd.DataFrame:
    """Combine a scenario with municipality indicators into model features."""
    return pd.DataFrame(
        {
            "category": scenario.category,
            "rainfall_mm_24h": scenario.rainfall_mm_24h,
            "storm_surge_m": scenario.storm_surge_m,
            "distance_to_track_km": [
                distance_to_track_km(r.lat, r.lon, scenario.landfall_zone) for r in munis.itertuples()
            ],
            "coastal": munis["coastal"].astype(int),
            "flood_vulnerability": munis["flood_vulnerability"],
            "elevation_m": munis["elevation_m"],
            "maya_speaking_pct": munis["maya_speaking_pct"],
            "shelter_ratio": munis["shelter_capacity"] / munis["population"],
            "log_population": np.log(munis["population"]),
        }
    )[FEATURES]


def _describe_factor(feature: str, row: pd.Series) -> str:
    """Human-readable label for a contributing factor with its value."""
    value = row[feature]
    detail = {
        "category": f"Category {int(value)}",
        "rainfall_mm_24h": f"{value:.0f} mm/24h",
        "storm_surge_m": f"{value:.1f} m surge",
        "distance_to_track_km": f"{value:.0f} km from track",
        "coastal": "coastline",
        "flood_vulnerability": f"index {value:.2f}",
        "elevation_m": f"{value:.0f} m a.s.l.",
        "maya_speaking_pct": f"{value:.0f}% Maya-speaking",
        "shelter_ratio": f"{value * 100:.0f} beds / 100 residents",
        "log_population": f"{math.exp(value):,.0f} residents",
    }[feature]
    return f"{FEATURE_LABELS[feature]} ({detail})"


def explain_local(artifact: dict[str, Any], X: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, Any]]]:
    """Occlusion-based local explanation of each prediction.

    For each feature, replace it with its training mean and record how much the
    prediction falls. Returns the ``top_k`` positive drivers per row.
    """
    model = artifact["model"]
    base_pred = model.predict(X)
    contributions = np.zeros((len(X), len(FEATURES)))
    for j, feat in enumerate(FEATURES):
        X_ref = X.copy()
        X_ref[feat] = artifact["feature_means"][feat]
        contributions[:, j] = base_pred - model.predict(X_ref)

    explanations: list[list[dict[str, Any]]] = []
    for i in range(len(X)):
        order = np.argsort(-contributions[i])[:top_k]
        explanations.append(
            [
                {
                    "feature": FEATURES[j],
                    "label": _describe_factor(FEATURES[j], X.iloc[i]),
                    "contribution": round(float(contributions[i, j]), 1),
                }
                for j in order
            ]
        )
    return explanations


def estimate_exposed_population(population: float, score: float, coastal: bool) -> int:
    """Estimate residents directly exposed to life-threatening hazards."""
    fraction = (score / 100.0) ** 2 * (0.85 if coastal else 0.30)
    return int(round(population * fraction, -1))


def predict_municipality_risks(scenario: Scenario | dict[str, Any], artifact: dict[str, Any] | None = None) -> pd.DataFrame:
    """Predict hyperlocal risk for every municipality under a scenario.

    Args:
        scenario: A :class:`Scenario` or dict with ``category``,
            ``rainfall_mm_24h``, ``storm_surge_m`` and ``landfall_zone``.
        artifact: Optional pre-loaded model artifact.

    Returns:
        DataFrame with municipality info plus ``risk_score``, ``risk_level``,
        ``distance_to_track_km``, ``exposed_population``, ``shelter_demand``,
        ``top_factors`` (list of dicts) and ``top_factors_text``.
    """
    if isinstance(scenario, dict):
        scenario = Scenario(**scenario)
    artifact = artifact or load_model()
    munis, _ = load_municipalities()
    X = build_feature_frame(scenario, munis)

    scores = np.clip(artifact["model"].predict(X), 0, 100).round(1)
    explanations = explain_local(artifact, X)

    out = munis.copy()
    out["distance_to_track_km"] = X["distance_to_track_km"].round(0).to_numpy()
    out["risk_score"] = scores
    out["risk_level"] = [risk_level(s) for s in scores]
    out["exposed_population"] = [
        estimate_exposed_population(p, s, c) for p, s, c in zip(out["population"], scores, out["coastal"])
    ]
    out["shelter_demand"] = (out["exposed_population"] * SHELTER_SEEKING_RATE).round().astype(int)
    out["top_factors"] = explanations
    out["top_factors_text"] = [" · ".join(f["label"] for f in exp) for exp in explanations]
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    """Serialise a scenario (handy for caching keys and prompts)."""
    return asdict(scenario)


if __name__ == "__main__":
    art = train_model(save=True)
    print(f"Model saved to {ARTIFACT_PATH}")
    print("Metrics:", art["metrics"])
    demo = predict_municipality_risks(Scenario(4, 300, 3.5, "North Coast"), art)
    print(demo[["name", "risk_score", "risk_level", "exposed_population", "top_factors_text"]].to_string())
