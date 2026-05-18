"""
EHA MVP v2 - Model Serving API (BentoML 1.4.x)
================================================
Serves all four trained models:
    Cholera Option A, Cholera Option B
    Malaria Option A, Malaria Option B

Start:
    bentoml serve serving/serve_final.py --port 3000

Endpoints:
    POST /predict    Single county prediction
    POST /health     Service status

Example request:
    {
        "county": "Nakuru",
        "year": 2024,
        "disease": "cholera",
        "option": "a",
        "features": {
            "total_rainfall_mm": 850.5,
            "mean_rainfall_mm": 70.9,
            ...
        }
    }
"""

import json
import numpy as np
import xgboost as xgb
import shap
import bentoml
from bentoml.io import JSON
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("data/models")
CONFIG_DIR = Path("serving/config")

DISCLAIMER = (
    "Model-generated risk signal for early-warning purposes only. "
    "Always validate with field surveillance data before taking action."
)

# Climate-only features — geographic identifiers excluded (see training scripts).
FEATURES_A = [
    "total_rainfall_mm", "mean_rainfall_mm", "max_rainfall_mm",
    "rainfall_variability", "mean_temperature_c", "max_temperature_c",
    "min_temperature_c", "temp_range_c", "peak_rainfall_month",
]

FEATURES_B = [
    "avg_rainfall_mm", "avg_rainfall_mm_lag1", "avg_rainfall_mm_lag2",
    "avg_rainfall_mm_roll3",
    "mean_temperature_celcius", "mean_temperature_celcius_lag1",
    "mean_temperature_celcius_lag2", "mean_temperature_celcius_roll3",
    "month_sin", "month_cos",
]

VALID_DISEASES = ("cholera", "malaria")
VALID_OPTIONS  = ("a", "b")


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    county:   str
    year:     int
    disease:  str = "cholera"
    option:   str = "a"
    features: Dict[str, float]


class RiskDriver(BaseModel):
    feature:    str
    impact:     str
    shap_value: float


class PredictResponse(BaseModel):
    county:           str
    year:             int
    disease:          str
    risk_level:       str
    risk_probability: float
    confidence:       str
    prediction:       int
    threshold_used:   float
    top_risk_drivers: List[RiskDriver]
    model_key:        str
    disclaimer:       str


class HealthResponse(BaseModel):
    status:        str
    models_loaded: List[str]
    project:       str
    version:       str


# ── Model Registry ────────────────────────────────────────────────────────────
def _load_all():
    """Load all available models at startup. Missing models are skipped."""
    models     = {}
    explainers = {}
    thresholds = {}
    features   = {}

    for disease in VALID_DISEASES:
        for option in VALID_OPTIONS:
            key        = f"{disease}_{option}"
            model_path = MODELS_DIR / f"xgb_{key}.json"
            thresh_path = CONFIG_DIR / f"threshold_{key}.json"

            if not model_path.exists():
                print(f"  WARNING: {model_path} not found — skipping")
                continue

            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            models[key]     = model
            explainers[key] = shap.TreeExplainer(model)
            features[key]   = FEATURES_A if option == "a" else FEATURES_B

            threshold = 0.5
            if thresh_path.exists():
                with open(thresh_path) as f:
                    threshold = json.load(f)["threshold"]
            thresholds[key] = threshold

            print(f"  Loaded {key} | threshold={threshold:.3f}")

    return models, explainers, thresholds, features


print("Loading EHA risk models...")
_models, _explainers, _thresholds, _features = _load_all()


# ── Prediction Logic ──────────────────────────────────────────────────────────
def _confidence(probability, threshold):
    distance = abs(probability - threshold)
    if distance >= 0.30:
        return "High"
    if distance >= 0.15:
        return "Medium"
    return "Low"


def _predict(req: PredictRequest) -> PredictResponse:
    disease = req.disease.lower()
    option  = req.option.lower()

    if disease not in VALID_DISEASES:
        raise ValueError(f"disease must be one of {VALID_DISEASES}")
    if option not in VALID_OPTIONS:
        raise ValueError(f"option must be one of {VALID_OPTIONS}")

    key = f"{disease}_{option}"
    if key not in _models:
        raise ValueError(f"Model '{key}' is not loaded. "
                         f"Available: {list(_models.keys())}")

    model        = _models[key]
    explainer    = _explainers[key]
    threshold    = _thresholds[key]
    feature_cols = _features[key]

    missing = [f for f in feature_cols if f not in req.features]
    if missing:
        raise ValueError(f"Missing features for {key}: {missing}")

    X = np.array([[req.features[f] for f in feature_cols]])

    probability = float(model.predict_proba(X)[0][1])
    prediction  = int(probability >= threshold)
    risk_level  = "HIGH" if prediction == 1 else "LOW"
    confidence  = _confidence(probability, threshold)

    shap_vals   = explainer.shap_values(X)[0]
    top_idx     = np.argsort(np.abs(shap_vals))[::-1][:3]
    top_drivers = [
        RiskDriver(
            feature    = feature_cols[i],
            impact     = "increases risk" if shap_vals[i] > 0 else "decreases risk",
            shap_value = round(float(shap_vals[i]), 4),
        )
        for i in top_idx
    ]

    return PredictResponse(
        county           = req.county,
        year             = req.year,
        disease          = disease,
        risk_level       = risk_level,
        risk_probability = round(probability, 4),
        confidence       = confidence,
        prediction       = prediction,
        threshold_used   = round(threshold, 4),
        top_risk_drivers = top_drivers,
        model_key        = key,
        disclaimer       = DISCLAIMER,
    )


# ── BentoML Service ───────────────────────────────────────────────────────────
@bentoml.service(name="eha_disease_risk_predictor")
class EHARiskPredictor:

    def __init__(self):
        # Models are loaded at module level to avoid reloading per instance
        pass

    @bentoml.api
    def predict(self, request: PredictRequest) -> PredictResponse:
        """
        Single county disease risk prediction.
        Returns risk level, probability, confidence, and top 3 SHAP drivers.
        """
        return _predict(request)

    @bentoml.api
    def health(self, request: Dict) -> HealthResponse:
        """Service health check. Returns loaded model keys."""
        return HealthResponse(
            status        = "ok",
            models_loaded = list(_models.keys()),
            project       = "EHA24-7-MVP-v2",
            version       = "2.0.0",
        )
