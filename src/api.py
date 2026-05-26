"""
FastAPI Prediction Service

WHY  : Mobile apps, web apps, dashboards all need predictions via HTTP.
       A REST API is language-agnostic — any client can call it.
HOW  : FastAPI + Pydantic for request validation; model loaded at startup
WHEN : After model is promoted to Production in the registry
WHERE: Docker container, Kubernetes pod, cloud function
WHAT : Exposes /predict, /health, /model-info endpoints

Test with:
  uvicorn src.api:app --reload --port 8000
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Iris Classifier API",
    description = "Production ML model served via FastAPI + MLflow",
    version     = "1.0.0"
)

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MODEL_NAME          = os.getenv('MODEL_NAME', 'iris_classifier')
CLASS_NAMES         = ['setosa', 'versicolor', 'virginica']

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model once at startup (not on every request)
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        logger.info(f"✅  Model loaded: {MODEL_NAME}/Production")
    except Exception as e:
        logger.error(f"Could not load model: {e}")
        model = None


# ── Request / Response Schemas ────────────────────────────────────────────────
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width:  float = Field(..., gt=0, description="Petal width in cm")

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width":  3.5,
                "petal_length": 1.4,
                "petal_width":  0.2
            }
        }


class PredictionResponse(BaseModel):
    prediction:  int
    class_name:  str
    confidence:  float
    model_name:  str
    timestamp:   str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """
    WHY  : Load balancers and Kubernetes use /health to check if pod is alive.
    WHEN : Called every 30 seconds by orchestration layer.
    """
    return {
        "status"      : "healthy" if model else "model_not_loaded",
        "model_loaded": model is not None,
        "timestamp"   : datetime.now().isoformat()
    }


@app.get("/model-info")
def model_info():
    """Returns metadata about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name" : MODEL_NAME,
        "stage"      : "Production",
        "classes"    : CLASS_NAMES,
        "n_features" : 4,
        "feature_names": [
            "sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """
    WHY  : Core endpoint — returns predicted class for input measurements.
    HOW  : Pydantic validates input → pandas DataFrame → model.predict
    WHEN : Called by any client application needing a classification.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        data = pd.DataFrame([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]], columns=[
            'sepal length (cm)', 'sepal width (cm)',
            'petal length (cm)', 'petal width (cm)'
        ])

        prediction  = model.predict(data)[0]
        proba       = model.predict_proba(data)[0]
        confidence  = float(proba.max())

        return PredictionResponse(
            prediction = int(prediction),
            class_name = CLASS_NAMES[int(prediction)],
            confidence = round(confidence, 4),
            model_name = MODEL_NAME,
            timestamp  = datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
