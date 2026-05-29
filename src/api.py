"""
FastAPI Prediction Service — with Prometheus instrumentation

WHY  : Mobile apps, web apps, dashboards all need predictions via HTTP.
HOW  : FastAPI + Pydantic v2 + prometheus_fastapi_instrumentator
WHEN : After model is promoted to champion alias in the registry
WHAT : Exposes /predict /health /model-info /metrics endpoints

NEW: /metrics endpoint auto-exposed by prometheus_fastapi_instrumentator
     Prometheus scrapes it every 15s → Grafana reads from Prometheus.

Test with:
  uvicorn src.api:app --reload --port 8000
  curl http://localhost:8000/metrics   ← Prometheus metrics
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

BUG FIX (Bug 1):
  PredictionResponse and HealthResponse were missing
  model_config = {"protected_namespaces": ()}
  This caused Pydantic v2 UserWarning on every API startup because
  fields named model_name / model_uri / model_alias start with "model_",
  which Pydantic v2 reserves as a protected namespace by default.
  Fix: add model_config to both response models (same as IrisFeatures).
"""

from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration — reads from your .env ──────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "iris_classifier")
MODEL_ALIAS         = os.getenv("MODEL_ALIAS", "champion")
CLASS_NAMES         = ["setosa", "versicolor", "virginica"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded: {model_uri}")
    except Exception as e:
        logger.error(f"Could not load model '{model_uri}': {e}")
        model = None
    yield
    logger.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Iris Classifier API",
    description = "Production ML model — FastAPI + MLflow + Prometheus",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── Prometheus instrumentation ────────────────────────────────────────────────
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    inprogress_labels=True,
).instrument(app).expose(app, include_in_schema=True, tags=["monitoring"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width:  float = Field(..., gt=0, description="Petal width in cm")

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width":  3.5,
                "petal_length": 1.4,
                "petal_width":  0.2,
            }
        }
    }


# ── BUG FIX: added model_config to suppress Pydantic v2 protected namespace
#    warning for fields starting with "model_" (model_name, model_alias, model_uri)
class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # ← FIX

    prediction:  int
    class_name:  str
    confidence:  float
    model_name:  str
    model_alias: str
    timestamp:   str


# ── BUG FIX: added model_config to suppress Pydantic v2 protected namespace
#    warning for model_uri field
class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}   # ← FIX

    status:       str
    model_loaded: bool
    model_uri:    str
    timestamp:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health_check():
    """Kubernetes liveness probe + load balancer health check."""
    return HealthResponse(
        status       = "healthy" if model else "model_not_loaded",
        model_loaded = model is not None,
        model_uri    = f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        timestamp    = datetime.now().isoformat(),
    )


@app.get("/model-info", tags=["model"])
def model_info():
    """Returns which model version and alias is currently serving."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_name"   : MODEL_NAME,
        "model_alias"  : MODEL_ALIAS,
        "model_uri"    : f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        "classes"      : CLASS_NAMES,
        "n_features"   : 4,
        "feature_names": [
            "sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)",
        ],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: IrisFeatures):
    """
    Returns predicted iris class.
    503 if model not loaded. 422 if input invalid. 500 for unexpected errors.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{MODEL_NAME}@{MODEL_ALIAS}' not available. "
                   "Run model_registry.py first.",
        )
    try:
        data = pd.DataFrame(
            [[features.sepal_length, features.sepal_width,
              features.petal_length, features.petal_width]],
            columns=["sepal length (cm)", "sepal width (cm)",
                     "petal length (cm)", "petal width (cm)"],
        )
        prediction = int(model.predict(data)[0])
        proba      = model.predict_proba(data)[0]
        confidence = round(float(proba.max()), 4)

        return PredictionResponse(
            prediction  = prediction,
            class_name  = CLASS_NAMES[prediction],
            confidence  = confidence,
            model_name  = MODEL_NAME,
            model_alias = MODEL_ALIAS,
            timestamp   = datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))