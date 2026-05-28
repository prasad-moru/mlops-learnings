"""
FastAPI Prediction Service

WHY  : Mobile apps, web apps, dashboards all need predictions via HTTP.
       A REST API is language-agnostic — any client can call it.
HOW  : FastAPI + Pydantic for request validation; model loaded at startup
WHEN : After model is promoted to Production alias in the registry
WHERE: Docker container, Kubernetes pod, cloud function
WHAT : Exposes /predict, /health, /model-info endpoints

Test with:
  uvicorn src.api:app --reload --port 8000
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
"""

from contextlib import asynccontextmanager
from datetime import datetime
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "iris_classifier")
MODEL_ALIAS         = os.getenv("MODEL_ALIAS", "champion")
CLASS_NAMES         = ["setosa", "versicolor", "virginica"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model — loaded once at startup, reused on every request
model = None


# ── Lifespan: load model on startup, clean up on shutdown ─────────────────────
# WHY lifespan instead of @app.on_event("startup"):
#   on_event is deprecated since FastAPI 0.93.
#   lifespan is the modern, recommended approach.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Model loaded: {model_uri}")
    except Exception as e:
        logger.error(f"Could not load model '{model_uri}': {e}")
        logger.warning("API will start but /predict will return 503 until model is available.")
        model = None
    yield
    # Shutdown: nothing to clean up for sklearn models
    logger.info("API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Iris Classifier API",
    description = "Production ML model served via FastAPI + MLflow",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Request / Response Schemas ────────────────────────────────────────────────
# WHY Pydantic v2 style:
#   schema_extra was renamed to json_schema_extra in Pydantic v2.
#   class Config is replaced by model_config dict.
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width:  float = Field(..., gt=0, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width":  3.5,
                "petal_length": 1.4,
                "petal_width":  0.2,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction:  int
    class_name:  str
    confidence:  float
    model_name:  str
    model_alias: str
    timestamp:   str


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    model_uri:    str
    timestamp:    str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    WHY  : Load balancers and Kubernetes liveness probes call /health every 30s.
           Returns 200 if API is up regardless of model state.
    """
    return HealthResponse(
        status       = "healthy" if model else "model_not_loaded",
        model_loaded = model is not None,
        model_uri    = f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        timestamp    = datetime.now().isoformat(),
    )


@app.get("/model-info")
def model_info():
    """
    WHY  : Clients need to know which model version and alias is live.
    WHAT : Returns model name, alias, feature names, and output classes.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name"   : MODEL_NAME,
        "model_alias"  : MODEL_ALIAS,
        "model_uri"    : f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
        "classes"      : CLASS_NAMES,
        "n_features"   : 4,
        "feature_names": [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    """
    WHY  : Core endpoint — returns predicted iris class for input measurements.
    HOW  : Pydantic validates input → pandas DataFrame → model.predict()
    WHEN : Called by any client application needing a classification.
    NOTE : Returns 503 if model failed to load at startup.
           Returns 422 if input fails Pydantic validation (missing field, negative value).
           Returns 500 for unexpected prediction errors.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{MODEL_NAME}@{MODEL_ALIAS}' is not available. "
                   "Register and alias a model first.",
        )

    try:
        # Build DataFrame with exact column names the model was trained on
        data = pd.DataFrame(
            [[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]],
            columns=[
                "sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)",
            ],
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
