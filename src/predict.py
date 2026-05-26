"""
Inference / Prediction Script

WHY  : Provides a simple interface to make predictions using the
       registered Production model — no need to know the run ID.
HOW  : Loads model from MLflow Registry by stage name ("Production")
WHEN : Batch scoring, ad-hoc predictions, API fallback, testing
WHERE: Any environment that can reach the MLflow server
WHAT : Accepts a pandas DataFrame, returns class predictions + names
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

MODEL_NAME  = os.getenv('MODEL_NAME', 'iris_classifier')
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']


def get_latest_run_id(experiment_name="iris_classification"):
    """Fall-back: get the most recent run ID if registry is not available."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs.empty:
        raise ValueError("No runs found. Run train.py first.")

    return runs.iloc[0]['run_id']


def load_production_model():
    """Load model from the Production stage of the Model Registry."""
    try:
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded Production model: {MODEL_NAME}")
        return model
    except Exception as e:
        logger.warning(f"Could not load from registry ({e}). Falling back to latest run.")
        run_id = get_latest_run_id()
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        logger.info(f"Loaded model from run {run_id}")
        return model


def predict(model, data: pd.DataFrame):
    """
    Make predictions on input DataFrame.

    Returns dict with:
        predictions  : list of class indices
        class_names  : list of human-readable class names
        confidence   : list of max probabilities
    """
    preds = model.predict(data)
    proba = model.predict_proba(data)

    return {
        'predictions': preds.tolist(),
        'class_names': [CLASS_NAMES[p] for p in preds],
        'confidence' : proba.max(axis=1).round(4).tolist()
    }


if __name__ == "__main__":
    model = load_production_model()

    # Sample inputs (sepal_length, sepal_width, petal_length, petal_width)
    samples = pd.DataFrame([
        [5.1, 3.5, 1.4, 0.2],   # setosa
        [6.0, 2.7, 5.1, 1.6],   # versicolor
        [6.9, 3.1, 5.4, 2.1],   # virginica
    ], columns=['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)'])

    results = predict(model, samples)

    print("\n=== PREDICTIONS ===")
    for i, (cls, conf) in enumerate(zip(results['class_names'], results['confidence'])):
        print(f"  Sample {i+1}: {cls}  (confidence: {conf:.2%})")
