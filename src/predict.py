"""
Inference / Prediction Script

WHY  : Provides a CLI and importable interface to make predictions
       using the registered champion model — no run ID needed.
HOW  : Loads model from MLflow Registry by alias (champion)
       Falls back to latest run if registry is unavailable.
WHEN : Batch scoring, ad-hoc predictions, API fallback, debugging
WHERE: Any environment that can reach the MLflow server
WHAT : Accepts a pandas DataFrame, returns class predictions + names

Usage:
  python src/predict.py                    # runs built-in sample inputs
  python src/predict.py --alias champion   # load specific alias
  python src/predict.py --alias challenger # A/B comparison
"""

import argparse
import logging
import os

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "iris_classifier")
MODEL_ALIAS         = os.getenv("MODEL_ALIAS", "champion")   # matches api.py
CLASS_NAMES         = ["setosa", "versicolor", "virginica"]
FEATURE_COLUMNS     = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_by_alias(alias: str = MODEL_ALIAS):
    """
    Load model from the Model Registry using an alias.
    WHY alias not stage:
      MLflow 3.x deprecated Production/Staging stages.
      Aliases are flexible — champion, challenger, shadow can
      all coexist pointing to different versions.

    Args:
        alias: Registry alias to load (default: champion)

    Returns:
        Loaded sklearn model object
    """
    model_uri = f"models:/{MODEL_NAME}@{alias}"
    logger.info(f"Loading model: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Model loaded successfully from {model_uri}")
    return model


def load_model_fallback():
    """
    Fallback loader — uses the most recent run if registry alias
    is unavailable (e.g. no model registered yet).

    WHY fallback exists:
      During initial setup, train.py runs before model_registry.py.
      predict.py should still work for a quick sanity check.
    """
    logger.warning(
        "Falling back to latest run — no alias available. "
        "Run model_registry.py to register a champion."
    )
    experiment = mlflow.get_experiment_by_name("iris_classification")
    if experiment is None:
        raise ValueError(
            "Experiment 'iris_classification' not found. Run train.py first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError("No runs found. Run train.py first.")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Loading fallback model from run: {run_id}")
    return mlflow.sklearn.load_model(model_uri)


def load_production_model(alias: str = MODEL_ALIAS):
    """
    Load champion model with automatic fallback to latest run.

    Args:
        alias: Registry alias to load (default from env MODEL_ALIAS)

    Returns:
        Loaded sklearn model object
    """
    try:
        return load_model_by_alias(alias)
    except mlflow.exceptions.MlflowException as e:
        logger.warning(f"Could not load alias '{alias}': {e}")
        return load_model_fallback()


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, data: pd.DataFrame) -> dict:
    """
    Run inference on a DataFrame of feature rows.

    Args:
        model : Loaded sklearn model (from load_production_model)
        data  : DataFrame with exact column names matching training data

    Returns:
        dict with:
          predictions  — list of class indices (int)
          class_names  — list of human-readable class labels
          confidence   — list of max probabilities per sample
          probabilities — full probability matrix (all classes per sample)
    """
    if not all(col in data.columns for col in FEATURE_COLUMNS):
        missing = set(FEATURE_COLUMNS) - set(data.columns)
        raise ValueError(f"Input DataFrame is missing columns: {missing}")

    # Ensure column order matches training
    data = data[FEATURE_COLUMNS]

    preds  = model.predict(data)
    probas = model.predict_proba(data)

    return {
        "predictions"  : preds.tolist(),
        "class_names"  : [CLASS_NAMES[int(p)] for p in preds],
        "confidence"   : probas.max(axis=1).round(4).tolist(),
        "probabilities": {
            sample_idx: {
                CLASS_NAMES[cls_idx]: round(float(prob), 4)
                for cls_idx, prob in enumerate(row)
            }
            for sample_idx, row in enumerate(probas)
        },
    }


def predict_single(
    model,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
) -> dict:
    """
    Convenience wrapper for a single sample prediction.
    Used by the API and direct scripting.

    Returns:
        dict with prediction, class_name, confidence, all_probabilities
    """
    data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURE_COLUMNS,
    )
    result = predict(model, data)
    return {
        "prediction"      : result["predictions"][0],
        "class_name"      : result["class_names"][0],
        "confidence"      : result["confidence"][0],
        "all_probabilities": result["probabilities"][0],
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _print_results(results: dict, samples: pd.DataFrame):
    print("\n" + "=" * 60)
    print(f"  PREDICTIONS  (model: {MODEL_NAME}@{MODEL_ALIAS})")
    print("=" * 60)
    for i, (cls, conf) in enumerate(
        zip(results["class_names"], results["confidence"])
    ):
        row = samples.iloc[i].tolist()
        print(
            f"  Sample {i + 1}: {cls:<12} "
            f"(confidence: {conf:.2%})  "
            f"inputs: {[round(v, 1) for v in row]}"
        )
        probs = results["probabilities"][i]
        for cls_name, prob in probs.items():
            bar = "█" * int(prob * 20)
            print(f"           {cls_name:<12} {prob:.4f}  {bar}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run predictions using a registered MLflow model alias"
    )
    parser.add_argument(
        "--alias",
        type=str,
        default=MODEL_ALIAS,
        help=f"Model alias to load (default: {MODEL_ALIAS})",
    )
    args = parser.parse_args()

    model = load_production_model(alias=args.alias)

    # ── Sample inputs covering all 3 classes ─────────────────────────────────
    samples = pd.DataFrame(
        [
            [5.1, 3.5, 1.4, 0.2],   # setosa
            [6.0, 2.7, 5.1, 1.6],   # versicolor
            [6.9, 3.1, 5.4, 2.1],   # virginica
            [5.8, 2.8, 5.1, 2.4],   # virginica (borderline)
            [5.7, 3.8, 1.7, 0.3],   # setosa
        ],
        columns=FEATURE_COLUMNS,
    )

    results = predict(model, samples)
    _print_results(results, samples)