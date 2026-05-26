"""
Model Registry — Register, Promote & Rollback Models

WHY  : Without a registry you have loose .pkl files with no lifecycle
       management, no stage tracking, and no easy rollback.
HOW  : MLflow Model Registry stores versions with stages:
         None → Staging → Production → Archived
WHEN : After hyperparameter tuning; before API deployment
WHERE: CI/CD pipeline (auto) or manually by ML engineer
WHAT : Finds the best-performing run, registers the model,
       promotes it to Production, archives the old version

Model Stage Flow:
  New run ──► Register ──► Staging (QA testing) ──► Production ──► Archived
                                                         ↑
                                              Only this version serves traffic
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))

MODEL_NAME       = os.getenv('MODEL_NAME', 'iris_classifier')
EXPERIMENT_NAME  = 'iris_classification'
MIN_ACCURACY     = float(os.getenv('MIN_ACCURACY', '0.90'))   # gate: won't promote below this


def get_best_run():
    """Find the run with the highest accuracy in the experiment."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.accuracy > 0",
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError("No runs found. Run train.py first.")

    best = runs.iloc[0]
    logger.info(f"Best run  → ID: {best.run_id}  accuracy: {best['metrics.accuracy']:.4f}")
    return best


def register_model(run_id, accuracy):
    """Register model and return the new ModelVersion object."""
    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Accuracy {accuracy:.4f} is below minimum threshold {MIN_ACCURACY}. "
            "Model will NOT be registered."
        )

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from run {run_id} …")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        tags={"registered_from_run": run_id, "accuracy": str(round(accuracy, 4))}
    )

    logger.info(f"Registered  → {MODEL_NAME} version {model_version.version}")
    return model_version


def promote_to_production(client, version):
    """Move the given version to Production; archive any existing Production version."""

    # Archive current Production version (if any)
    current_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    for v in current_prod:
        logger.info(f"Archiving old Production version {v.version}")
        client.transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Archived",
            archive_existing_versions=False
        )

    # Promote new version
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Production"
    )
    logger.info(f"✅  {MODEL_NAME} v{version} is now in Production")


def register_best_model():
    """Full workflow: find best run → register → promote → summary."""
    logger.info("=" * 70)
    logger.info("MODEL REGISTRY WORKFLOW")
    logger.info("=" * 70)

    client   = MlflowClient()
    best_run = get_best_run()
    accuracy = best_run['metrics.accuracy']

    # Register
    model_version = register_model(best_run.run_id, accuracy)

    # Promote
    promote_to_production(client, model_version.version)

    # Show all versions
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    print("\n" + "=" * 70)
    print(f"📦  {MODEL_NAME} — All Versions")
    print("=" * 70)
    for v in sorted(all_versions, key=lambda x: int(x.version)):
        print(f"  v{v.version}  stage={v.current_stage:<12}  run={v.run_id[:8]}…")
    print("=" * 70)

    return model_version


if __name__ == "__main__":
    register_best_model()
