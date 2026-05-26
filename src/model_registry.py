"""
Model Registry — Register, Alias & Rollback Models

WHY  : Without a registry you have loose .pkl files with no lifecycle
       management, no versioning, and no easy rollback.
HOW  : MLflow Model Registry + aliases (MLflow 3.x recommended pattern):
         Register → set alias "champion" → API loads by alias
WHEN : After training / hyperparameter tuning; before API deployment
WHERE: CI/CD pipeline (auto) or manually by ML engineer
WHAT : Finds the best-performing run, registers the model,
       sets it as "champion", moves old champion to "archived" tag.

WHY ALIASES instead of stages (Staging/Production/Archived):
  MLflow 3.x deprecated transition_model_version_stage() and
  get_latest_versions(stages=[...]).
  Aliases are more flexible — you can have "champion", "challenger",
  "shadow" all pointing to different versions simultaneously.
  The API loads: models:/iris_classifier@champion

Alias Flow:
  New run ──► Register ──► set alias "champion"
                                   ↑
                    old champion alias is removed + version tagged "archived"
                    API immediately serves new champion on next request
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

MODEL_NAME      = os.getenv("MODEL_NAME", "iris_classifier")
EXPERIMENT_NAME = "iris_classification"
MIN_ACCURACY    = float(os.getenv("MIN_ACCURACY", "0.90"))
CHAMPION_ALIAS  = "champion"
CHALLENGER_ALIAS = "challenger"


# ── Step 1: Find the best run ─────────────────────────────────────────────────
def get_best_run():
    """
    Find the run with the highest accuracy in the experiment.
    WHY : We always promote the objectively best model, not the latest.
    """
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.accuracy > 0",
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if runs.empty:
        raise ValueError("No runs found in experiment. Run train.py first.")

    best = runs.iloc[0]
    logger.info(
        f"Best run → ID: {best.run_id}  accuracy: {best['metrics.accuracy']:.4f}"
    )
    return best


# ── Step 2: Register the model ────────────────────────────────────────────────
def register_model(run_id: str, accuracy: float):
    """
    Register the model artifact from a run into the Model Registry.
    Applies the accuracy gate — won't register below MIN_ACCURACY.

    WHY accuracy gate: Prevents a degraded model from accidentally
    entering the registry during automated retraining.
    """
    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Accuracy {accuracy:.4f} is below the minimum threshold "
            f"{MIN_ACCURACY}. Model will NOT be registered. "
            "Investigate model quality before promoting."
        )

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Registering model from run {run_id} ...")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
        tags={
            "registered_from_run": run_id,
            "accuracy": str(round(accuracy, 4)),
        },
    )

    logger.info(f"Registered → {MODEL_NAME} version {model_version.version}")
    return model_version


# ── Step 3: Set champion alias ────────────────────────────────────────────────
def promote_to_champion(client: MlflowClient, new_version: str, accuracy: float):
    """
    Set the new version as 'champion' alias.
    Archive the previous champion by removing its alias and tagging it.

    WHY aliases over stages:
      MLflow 3.x deprecated model stages (Production/Staging/Archived).
      Aliases are the new standard — flexible, multi-alias support,
      and the API loads by alias: models:/iris_classifier@champion
    """
    # Check if a champion already exists — demote it first
    try:
        old_champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        old_version  = old_champion.version

        if old_version == new_version:
            logger.info(f"Version {new_version} is already the champion. No change.")
            return

        logger.info(f"Demoting old champion v{old_version} → removing alias, tagging as archived")

        # Remove champion alias from old version
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)

        # Tag old version so we know its history
        client.set_model_version_tag(
            name    = MODEL_NAME,
            version = old_version,
            key     = "status",
            value   = "archived",
        )
        client.set_model_version_tag(
            name    = MODEL_NAME,
            version = old_version,
            key     = "archived_reason",
            value   = f"superseded by v{new_version}",
        )

    except mlflow.exceptions.MlflowException:
        # No champion alias exists yet — first time registration
        logger.info("No existing champion alias found. Setting first champion.")

    # Set new version as champion
    client.set_registered_model_alias(
        name    = MODEL_NAME,
        alias   = CHAMPION_ALIAS,
        version = new_version,
    )

    # Tag new version
    client.set_model_version_tag(
        name    = MODEL_NAME,
        version = new_version,
        key     = "status",
        value   = "champion",
    )
    client.set_model_version_tag(
        name    = MODEL_NAME,
        version = new_version,
        key     = "accuracy",
        value   = str(round(accuracy, 4)),
    )

    logger.info(
        f"✅  {MODEL_NAME} v{new_version} is now '{CHAMPION_ALIAS}' "
        f"(accuracy={accuracy:.4f})"
    )


# ── Optional: set challenger alias for A/B testing ────────────────────────────
def set_challenger(client: MlflowClient, version: str):
    """
    Set a version as 'challenger' alias for shadow/A-B testing.
    WHY : Allows the API to route a percentage of traffic to a new model
          before making it the champion.
    """
    try:
        client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
    except mlflow.exceptions.MlflowException:
        pass

    client.set_registered_model_alias(
        name    = MODEL_NAME,
        alias   = CHALLENGER_ALIAS,
        version = version,
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=version, key="status", value="challenger"
    )
    logger.info(f"Version {version} set as '{CHALLENGER_ALIAS}'")


# ── Rollback: restore a previous version as champion ─────────────────────────
def rollback_champion(client: MlflowClient, version: str):
    """
    Roll back the champion alias to a specific previous version.
    WHY : If a newly promoted champion degrades in production,
          one call restores the previous version instantly.
          The API picks it up on the next model reload.
    """
    logger.info(f"Rolling back champion to version {version} ...")
    current = client.get_model_version(MODEL_NAME, version)

    try:
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        pass

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    client.set_model_version_tag(
        name=MODEL_NAME, version=version, key="status", value="champion_rollback"
    )
    logger.info(f"✅  Rolled back — {MODEL_NAME} v{version} is now champion")


# ── Full pipeline ──────────────────────────────────────────────────────────────
def register_best_model():
    """
    End-to-end workflow:
      1. Find best run by accuracy
      2. Apply accuracy gate
      3. Register to Model Registry
      4. Set as champion alias
      5. Print version summary
    """
    logger.info("=" * 70)
    logger.info("MODEL REGISTRY WORKFLOW")
    logger.info("=" * 70)

    client   = MlflowClient()
    best_run = get_best_run()
    accuracy = best_run["metrics.accuracy"]

    # Register
    model_version = register_model(best_run.run_id, accuracy)

    # Promote via alias
    promote_to_champion(client, model_version.version, accuracy)

    # Show all versions summary
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    print("\n" + "=" * 70)
    print(f"  {MODEL_NAME} — all versions")
    print("=" * 70)
    print(f"  {'ver':<5} {'status tag':<20} {'run id':<12} {'accuracy'}")
    print("  " + "-" * 60)
    for v in sorted(all_versions, key=lambda x: int(x.version)):
        status   = v.tags.get("status", "—")
        accuracy_tag = v.tags.get("accuracy", "—")
        alias_str = ""
        if v.aliases:
            alias_str = f"  [{', '.join(v.aliases)}]"
        print(
            f"  v{v.version:<4} {status:<20} {v.run_id[:8]}...  "
            f"{accuracy_tag}{alias_str}"
        )
    print("=" * 70)
    print(f"\n  Load in API / predict.py:")
    print(f"  mlflow.sklearn.load_model('models:/{MODEL_NAME}@{CHAMPION_ALIAS}')")
    print()

    return model_version


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLflow Model Registry operations")
    parser.add_argument(
        "--rollback",
        type=str,
        default=None,
        metavar="VERSION",
        help="Rollback champion to a specific version number (e.g. --rollback 2)",
    )
    parser.add_argument(
        "--challenger",
        type=str,
        default=None,
        metavar="VERSION",
        help="Set a version as challenger alias (e.g. --challenger 3)",
    )
    args = parser.parse_args()

    client = MlflowClient()

    if args.rollback:
        rollback_champion(client, args.rollback)
    elif args.challenger:
        set_challenger(client, args.challenger)
    else:
        register_best_model()
