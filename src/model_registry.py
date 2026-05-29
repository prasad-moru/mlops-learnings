"""
Model Registry — Register, Alias & Rollback Models

BUG FIX:
  Replaced the unreliable artifact-path check with a direct try/except
  approach. Instead of guessing what list_artifacts() returns (which varies
  by MLflow version and artifact store backend), we simply attempt to
  register each run and skip it if no model artifact exists.
  This works correctly regardless of MLflow version or storage backend.
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

MODEL_NAME       = os.getenv("MODEL_NAME", "iris_classifier")
MIN_ACCURACY     = float(os.getenv("MIN_ACCURACY", "0.90"))
CHAMPION_ALIAS   = "champion"
CHALLENGER_ALIAS = "challenger"

SEARCH_EXPERIMENTS = [
    "iris_classification",
    "iris_classification_tuning",
]


# ── Step 1: Find best run ─────────────────────────────────────────────────────
def get_candidate_runs(max_results: int = 20):
    """
    Return top runs by accuracy across all configured experiments.
    Does NOT filter by artifact existence — that is handled at registration.
    """
    client = MlflowClient()
    experiment_ids = []

    for exp_name in SEARCH_EXPERIMENTS:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is not None:
            experiment_ids.append(exp.experiment_id)
            logger.info(f"Searching experiment: '{exp_name}' (id={exp.experiment_id})")
        else:
            logger.warning(f"Experiment '{exp_name}' not found — skipping.")

    if not experiment_ids:
        raise ValueError(
            f"None of the configured experiments found: {SEARCH_EXPERIMENTS}. "
            "Run train.py first."
        )

    runs = mlflow.search_runs(
        experiment_ids=experiment_ids,
        filter_string="metrics.accuracy > 0",
        order_by=["metrics.accuracy DESC"],
        max_results=max_results,
    )

    if runs.empty:
        raise ValueError("No runs with accuracy metric found. Run train.py first.")

    return runs


# ── Step 2: Register model ────────────────────────────────────────────────────
def try_register_model(run_id: str, accuracy: float):
    """
    Attempt to register a model from the given run.
    Returns the ModelVersion on success.
    Returns None if this run has no model artifact (no log_model() was called).
    Raises ValueError if accuracy is below the gate.
    Raises MlflowException for any other registry error.
    """
    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Accuracy {accuracy:.4f} is below the minimum threshold "
            f"{MIN_ACCURACY}. Model will NOT be registered."
        )

    model_uri = f"runs:/{run_id}/model"

    try:
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

    except mlflow.exceptions.MlflowException as e:
        # This specific error means the run exists but has no model artifact.
        # Skip to the next candidate run.
        if "Unable to find a logged_model" in str(e) or "No such file" in str(e):
            logger.debug(
                f"Run {run_id[:8]} has no model artifact — skipping. "
                f"(accuracy={accuracy:.4f})"
            )
            return None
        # Any other MLflow error is unexpected — re-raise it.
        raise


# ── Step 3: Promote to champion ───────────────────────────────────────────────
def promote_to_champion(client: MlflowClient, new_version: str, accuracy: float):
    try:
        old_champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        old_version  = old_champion.version

        if old_version == new_version:
            logger.info(f"Version {new_version} is already the champion. No change.")
            return

        logger.info(f"Demoting old champion v{old_version} → archiving")
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
        client.set_model_version_tag(
            name=MODEL_NAME, version=old_version, key="status", value="archived"
        )
        client.set_model_version_tag(
            name=MODEL_NAME, version=old_version,
            key="archived_reason", value=f"superseded by v{new_version}"
        )

    except mlflow.exceptions.MlflowException:
        logger.info("No existing champion alias found. Setting first champion.")

    client.set_registered_model_alias(
        name=MODEL_NAME, alias=CHAMPION_ALIAS, version=new_version
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=new_version, key="status", value="champion"
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=new_version,
        key="accuracy", value=str(round(accuracy, 4))
    )
    logger.info(
        f"  {MODEL_NAME} v{new_version} is now '{CHAMPION_ALIAS}' "
        f"(accuracy={accuracy:.4f})"
    )


# ── Optional helpers ──────────────────────────────────────────────────────────
def set_challenger(client: MlflowClient, version: str):
    try:
        client.delete_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS)
    except mlflow.exceptions.MlflowException:
        pass
    client.set_registered_model_alias(
        name=MODEL_NAME, alias=CHALLENGER_ALIAS, version=version
    )
    client.set_model_version_tag(
        name=MODEL_NAME, version=version, key="status", value="challenger"
    )
    logger.info(f"Version {version} set as '{CHALLENGER_ALIAS}'")


def rollback_champion(client: MlflowClient, version: str):
    logger.info(f"Rolling back champion to version {version} ...")
    try:
        client.delete_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS)
    except mlflow.exceptions.MlflowException:
        pass
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    client.set_model_version_tag(
        name=MODEL_NAME, version=version, key="status", value="champion_rollback"
    )
    logger.info(f"  Rolled back — {MODEL_NAME} v{version} is now champion")


# ── Full pipeline ──────────────────────────────────────────────────────────────
def register_best_model():
    logger.info("=" * 70)
    logger.info("MODEL REGISTRY WORKFLOW")
    logger.info("=" * 70)
    logger.info(f"Searching experiments: {SEARCH_EXPERIMENTS}")

    client = MlflowClient()
    runs   = get_candidate_runs(max_results=20)

    logger.info(f"Found {len(runs)} candidate runs — trying each in accuracy order ...")

    model_version = None
    for _, run in runs.iterrows():
        accuracy = run["metrics.accuracy"]
        run_id   = run.run_id

        logger.info(
            f"Trying run {run_id[:8]}  accuracy={accuracy:.4f}  "
            f"experiment_id={run['experiment_id']}"
        )

        model_version = try_register_model(run_id, accuracy)
        if model_version is not None:
            # Successfully registered — promote and stop
            promote_to_champion(client, model_version.version, accuracy)
            break

    if model_version is None:
        raise ValueError(
            "None of the top 20 runs had a logged model artifact.\n"
            "This means log_model() was never called in any training script.\n"
            "Run: python src/train.py   — and try again."
        )

    # Print version summary
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    print("\n" + "=" * 70)
    print(f"  {MODEL_NAME} — all versions")
    print("=" * 70)
    print(f"  {'ver':<5} {'status tag':<20} {'run id':<12} {'accuracy'}")
    print("  " + "-" * 60)
    for v in sorted(all_versions, key=lambda x: int(x.version)):
        status       = v.tags.get("status", "—")
        accuracy_tag = v.tags.get("accuracy", "—")
        alias_str    = f"  [{', '.join(v.aliases)}]" if v.aliases else ""
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
    parser.add_argument("--rollback",   type=str, default=None, metavar="VERSION")
    parser.add_argument("--challenger", type=str, default=None, metavar="VERSION")
    args = parser.parse_args()

    client = MlflowClient()
    if args.rollback:
        rollback_champion(client, args.rollback)
    elif args.challenger:
        set_challenger(client, args.challenger)
    else:
        register_best_model()
