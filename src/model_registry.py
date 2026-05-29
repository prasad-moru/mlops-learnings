"""
Model Registry — Register, Alias & Rollback Models

ROOT CAUSE FIX:
  The previous approach called mlflow.register_model() to "test" if a run
  had a model artifact. This created orphaned version entries in the registry
  for every failed attempt (visible as "Creating a new version..." messages).

  The actual search problem: both experiments are searched together with
  max_results=20. iris_classification_tuning has 50+ trial runs (none with
  models) all with the same accuracy=0.9667. These fill all 20 slots.
  iris_classification runs (which have models) are never reached.

  Fix — two changes:
  1. Use mlflow.artifacts.list_artifacts() to check artifact existence
     BEFORE attempting registration. Zero orphaned versions created.
  2. Search each experiment separately with max_results=50 per experiment,
     then combine and sort. iris_classification runs (6 total) are always
     included regardless of how many tuning trial runs exist.
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
import mlflow
import mlflow.artifacts
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


# ── Artifact check — before registering ──────────────────────────────────────
def _model_artifact_exists(run_id: str) -> bool:
    """
    Check if a model artifact exists in this run WITHOUT registering.

    Uses mlflow.artifacts.list_artifacts() which queries the artifact store
    directly using the run URI. Returns True if the model/ directory has files.

    WHY this instead of mlflow.register_model() for checking:
      register_model() creates a DB entry before copying files. When the
      copy fails, the DB entry may persist as an orphaned "failed" version.
      list_artifacts() only reads — no side effects, no orphaned versions.
    """
    try:
        artifacts = mlflow.artifacts.list_artifacts(f"runs:/{run_id}/model")
        return len(artifacts) > 0
    except Exception as e:
        logger.debug(f"No model artifact in run {run_id[:8]}: {e}")
        return False


# ── Step 1: Find the best run that has a model artifact ───────────────────────
def get_best_run():
    """
    Search each experiment independently with max_results=50 per experiment.
    Combine results, sort by accuracy, return the best run that has a
    logged model artifact.

    WHY per-experiment search instead of combined:
      iris_classification_tuning has 50+ trial runs (no models) all with
      accuracy=0.9667. A combined search with max_results=20 fills all slots
      with tuning trial runs, never reaching iris_classification runs.
      Searching per-experiment (50 each) guarantees both are represented.
    """
    all_runs = []

    for exp_name in SEARCH_EXPERIMENTS:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            logger.warning(f"Experiment '{exp_name}' not found — skipping.")
            continue

        logger.info(f"Searching '{exp_name}' (id={exp.experiment_id})")

        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.accuracy > 0",
            order_by=["metrics.accuracy DESC"],
            max_results=50,
        )

        if not runs.empty:
            logger.info(f"  Found {len(runs)} runs in '{exp_name}'")
            all_runs.append(runs)

    if not all_runs:
        raise ValueError(
            f"No runs found across experiments: {SEARCH_EXPERIMENTS}. "
            "Run train.py first."
        )

    import pandas as pd
    combined = pd.concat(all_runs, ignore_index=True)
    combined = combined.sort_values("metrics.accuracy", ascending=False)

    logger.info(
        f"Checking {len(combined)} total runs for model artifacts "
        f"(from both experiments) ..."
    )

    for _, candidate in combined.iterrows():
        run_id   = candidate.run_id
        accuracy = candidate["metrics.accuracy"]
        exp_id   = candidate["experiment_id"]

        if _model_artifact_exists(run_id):
            logger.info(
                f"Best run with model artifact → "
                f"ID: {run_id}  accuracy: {accuracy:.4f}  experiment: {exp_id}"
            )
            return candidate
        else:
            logger.debug(
                f"Run {run_id[:8]} (exp={exp_id}) has no model — skipping"
            )

    raise ValueError(
        "No runs with a logged model artifact found.\n"
        "All candidate runs exist but none called mlflow.sklearn.log_model().\n"
        "Run: python src/train.py — then try again."
    )


# ── Step 2: Register the model ────────────────────────────────────────────────
def register_model(run_id: str, accuracy: float):
    """Register model from a run that is confirmed to have an artifact."""
    if accuracy < MIN_ACCURACY:
        raise ValueError(
            f"Accuracy {accuracy:.4f} is below the minimum threshold "
            f"{MIN_ACCURACY}. Model will NOT be registered."
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

    client       = MlflowClient()
    best_run     = get_best_run()
    accuracy     = best_run["metrics.accuracy"]

    model_version = register_model(best_run.run_id, accuracy)
    promote_to_champion(client, model_version.version, accuracy)

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
