"""
Automated Hyperparameter Tuning with Optuna + MLflow

WHY  : Manual grid-search wastes time on bad regions of the search space.
       Optuna uses Bayesian optimisation — it LEARNS from previous trials
       and focuses on promising parameter combinations.
HOW  : Optuna suggests params → train → evaluate → Optuna updates belief
WHEN : After initial experiments; before production deployment
WHERE: High-compute env (cloud VM / local GPU)
WHAT : Finds optimal hyperparameters and logs every trial to MLflow

BUG FIX (Bug 4 + new bug):
  Two problems existed:
  1. log_train_test_datasets() was never called — no data lineage.
  2. The objective() function logged metrics per trial but never called
     mlflow.sklearn.log_model(). So model_registry.py found a run with
     great accuracy but no model artifact to register from.

  Fix:
  - log_train_test_datasets() is called once on the parent run (all trials
    share the same data split, so one log on the parent is correct).
  - After study.optimize() completes, the best params are used to retrain
    one final model which is logged to the parent run. This gives
    model_registry.py a real model artifact to register from.
  - Individual trial runs still only log metrics (no model) to save disk.
"""

import mlflow
import mlflow.sklearn
import optuna
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from train import load_data, train_model, evaluate_model
from data_versioning import log_train_test_datasets
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment("iris_classification_tuning")

# Load data once — reuse across all trials (saves time)
X_train, X_test, y_train, y_test, _, feature_names, target_names = load_data()


def objective(trial):
    """
    Optuna calls this function for each trial.
    Logs params + metrics only — model artifact is NOT logged here.
    The best model is logged once on the parent run after study completes.
    """
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 50, 300),
        'max_depth'       : trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 5),
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        model = train_model(X_train, y_train, **params)
        metrics, _, _, _ = evaluate_model(model, X_test, y_test, target_names)
        mlflow.log_metrics(metrics)
        # NOTE: intentionally no log_model() here — saves disk space.
        # 50 model files × ~1MB each = 50MB wasted for intermediate results.
        # Only the winning model is saved, on the parent run below.

    return metrics['accuracy']


def run_tuning(n_trials=50):
    logger.info("=" * 70)
    logger.info(f"HYPERPARAMETER TUNING  ({n_trials} trials)")
    logger.info("=" * 70)

    with mlflow.start_run(run_name=f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Log dataset lineage once on the parent run.
        # All 50 trials use the same X_train/X_test — one entry is correct.
        log_train_test_datasets(X_train, X_test, y_train, y_test)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best     = study.best_params
        best_val = study.best_value

        # Log best hyperparameters and accuracy to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_accuracy", best_val)
        # Also log as plain 'accuracy' so model_registry.py can find this
        # run with filter_string="metrics.accuracy > 0"
        mlflow.log_metric("accuracy", best_val)
        mlflow.set_tag("purpose", "hyperparameter_tuning")

        # ── FIX: retrain best model and log it to the parent run ──────────────
        # model_registry.py looks for runs:/{run_id}/model
        # Without this, the parent run has no model artifact and registration fails.
        # We retrain with best params (takes ~1 second for Iris) and log it once.
        logger.info(f"Retraining with best params for model logging: {best}")
        best_model = train_model(X_train, y_train, **best)
        signature  = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(
            best_model,
            "model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )
        logger.info("Best model logged to parent run — ready for model_registry.py")
        # ─────────────────────────────────────────────────────────────────────

        print("\n" + "=" * 70)
        print("TUNING COMPLETE")
        print("=" * 70)
        print(f"Best accuracy : {best_val:.4f}")
        print(f"Best params   :")
        for k, v in best.items():
            print(f"   -- {k}: {v}")
        print("=" * 70)

        return best, best_val


if __name__ == "__main__":
    run_tuning(n_trials=50)