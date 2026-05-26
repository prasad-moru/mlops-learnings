"""
Automated Hyperparameter Tuning with Optuna + MLflow

WHY  : Manual grid-search wastes time on bad regions of the search space.
       Optuna uses Bayesian optimisation — it LEARNS from previous trials
       and focuses on promising parameter combinations.
HOW  : Optuna suggests params → train → evaluate → Optuna updates belief
WHEN : After initial experiments; before production deployment
WHERE: High-compute env (cloud VM / local GPU)
WHAT : Finds optimal hyperparameters and logs every trial to MLflow

Comparison:
  Grid Search  → tries ALL combos  (4×4 = 16 for 2 params)
  Random Search→ tries random combos (unintelligent)
  Optuna       → learns from results (intelligent, 50 trials ≈ grid of 1000)
"""

import mlflow
import optuna
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from train import load_data, train_model, evaluate_model
from dotenv import load_dotenv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)   # keep output clean

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment("iris_classification_tuning")

# Load data once — reuse across all trials (saves time)
X_train, X_test, y_train, y_test, _, feature_names, target_names = load_data()


def objective(trial):
    """
    Optuna calls this function for each trial.
    It suggests hyperparameters, trains a model, returns the score.
    """
    # ── Search space ─────────────────────────────────────────────────────────
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 50, 300),
        'max_depth'       : trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 5),
    }

    # ── MLflow nested run for each trial ─────────────────────────────────────
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number)

        model = train_model(X_train, y_train, **params)
        metrics, _, _, _ = evaluate_model(model, X_test, y_test, target_names)
        mlflow.log_metrics(metrics)

    return metrics['accuracy']


def run_tuning(n_trials=50):
    logger.info("=" * 70)
    logger.info(f"HYPERPARAMETER TUNING  ({n_trials} trials)")
    logger.info("=" * 70)

    # Parent MLflow run wraps all trials
    with mlflow.start_run(run_name=f"optuna_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best_val = study.best_value

        # Log best result to parent run
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_accuracy", best_val)
        mlflow.set_tag("purpose", "hyperparameter_tuning")

        print("\n" + "=" * 70)
        print("✅  TUNING COMPLETE")
        print("=" * 70)
        print(f"🏆 Best accuracy : {best_val:.4f}")
        print(f"🔧 Best params   :")
        for k, v in best.items():
            print(f"   ├── {k}: {v}")
        print("=" * 70)

        return best, best_val


if __name__ == "__main__":
    run_tuning(n_trials=50)
