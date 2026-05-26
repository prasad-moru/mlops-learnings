"""
Multi-Experiment Runner

WHY  : Systematically compare multiple hyperparameter configurations
HOW  : Loops over param combos, logs each as a separate MLflow run
WHEN : Initial model development or periodic re-evaluation
WHERE: Dev laptop, CI/CD, cloud VM
WHAT : Trains N models, logs all results, prints ranked summary
"""

import mlflow
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from train import load_data, train_model, evaluate_model
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment("iris_classification")


def run_experiment():
    logger.info("=" * 70)
    logger.info("MULTI-EXPERIMENT RUN")
    logger.info("=" * 70)

    X_train, X_test, y_train, y_test, _, feature_names, target_names = load_data()

    # ── Configurations to test ────────────────────────────────────────────────
    experiments = [
        {'n_estimators':  50, 'max_depth':  3, 'name': 'shallow_small'},
        {'n_estimators': 100, 'max_depth':  5, 'name': 'baseline'},
        {'n_estimators': 150, 'max_depth':  7, 'name': 'medium'},
        {'n_estimators': 200, 'max_depth': 10, 'name': 'deep_large'},
        {'n_estimators': 300, 'max_depth': None, 'name': 'unlimited_depth'},
    ]

    results = []

    for i, cfg in enumerate(experiments, 1):
        run_name = f"rf_n{cfg['n_estimators']}_d{cfg['max_depth']}_{cfg['name']}"
        logger.info(f"\n[{i}/{len(experiments)}] {run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                'n_estimators'  : cfg['n_estimators'],
                'max_depth'     : str(cfg['max_depth']),
                'config_name'   : cfg['name'],
                'experiment_type': 'grid_comparison'
            })
            mlflow.set_tags({
                "batch"  : datetime.now().strftime('%Y%m%d'),
                "purpose": "hyperparameter_comparison"
            })

            model = train_model(X_train, y_train,
                                n_estimators=cfg['n_estimators'],
                                max_depth=cfg['max_depth'])

            metrics, _, _, _ = evaluate_model(model, X_test, y_test, target_names)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            results.append({**cfg, **metrics})
            logger.info(f"  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
    print("\n" + "=" * 70)
    print("📊  EXPERIMENT SUMMARY (ranked by accuracy)")
    print("=" * 70)
    print(df[['name', 'n_estimators', 'max_depth', 'accuracy', 'f1_score']].to_string(index=False))
    best = df.iloc[0]
    print(f"\n🏆 Best: {best['name']}  accuracy={best['accuracy']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
