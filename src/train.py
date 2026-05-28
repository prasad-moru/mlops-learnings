"""
Production-Ready Training Pipeline with MLOps Best Practices

MLOPS CONCEPTS COVERED:
=======================
1. EXPERIMENT TRACKING  - MLflow logs parameters, metrics, and models
2. REPRODUCIBILITY      - Random seeds + data versioning
3. MODEL VERSIONING     - Saved with metadata for deployment
4. ARTIFACT MANAGEMENT  - Plots, reports, feature importance
5. METADATA LOGGING     - Tags for search and governance

WHY THIS MATTERS:
=================
- Reproducibility : Anyone can recreate exact results
- Auditability    : Full history of what was tried
- Collaboration   : Team shares experiments via MLflow UI
- Production Path : Clear route from experiment → deployment
"""

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_versioning import log_train_test_datasets
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path


# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME     = os.getenv('EXPERIMENT_NAME', 'iris_classification')
RANDOM_SEED         = int(os.getenv('RANDOM_SEED', '42'))

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Artifacts directory ───────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")
logger.info(f"Experiment : {EXPERIMENT_NAME}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# WHY  : Standardised loading guarantees consistent splits across runs
# HOW  : sklearn iris → pandas DataFrame + stratified split
# WHEN : Start of every training/retraining run
# WHERE: Local dev, CI/CD, scheduled jobs
# WHAT : Returns X/y train-test splits + metadata dict
# ─────────────────────────────────────────────────────────────────────────────
def load_data(test_size=0.2, random_state=RANDOM_SEED):
    logger.info("Loading Iris dataset …")

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    data_stats = {
        'n_samples'           : len(X),
        'n_features'          : X.shape[1],
        'feature_names'       : list(X.columns),
        'target_names'        : list(iris.target_names),
        'target_distribution' : y.value_counts().to_dict(),
        'data_hash'           : str(hash(X.values.tobytes()))
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y          # keeps class balance in both splits
    )

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, data_stats, iris.feature_names, iris.target_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAIN MODEL
# WHY  : Separate function → easy swapping of algorithm
# HOW  : RandomForestClassifier with fully explicit params (no hidden defaults)
# WHEN : Called once per experiment run
# WHERE: Training server, laptop, cloud VM
# WHAT : Returns fitted model ready for evaluation
# ─────────────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train,
                n_estimators=100, max_depth=5,
                min_samples_split=2, min_samples_leaf=1,
                random_state=RANDOM_SEED):

    logger.info(f"Training RF  n_estimators={n_estimators}  max_depth={max_depth}")

    model = RandomForestClassifier(
        n_estimators    = n_estimators,
        max_depth       = max_depth,
        min_samples_split = min_samples_split,
        min_samples_leaf  = min_samples_leaf,
        random_state    = random_state,
        n_jobs          = -1          # use all CPU cores
    )
    model.fit(X_train, y_train)
    logger.info("Training complete ✓")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. EVALUATE MODEL
# WHY  : One metric is never enough; full picture needed before production
# HOW  : accuracy, precision, recall, F1, ROC-AUC + per-class report
# WHEN : After every training run, and periodically in production
# WHERE: Training pipeline + monitoring pipeline
# WHAT : Returns metrics dict, predictions, probabilities, classification report
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, target_names=None):
    logger.info("Evaluating model …")

    y_pred       = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = {
        'accuracy' : accuracy_score(y_test, y_pred),
        'f1_score' : f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall'   : recall_score(y_test, y_pred, average='weighted'),
    }

    try:
        metrics['roc_auc'] = roc_auc_score(
            y_test, y_pred_proba, multi_class='ovr', average='weighted'
        )
    except Exception as e:
        logger.warning(f"ROC-AUC skipped: {e}")
        metrics['roc_auc'] = 0.0

    class_report = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )

    logger.info(f"Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1_score']:.4f}")
    return metrics, y_pred, y_pred_proba, class_report


# ─────────────────────────────────────────────────────────────────────────────
# 4. VISUALISATIONS
# WHY  : Plots reveal errors that numbers hide (e.g. class confusion)
# HOW  : seaborn / matplotlib → saved as PNG → logged as MLflow artifacts
# WHEN : After every evaluation
# WHERE: Stored in MLflow artifact store
# WHAT : Confusion matrix, feature importance, prediction distribution
# ─────────────────────────────────────────────────────────────────────────────
def create_visualizations(model, X_test, y_test, y_pred,
                          feature_names, target_names):
    logger.info("Creating visualisations …")
    sns.set_style("whitegrid")
    saved = {}

    # 4a. Confusion Matrix ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')
    path = str(ARTIFACTS_DIR / 'confusion_matrix.png')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    saved['confusion_matrix'] = path

    # 4b. Feature Importance ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    imp = pd.DataFrame({'feature': feature_names,
                        'importance': model.feature_importances_}
                       ).sort_values('importance', ascending=True)
    ax.barh(imp['feature'], imp['importance'], color='steelblue')
    ax.set_xlabel('Importance'); ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    path = str(ARTIFACTS_DIR / 'feature_importance.png')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    saved['feature_importance'] = path

    # 4c. Prediction Distribution ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    pd.Series(y_test).value_counts().sort_index().plot(kind='bar', ax=ax1,
        color='skyblue', edgecolor='black')
    ax1.set_title('True Distribution'); ax1.set_xticklabels(target_names, rotation=45)
    pd.Series(y_pred).value_counts().sort_index().plot(kind='bar', ax=ax2,
        color='lightcoral', edgecolor='black')
    ax2.set_title('Predicted Distribution'); ax2.set_xticklabels(target_names, rotation=45)
    path = str(ARTIFACTS_DIR / 'prediction_distribution.png')
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    saved['distribution'] = path

    logger.info(f"{len(saved)} visualisations created ✓")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# Complete MLOps workflow in one function
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=" * 70)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 70)

    try:
        # Step 1 ── Load data
        X_train, X_test, y_train, y_test, data_stats, feature_names, target_names = load_data()

        # Step 2 ── Start MLflow run
        run_name = f"rf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):

            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run ID: {run_id}")

            # Step 3 ── Log parameters
            params = {
                'model_type'         : 'RandomForest',
                'n_estimators'       : 100,
                'max_depth'          : 5,
                'min_samples_split'  : 2,
                'min_samples_leaf'   : 1,
                'random_state'       : RANDOM_SEED,
                'test_size'          : 0.2,
                'n_training_samples' : len(X_train)
            }
            mlflow.log_params(params)
            mlflow.log_dict(data_stats, "data_statistics.json")

            log_train_test_datasets(X_train, X_test, y_train, y_test)

            # Step 4 ── Train
            model = train_model(X_train, y_train,
                                n_estimators     = params['n_estimators'],
                                max_depth        = params['max_depth'],
                                min_samples_split= params['min_samples_split'],
                                min_samples_leaf = params['min_samples_leaf'])

            # Step 5 ── Evaluate
            metrics, y_pred, y_pred_proba, class_report = evaluate_model(
                model, X_test, y_test, target_names)

            # Step 6 ── Log metrics
            mlflow.log_metrics(metrics)
            mlflow.log_dict(class_report, "classification_report.json")

            # Step 7 ── Log visualisations
            figs = create_visualizations(model, X_test, y_test, y_pred,
                                         feature_names, target_names)
            for path in figs.values():
                mlflow.log_artifact(path)

            # Step 8 ── Log model with signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(model, "model",
                                     signature=signature,
                                     input_example=X_train.iloc[:5])

            # Step 9 ── Tags
            mlflow.set_tags({
                "team"         : "data-science",
                "project"      : "iris-classification",
                "framework"    : "scikit-learn",
                "algorithm"    : "RandomForest",
                "dataset"      : "iris",
                "environment"  : os.getenv("ENVIRONMENT", "development"),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "training_date": datetime.now().strftime("%Y-%m-%d")
            })

            # Step 10 ── Summary
            print("\n" + "=" * 70)
            print("✅  TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"\n📊 METRICS:")
            for k, v in metrics.items():
                print(f"   ├── {k:<12}: {v:.4f}")
            print(f"\n🔗 Run ID : {run_id}")
            print(f"🌐 UI     : {MLFLOW_TRACKING_URI}")
            print("=" * 70 + "\n")

            return model, metrics, run_id

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
