"""
Model Monitoring & Data Drift Detection

WHY  : Models degrade silently in production.
       Without monitoring, bad predictions reach users for weeks.
HOW  : Compare production data distribution vs training data
       using the Kolmogorov-Smirnov statistical test.
WHEN : Run daily/hourly in production as a scheduled job.
WHERE: Production environment with access to live prediction logs.
WHAT : Returns per-feature drift scores; logs alerts to MLflow.

Two Types of Drift:
  1. DATA DRIFT     — feature distributions change (e.g. user demographics shift)
  2. CONCEPT DRIFT  — relationship between features and target changes

Detection Method — KS Test:
  H0: Both distributions are the same
  p-value < 0.05 → Reject H0 → DRIFT DETECTED
"""

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
mlflow.set_experiment("iris_monitoring")

DRIFT_THRESHOLD = float(os.getenv('DRIFT_THRESHOLD', '0.05'))


def get_reference_data():
    """Load training data as the reference distribution."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    X_train, _, _, _ = train_test_split(X, iris.target, test_size=0.2, random_state=42)
    return X_train


def simulate_production_data(drift_factor=0.0):
    """
    Simulate incoming production data.
    drift_factor=0.0  → no drift  (same as training)
    drift_factor=1.0  → high drift (shifted distribution)
    """
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Add artificial drift by shifting feature values
    X_prod = X + np.random.normal(loc=drift_factor, scale=0.1, size=X.shape)
    return X_prod.sample(50, random_state=42)


def detect_data_drift(reference_data: pd.DataFrame,
                      production_data: pd.DataFrame,
                      threshold: float = DRIFT_THRESHOLD) -> dict:
    """
    Run KS test on each feature and return drift report.

    Args:
        reference_data : Training / baseline distribution
        production_data: Live / recent prediction data
        threshold      : p-value below which drift is flagged

    Returns:
        dict with per-feature drift status, p-values, and overall flag
    """
    report = {
        'timestamp'       : datetime.now().isoformat(),
        'threshold'       : threshold,
        'n_reference'     : len(reference_data),
        'n_production'    : len(production_data),
        'features'        : {},
        'drift_detected'  : False
    }

    for col in reference_data.columns:
        stat, p_value = ks_2samp(reference_data[col], production_data[col])
        drifted = p_value < threshold

        report['features'][col] = {
            'ks_statistic': round(stat, 4),
            'p_value'     : round(p_value, 4),
            'drift'       : drifted
        }

        if drifted:
            report['drift_detected'] = True
            logger.warning(f"⚠️  DRIFT in '{col}'  p={p_value:.4f} < {threshold}")
        else:
            logger.info(f"✅  '{col}'  p={p_value:.4f}  OK")

    return report


def run_monitoring(drift_factor=0.0):
    """
    Full monitoring workflow — detect drift and log results to MLflow.

    Args:
        drift_factor: 0.0 = normal, >0 = simulated drift
    """
    logger.info("=" * 70)
    logger.info("MONITORING RUN")
    logger.info("=" * 70)

    reference_data  = get_reference_data()
    production_data = simulate_production_data(drift_factor=drift_factor)

    report = detect_data_drift(reference_data, production_data)

    # Log to MLflow
    with mlflow.start_run(run_name=f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tags({
            "type"        : "monitoring",
            "drift_factor": str(drift_factor)
        })

        # Log per-feature p-values as metrics
        for feat, vals in report['features'].items():
            safe_name = feat.replace(' ', '_').replace('(', '').replace(')', '')
            mlflow.log_metric(f"pvalue_{safe_name}", vals['p_value'])
            mlflow.log_metric(f"ks_{safe_name}",     vals['ks_statistic'])

        mlflow.log_metric("overall_drift", int(report['drift_detected']))
        mlflow.log_dict(report, "drift_report.json")

    # Print summary
    print("\n" + "=" * 70)
    print("📊  DRIFT DETECTION REPORT")
    print("=" * 70)
    for feat, vals in report['features'].items():
        status = "🚨 DRIFT" if vals['drift'] else "✅  OK"
        print(f"  {feat:<30} p={vals['p_value']:.4f}  {status}")
    overall = "🚨  DRIFT DETECTED — consider retraining!" if report['drift_detected'] \
              else "✅  No drift detected"
    print(f"\n  Overall: {overall}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--drift', type=float, default=0.0,
                        help='Simulate drift: 0=none, 1=high')
    args = parser.parse_args()
    run_monitoring(drift_factor=args.drift)
