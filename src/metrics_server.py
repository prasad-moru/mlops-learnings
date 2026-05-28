"""
ML Metrics Prometheus Exporter

WHY  : FastAPI /metrics gives HTTP stats. This gives ML stats —
       model accuracy, drift scores, version, service health.
       Together they give a complete MLOps observability picture.
HOW  : prometheus_client HTTP server on port 8001.
       Runs KS drift detection + reads MLflow registry every INTERVAL seconds.
WHEN : Start alongside FastAPI, keep running as background process.
WHAT : Prometheus metrics exported:
         mlops_api_health              1=up 0=down
         mlops_mlflow_health           1=up 0=down
         mlops_model_accuracy          champion accuracy tag from registry
         mlops_model_version           champion version number
         mlops_drift_detected          0=clean 1=drift
         mlops_drift_ks_statistic      KS statistic per feature
         mlops_drift_pvalue            p-value per feature (< 0.05 = drift)
         mlops_scrape_duration_seconds how long each collection took
         mlops_scrape_errors_total     error counter per component

Run:
  python src/metrics_server.py
  python src/metrics_server.py --port 8001 --interval 60
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from prometheus_client import (
    Gauge, Counter, start_http_server,
)
import requests
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

load_dotenv()
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config — all values come from your .env ────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
API_URI         = os.getenv("API_URI",             "http://localhost:8000")
MODEL_NAME      = os.getenv("MODEL_NAME",          "iris_classifier")
MODEL_ALIAS     = os.getenv("MODEL_ALIAS",         "champion")
METRICS_PORT    = int(os.getenv("METRICS_PORT",    "8001"))
METRICS_INTERVAL= int(os.getenv("METRICS_INTERVAL","60"))

mlflow.set_tracking_uri(MLFLOW_URI)

# ── Prometheus gauge / counter definitions ────────────────────────────────────
API_HEALTH = Gauge(
    "mlops_api_health",
    "1 if FastAPI /health returns 200, else 0",
)
MLFLOW_HEALTH = Gauge(
    "mlops_mlflow_health",
    "1 if MLflow /health returns 200, else 0",
)
MODEL_ACCURACY = Gauge(
    "mlops_model_accuracy",
    "Champion model accuracy stored as a registry tag",
    ["model_name", "alias"],
)
MODEL_VERSION = Gauge(
    "mlops_model_version",
    "Champion model version number",
    ["model_name", "alias"],
)
DRIFT_DETECTED = Gauge(
    "mlops_drift_detected",
    "1 if data drift detected, 0 if distributions are clean",
    ["model_name"],
)
DRIFT_KS = Gauge(
    "mlops_drift_ks_statistic",
    "Kolmogorov-Smirnov statistic per feature (0=no drift, 1=max drift)",
    ["feature"],
)
DRIFT_PVALUE = Gauge(
    "mlops_drift_pvalue",
    "KS p-value per feature — below 0.05 means drift is statistically significant",
    ["feature"],
)
SCRAPE_DURATION = Gauge(
    "mlops_scrape_duration_seconds",
    "Wall-clock time taken for the last metrics collection cycle",
)
SCRAPE_ERRORS = Counter(
    "mlops_scrape_errors_total",
    "Number of errors during metric collection, labelled by component",
    ["component"],
)


# ── Reference data (same split used in train.py) ──────────────────────────────

def _build_reference_data() -> pd.DataFrame:
    iris = load_iris()
    X    = pd.DataFrame(iris.data, columns=iris.feature_names)
    X_train, _, _, _ = train_test_split(
        X, iris.target, test_size=0.2, random_state=42
    )
    return X_train


def _simulate_production_data() -> pd.DataFrame:
    """
    In a real project this would read from a prediction log store.
    Here we sample from the same iris distribution (no drift by default).
    To simulate drift: replace with shifted data or pass --drift to monitoring.py.
    """
    iris = load_iris()
    X    = pd.DataFrame(iris.data, columns=iris.feature_names)
    return X.sample(50, random_state=int(time.time()) % 999)


REFERENCE_DATA = _build_reference_data()


# ── Collector functions ───────────────────────────────────────────────────────

def collect_health():
    """Ping /health on both services — set gauge 1/0."""
    for name, url, gauge in [
        ("api",    f"{API_URI}/health",    API_HEALTH),
        ("mlflow", f"{MLFLOW_URI}/health", MLFLOW_HEALTH),
    ]:
        try:
            r = requests.get(url, timeout=3)
            value = 1 if r.status_code == 200 else 0
        except Exception:
            value = 0
            SCRAPE_ERRORS.labels(component=name).inc()
        gauge.set(value)
        logger.debug(f"health/{name}: {value}")


def collect_model_metrics():
    """
    Read champion alias from MLflow Model Registry.
    Extracts version number and accuracy tag — both stored by model_registry.py.
    """
    try:
        client  = MlflowClient()
        aliased = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        version  = int(aliased.version)
        accuracy = float(aliased.tags.get("accuracy", 0.0))

        MODEL_VERSION.labels(model_name=MODEL_NAME, alias=MODEL_ALIAS).set(version)
        MODEL_ACCURACY.labels(model_name=MODEL_NAME, alias=MODEL_ALIAS).set(accuracy)
        logger.info(f"Registry: {MODEL_NAME}@{MODEL_ALIAS} v{version} acc={accuracy:.4f}")

    except mlflow.exceptions.MlflowException as e:
        SCRAPE_ERRORS.labels(component="model_registry").inc()
        logger.warning(f"Registry read failed (model not registered yet?): {e}")
    except Exception as e:
        SCRAPE_ERRORS.labels(component="model_registry").inc()
        logger.error(f"Unexpected registry error: {e}")


def collect_drift():
    """
    KS test on each feature: training distribution vs simulated production.
    Sets per-feature KS statistic, p-value, and overall drift flag.

    Threshold: p-value < DRIFT_THRESHOLD from your .env (default 0.05).
    """
    drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.05"))

    try:
        production_data = _simulate_production_data()
        overall_drift   = False

        for col in REFERENCE_DATA.columns:
            stat, pvalue = ks_2samp(REFERENCE_DATA[col], production_data[col])
            safe_name    = col.replace(" ", "_").replace("(", "").replace(")", "")

            DRIFT_KS.labels(feature=safe_name).set(round(stat, 4))
            DRIFT_PVALUE.labels(feature=safe_name).set(round(pvalue, 4))

            if pvalue < drift_threshold:
                overall_drift = True
                logger.warning(f"DRIFT: {col}  KS={stat:.4f}  p={pvalue:.4f}")
            else:
                logger.debug(f"clean: {col}  KS={stat:.4f}  p={pvalue:.4f}")

        DRIFT_DETECTED.labels(model_name=MODEL_NAME).set(int(overall_drift))
        logger.info(f"Drift: {'⚠ DETECTED' if overall_drift else '✓ clean'}")

    except Exception as e:
        SCRAPE_ERRORS.labels(component="drift_detection").inc()
        logger.error(f"Drift collection failed: {e}")


def collect_all():
    """Run all collectors in sequence, record total duration."""
    t0 = time.time()
    collect_health()
    collect_model_metrics()
    collect_drift()
    duration = time.time() - t0
    SCRAPE_DURATION.set(duration)
    logger.info(f"Collection complete in {duration:.2f}s")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Metrics Prometheus exporter")
    parser.add_argument(
        "--port", type=int,
        default=METRICS_PORT,
        help=f"HTTP port to expose /metrics (default: {METRICS_PORT})",
    )
    parser.add_argument(
        "--interval", type=int,
        default=METRICS_INTERVAL,
        help=f"Seconds between collection cycles (default: {METRICS_INTERVAL})",
    )
    args = parser.parse_args()

    start_http_server(args.port)
    logger.info(f"ML metrics exporter listening on :{args.port}/metrics")
    logger.info(f"Collection interval: {args.interval}s")
    logger.info(f"MLflow:  {MLFLOW_URI}")
    logger.info(f"API:     {API_URI}")
    logger.info(f"Model:   {MODEL_NAME}@{MODEL_ALIAS}")

    # First collection immediately on startup
    collect_all()

    while True:
        time.sleep(args.interval)
        collect_all()
