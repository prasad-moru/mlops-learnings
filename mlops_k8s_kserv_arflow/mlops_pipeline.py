
"""
MLOps Training Pipeline DAG
============================
File: k8s/phase5-airflow/dags/mlops_pipeline.py
Copy to Airflow scheduler pod via: kubectl cp (see 03-deploy-dag.txt)

WHAT THIS REPLACES:
  Before (manual):
    python src/train.py
    python src/hyperparameter_tuning.py
    python src/model_registry.py
    kubectl rollout restart deployment/fastapi -n mlops

  After (this DAG — automatic):
    Runs every Monday 2am.
    Each step runs as a pod in the mlops namespace.
    Failures retry once. Bad accuracy triggers alert branch.

PIPELINE FLOW:
  check_mlflow_health
         ↓
    train_model              (KubernetesPodOperator → train.py)
         ↓
    tune_model               (KubernetesPodOperator → hyperparameter_tuning.py)
         ↓
    accuracy_gate            (BranchPythonOperator — checks accuracy from XCom)
         ↓                              ↓
  register_champion          accuracy_below_threshold
         ↓                   (add Slack/email here in production)
    deploy_model
         ↓
  verify_deployment

KEY FIX — sidecar.istio.io/inject: "false":
  The mlops namespace has Istio injection enabled (required for KServe).
  Training pods are batch jobs — they run, complete, and exit.
  Istio sidecar injection on batch pods causes pod phase=Failed even when
  the main container exits with code 0. Disabling injection on training
  pods fixes this — they don't need the service mesh, just MLflow access.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

# ── Default args applied to every task in the DAG ────────────────────────────
default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ── Environment variables injected into every KubernetesPodOperator task ─────
MLFLOW_ENV = [
    k8s.V1EnvVar(
        name="MLFLOW_TRACKING_URI",
        value="http://mlflow.mlops.svc.cluster.local:5000"
    ),
    k8s.V1EnvVar(name="PYTHONUNBUFFERED",  value="1"),
    k8s.V1EnvVar(name="MODEL_NAME",        value="iris_classifier"),
    k8s.V1EnvVar(name="MODEL_ALIAS",       value="champion"),
    k8s.V1EnvVar(name="EXPERIMENT_NAME",   value="iris_classification"),
    k8s.V1EnvVar(name="MIN_ACCURACY",      value="0.90"),
    k8s.V1EnvVar(name="RANDOM_SEED",       value="42"),
    k8s.V1EnvVar(
        name="PYTHONWARNINGS",
        value="ignore::UserWarning:pydantic"
    ),
]

# ── Annotation applied to every KubernetesPodOperator task ───────────────────
# WHY: mlops namespace has Istio injection enabled (needed for KServe routing).
# Training pods are batch jobs — they run train.py and exit. Istio sidecar
# on batch pods causes pod phase=Failed even when the app exits with code 0,
# because the sidecar lifecycle conflicts with the pod completion lifecycle.
# Disabling injection on training pods fixes this completely.
NO_ISTIO = {"sidecar.istio.io/inject": "false"}

# ── Resource definitions ──────────────────────────────────────────────────────
TRAINING_RESOURCES = k8s.V1ResourceRequirements(
    requests={"cpu": "200m", "memory": "512Mi"},
    limits={"cpu": "1000m", "memory": "1Gi"},
)

LIGHT_RESOURCES = k8s.V1ResourceRequirements(
    requests={"cpu": "100m", "memory": "256Mi"},
    limits={"cpu": "300m", "memory": "512Mi"},
)


# ── Python callables ──────────────────────────────────────────────────────────

def check_mlflow_health(**context):
    """
    PythonOperator task.
    Pings MLflow /health before starting training.
    Raises RuntimeError if MLflow is unreachable — stops the pipeline early.
    """
    import urllib.request
    import urllib.error
    uri = "http://mlflow.mlops.svc.cluster.local:5000/health"
    try:
        with urllib.request.urlopen(uri, timeout=10) as resp:
            if resp.status == 200:
                logging.info("MLflow health check passed")
                return True
    except urllib.error.URLError as e:
        raise RuntimeError(f"MLflow not reachable at {uri}: {e}") from e


def accuracy_gate(**context):
    """
    BranchPythonOperator task.
    Reads model accuracy from XCom (pushed by train_model task).
    Returns the task_id of the next task to run:
      - "register_champion"          if accuracy >= threshold
      - "accuracy_below_threshold"   if accuracy < threshold
    """
    ti = context["ti"]
    accuracy = ti.xcom_pull(task_ids="train_model", key="accuracy")

    if accuracy is None:
        logging.warning("No accuracy in XCom — reading from MLflow directly")
        import mlflow
        client = mlflow.tracking.MlflowClient(
            "http://mlflow.mlops.svc.cluster.local:5000"
        )
        runs = client.search_runs(
            experiment_names=["iris_classification"],
            order_by=["start_time DESC"],
            max_results=1,
        )
        accuracy = runs[0].data.metrics.get("accuracy", 0.0) if runs else 0.0

    min_accuracy = float(os.getenv("MIN_ACCURACY", "0.90"))
    logging.info(f"Accuracy: {accuracy:.4f}  |  Threshold: {min_accuracy}")

    if accuracy >= min_accuracy:
        logging.info("Accuracy gate PASSED — registering as champion")
        return "register_champion"
    else:
        logging.warning(f"Accuracy gate FAILED — {accuracy:.4f} < {min_accuracy}")
        return "accuracy_below_threshold"


def trigger_kserve_rollout(**context):
    """
    PythonOperator task.
    Patches KServe InferenceService annotation to force a new Knative revision.
    Falls back gracefully if KServe is not installed.
    """
    from kubernetes import client, config
    config.load_incluster_config()

    custom_api = client.CustomObjectsApi()
    patch_body = {
        "metadata": {
            "annotations": {
                "mlops/last-trained": datetime.utcnow().isoformat()
            }
        }
    }
    try:
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace="mlops",
            plural="inferenceservices",
            name="iris-classifier",
            body=patch_body,
        )
        logging.info("InferenceService patched — new Knative revision will be created")
    except Exception as e:
        logging.warning(f"KServe patch skipped (not installed or error): {e}")
        logging.info("Model registered in MLflow. FastAPI will use it on next restart.")


# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="mlops_training_pipeline",
    description="Weekly iris classifier retraining — train, evaluate, register, deploy",
    default_args=default_args,
    schedule="0 2 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "training", "iris"],
) as dag:

    # ── Task 1: health check ──────────────────────────────────────────────────
    health_check = PythonOperator(
        task_id="check_mlflow_health",
        python_callable=check_mlflow_health,
    )

    # ── Task 2: train ─────────────────────────────────────────────────────────
    train_model = KubernetesPodOperator(
        task_id="train_model",
        name="airflow-train-model",
        namespace="mlops",
        image="prasad890/docker-fastapi:latest",
        image_pull_policy="IfNotPresent",
        annotations=NO_ISTIO,           # disable Istio sidecar on batch pod
        cmds=["python", "src/train.py"],
        env_vars=MLFLOW_ENV,
        container_resources=TRAINING_RESOURCES,
        is_delete_operator_pod=True,
        get_logs=True,
        in_cluster=True,
        startup_timeout_seconds=120,
    )

    # ── Task 3: hyperparameter tuning ─────────────────────────────────────────
    tune_model = KubernetesPodOperator(
        task_id="tune_model",
        name="airflow-tune-model",
        namespace="mlops",
        image="prasad890/docker-fastapi:latest",
        image_pull_policy="IfNotPresent",
        annotations=NO_ISTIO,           # disable Istio sidecar on batch pod
        cmds=["python", "src/hyperparameter_tuning.py"],
        env_vars=MLFLOW_ENV,
        container_resources=TRAINING_RESOURCES,
        is_delete_operator_pod=True,
        get_logs=True,
        in_cluster=True,
        startup_timeout_seconds=120,
    )

    # ── Task 4: accuracy gate ─────────────────────────────────────────────────
    accuracy_check = BranchPythonOperator(
        task_id="accuracy_gate",
        python_callable=accuracy_gate,
    )

    # ── Task 5a: register as champion ─────────────────────────────────────────
    register_champion = KubernetesPodOperator(
        task_id="register_champion",
        name="airflow-register-model",
        namespace="mlops",
        image="prasad890/docker-fastapi:latest",
        image_pull_policy="IfNotPresent",
        annotations=NO_ISTIO,           # disable Istio sidecar on batch pod
        cmds=["python", "src/model_registry.py"],
        env_vars=MLFLOW_ENV,
        container_resources=LIGHT_RESOURCES,
        is_delete_operator_pod=True,
        get_logs=True,
        in_cluster=True,
    )

    # ── Task 5b: alert branch (accuracy too low) ──────────────────────────────
    accuracy_below_threshold = EmptyOperator(
        task_id="accuracy_below_threshold",
    )

    # ── Task 6: deploy (patch KServe InferenceService) ────────────────────────
    deploy_model = PythonOperator(
        task_id="deploy_model",
        python_callable=trigger_kserve_rollout,
    )

    # ── Task 7: verify deployment ─────────────────────────────────────────────
    verify_deployment = KubernetesPodOperator(
        task_id="verify_deployment",
        name="airflow-verify-deployment",
        namespace="mlops",
        image="prasad890/docker-fastapi:latest",
        image_pull_policy="IfNotPresent",
        annotations=NO_ISTIO,           # disable Istio sidecar on batch pod
        cmds=["python", "-c", """
import requests, sys
url = 'http://fastapi.mlops.svc.cluster.local:8000'
r = requests.get(f'{url}/health')
health = r.json()
print('Health:', health)
if not health.get('model_loaded'):
    print('ERROR: model not loaded after deployment')
    sys.exit(1)
r = requests.post(f'{url}/predict',
    json={'sepal_length':5.1,'sepal_width':3.5,'petal_length':1.4,'petal_width':0.2})
print('Prediction:', r.json())
print('Deployment verified OK')
"""],
        env_vars=MLFLOW_ENV,
        container_resources=LIGHT_RESOURCES,
        is_delete_operator_pod=True,
        get_logs=True,
        in_cluster=True,
    )

    # ── Wire up the pipeline ──────────────────────────────────────────────────
    health_check >> train_model >> tune_model >> accuracy_check
    accuracy_check >> register_champion >> deploy_model >> verify_deployment
    accuracy_check >> accuracy_below_threshold
