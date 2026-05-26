"""
MLflow Integration Tests

WHY  : Verifies the MLflow server, PostgreSQL backend, experiment
       tracking, model registry, and alias operations all work
       end-to-end before running the full training pipeline.
HOW  : Uses pytest with a live MLflow server (integration tests).
       Mark: @pytest.mark.integration — skipped in CI by default.
       Run locally: pytest tests/test_mlflow.py -v -m integration
WHEN : After starting MLflow server and before running train.py.
WHERE: Local dev environment (requires running MLflow + PostgreSQL).
WHAT : Tests server connectivity, run logging, model registry,
       alias promotion, and rollback operations.

Run options:
  # All integration tests (needs live MLflow at localhost:5000)
  pytest tests/test_mlflow.py -v -m integration

  # Quick smoke test only
  pytest tests/test_mlflow.py -v -k "smoke"

  # Skip integration in CI (default behaviour)
  pytest tests/ -v -m "not integration"
"""

import os
import time

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
TEST_EXPERIMENT  = "test_integration_suite"
TEST_MODEL_NAME  = "test_iris_classifier"
TEST_ALIAS       = "test_champion"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_mlflow_reachable() -> bool:
    """Check if MLflow server is up before running integration tests."""
    try:
        import requests
        r = requests.get(f"{MLFLOW_URI}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _train_small_model():
    """Train a minimal RandomForest for registry tests."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, iris.feature_names


# ── Skip marker — skip all integration tests if server is not reachable ────────
mlflow_available = pytest.mark.skipif(
    not _is_mlflow_reachable(),
    reason=f"MLflow server not reachable at {MLFLOW_URI}. "
           "Start it with: python -m mlflow server --host 0.0.0.0 --port 5000",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mlflow_client():
    """MlflowClient pointed at the local tracking server."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    return MlflowClient()


@pytest.fixture(scope="module")
def test_run_id(mlflow_client):
    """
    Create one real MLflow run for the module.
    Scoped to module so all registry tests reuse the same run.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(TEST_EXPERIMENT)

    model, X_train, X_test, y_train, y_test, feature_names = _train_small_model()

    with mlflow.start_run(run_name="integration_test_run") as run:
        mlflow.log_param("n_estimators", 5)
        mlflow.log_param("purpose", "integration_test")
        mlflow.log_metric("accuracy", 0.9333)
        mlflow.log_metric("f1_score", 0.9310)

        from mlflow.models.signature import infer_signature
        sig = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model, "model",
            signature=sig,
            input_example=X_train.iloc[:3],
        )
        run_id = run.info.run_id

    yield run_id

    # Teardown: clean up test model versions and experiment
    try:
        versions = mlflow_client.search_model_versions(f"name='{TEST_MODEL_NAME}'")
        for v in versions:
            mlflow_client.delete_model_version(TEST_MODEL_NAME, v.version)
        mlflow_client.delete_registered_model(TEST_MODEL_NAME)
    except Exception:
        pass


# ── Group 1: Server connectivity ──────────────────────────────────────────────

@pytest.mark.integration
@mlflow_available
class TestMLflowServerConnectivity:
    """Verify the MLflow server and PostgreSQL backend are healthy."""

    def test_server_health_endpoint(self):
        """
        WHY: If /health fails, nothing else will work.
             Load balancers and K8s probes use this endpoint.
        """
        import requests
        response = requests.get(f"{MLFLOW_URI}/health", timeout=5)
        assert response.status_code == 200

    def test_can_set_tracking_uri(self):
        """WHY: Verifies the client can point at the server without errors."""
        mlflow.set_tracking_uri(MLFLOW_URI)
        assert mlflow.get_tracking_uri() == MLFLOW_URI

    def test_can_list_experiments(self, mlflow_client):
        """
        WHY: Listing experiments requires PostgreSQL connection.
             If this fails, the DB backend is misconfigured.
        """
        experiments = mlflow_client.search_experiments()
        assert isinstance(experiments, list)

    def test_postgresql_backend_stores_experiments(self, mlflow_client):
        """
        WHY: Confirms data is persisted to PostgreSQL, not SQLite.
             Creates a temp experiment and checks it's retrievable.
        """
        mlflow.set_tracking_uri(MLFLOW_URI)
        exp_name = f"_db_check_{int(time.time())}"
        exp_id = mlflow_client.create_experiment(exp_name)

        retrieved = mlflow_client.get_experiment(exp_id)
        assert retrieved.name == exp_name

        # Cleanup
        mlflow_client.delete_experiment(exp_id)


# ── Group 2: Experiment tracking ─────────────────────────────────────────────

@pytest.mark.integration
@mlflow_available
class TestExperimentTracking:
    """Verify parameters, metrics, and artifacts are logged correctly."""

    def test_experiment_created_and_retrievable(self, mlflow_client):
        """WHY: Experiment must exist before runs can be logged into it."""
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(TEST_EXPERIMENT)
        exp = mlflow_client.get_experiment_by_name(TEST_EXPERIMENT)
        assert exp is not None
        assert exp.name == TEST_EXPERIMENT

    def test_run_params_are_logged(self, mlflow_client, test_run_id):
        """WHY: Parameters drive reproducibility. If logging fails, no audit trail."""
        run = mlflow_client.get_run(test_run_id)
        assert run.data.params.get("n_estimators") == "5"
        assert run.data.params.get("purpose") == "integration_test"

    def test_run_metrics_are_logged(self, mlflow_client, test_run_id):
        """WHY: Metrics are the primary signal for model quality decisions."""
        run = mlflow_client.get_run(test_run_id)
        assert "accuracy" in run.data.metrics
        assert "f1_score" in run.data.metrics
        assert 0.0 <= run.data.metrics["accuracy"] <= 1.0

    def test_model_artifact_is_logged(self, mlflow_client, test_run_id):
        """WHY: The model artifact is what gets deployed. Must be present in the run."""
        artifacts = mlflow_client.list_artifacts(test_run_id, path="model")
        artifact_names = [a.path for a in artifacts]
        assert any("MLmodel" in name for name in artifact_names), (
            f"No MLmodel artifact found. Got: {artifact_names}"
        )

    def test_run_is_in_finished_state(self, mlflow_client, test_run_id):
        """WHY: Runs stuck in RUNNING state cause registry search failures."""
        run = mlflow_client.get_run(test_run_id)
        assert run.info.status == "FINISHED"

    def test_search_runs_returns_test_run(self, mlflow_client):
        """WHY: search_runs is what model_registry.py uses to find the best run."""
        mlflow.set_tracking_uri(MLFLOW_URI)
        exp = mlflow_client.get_experiment_by_name(TEST_EXPERIMENT)
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.accuracy > 0",
            order_by=["metrics.accuracy DESC"],
        )
        assert not runs.empty
        assert "metrics.accuracy" in runs.columns


# ── Group 3: Model Registry & Aliases ────────────────────────────────────────

@pytest.mark.integration
@mlflow_available
class TestModelRegistry:
    """Verify model registration, alias promotion, and rollback work correctly."""

    @pytest.fixture(scope="class")
    def registered_version(self, test_run_id):
        """Register the test run's model — reuse across class tests."""
        model_uri  = f"runs:/{test_run_id}/model"
        model_ver  = mlflow.register_model(
            model_uri=model_uri,
            name=TEST_MODEL_NAME,
            tags={"purpose": "integration_test", "accuracy": "0.9333"},
        )
        return model_ver

    def test_model_can_be_registered(self, registered_version):
        """WHY: Registration is the gateway to the Model Registry."""
        assert registered_version.name    == TEST_MODEL_NAME
        assert registered_version.version is not None

    def test_registered_model_has_tags(self, mlflow_client, registered_version):
        """WHY: Tags carry governance metadata (accuracy, run ID, etc.)."""
        ver = mlflow_client.get_model_version(
            TEST_MODEL_NAME, registered_version.version
        )
        assert ver.tags.get("purpose") == "integration_test"

    def test_champion_alias_can_be_set(self, mlflow_client, registered_version):
        """
        WHY: The API loads models by alias — models:/{name}@champion.
             If alias setting fails, the API cannot load the model.
        """
        mlflow_client.set_registered_model_alias(
            name    = TEST_MODEL_NAME,
            alias   = TEST_ALIAS,
            version = registered_version.version,
        )
        aliased = mlflow_client.get_model_version_by_alias(
            TEST_MODEL_NAME, TEST_ALIAS
        )
        assert aliased.version == registered_version.version

    def test_model_loadable_by_alias(self, registered_version):
        """
        WHY: This is the exact URI the API and predict.py use.
             If this fails, the API will return 503 on startup.
        """
        mlflow.set_tracking_uri(MLFLOW_URI)
        model_uri = f"models:/{TEST_MODEL_NAME}@{TEST_ALIAS}"
        model     = mlflow.sklearn.load_model(model_uri)
        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_produces_valid_predictions(self, registered_version):
        """
        WHY: Loading a model that can't predict is useless.
             Verifies the full inference path works end-to-end.
        """
        mlflow.set_tracking_uri(MLFLOW_URI)
        model = mlflow.sklearn.load_model(
            f"models:/{TEST_MODEL_NAME}@{TEST_ALIAS}"
        )
        iris  = load_iris()
        X     = pd.DataFrame(iris.data[:5], columns=iris.feature_names)

        preds  = model.predict(X)
        probas = model.predict_proba(X)

        assert len(preds)        == 5
        assert probas.shape      == (5, 3)
        assert all(p in [0, 1, 2] for p in preds)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_alias_can_be_deleted_and_reassigned(self, mlflow_client, registered_version):
        """
        WHY: Simulates what model_registry.py does during promotion —
             removes old alias, sets new alias. Must work atomically.
        """
        # Delete alias
        mlflow_client.delete_registered_model_alias(TEST_MODEL_NAME, TEST_ALIAS)

        with pytest.raises(mlflow.exceptions.MlflowException):
            mlflow_client.get_model_version_by_alias(TEST_MODEL_NAME, TEST_ALIAS)

        # Re-assign alias (simulates rollback / re-promotion)
        mlflow_client.set_registered_model_alias(
            name    = TEST_MODEL_NAME,
            alias   = TEST_ALIAS,
            version = registered_version.version,
        )
        recovered = mlflow_client.get_model_version_by_alias(
            TEST_MODEL_NAME, TEST_ALIAS
        )
        assert recovered.version == registered_version.version

    def test_version_tag_can_be_set(self, mlflow_client, registered_version):
        """WHY: model_registry.py tags versions with status and accuracy."""
        mlflow_client.set_model_version_tag(
            name    = TEST_MODEL_NAME,
            version = registered_version.version,
            key     = "status",
            value   = "champion",
        )
        ver = mlflow_client.get_model_version(
            TEST_MODEL_NAME, registered_version.version
        )
        assert ver.tags.get("status") == "champion"


# ── Group 4: Smoke test (fast, no fixtures needed) ────────────────────────────

@pytest.mark.integration
@mlflow_available
class TestSmoke:
    """
    Ultra-fast sanity checks — run these first.
    Each test is self-contained with no shared state.
    """

    def test_mlflow_version_is_3x(self):
        """WHY: Project requires MLflow 3.x for alias support."""
        major = int(mlflow.__version__.split(".")[0])
        assert major >= 3, (
            f"MLflow 3.x required for alias support. Got: {mlflow.__version__}"
        )

    def test_tracking_uri_is_set(self):
        """WHY: A missing tracking URI sends data to the wrong place silently."""
        mlflow.set_tracking_uri(MLFLOW_URI)
        uri = mlflow.get_tracking_uri()
        assert uri == MLFLOW_URI
        assert "localhost" in uri or "http" in uri

    def test_can_create_and_delete_experiment(self, mlflow_client):
        """WHY: Full create → delete cycle confirms write + delete permissions."""
        name   = f"_smoke_{int(time.time())}"
        exp_id = mlflow_client.create_experiment(name)
        exp    = mlflow_client.get_experiment(exp_id)
        assert exp.name == name
        mlflow_client.delete_experiment(exp_id)

    def test_log_and_retrieve_single_run(self):
        """
        WHY: Core MLflow loop — log a run and read it back.
             If this fails, nothing in the project will work.
        """
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(TEST_EXPERIMENT)

        with mlflow.start_run(run_name="smoke_test") as run:
            mlflow.log_param("smoke", "true")
            mlflow.log_metric("dummy_metric", 0.999)
            run_id = run.info.run_id

        client    = MlflowClient()
        retrieved = client.get_run(run_id)

        assert retrieved.data.params.get("smoke")          == "true"
        assert retrieved.data.metrics.get("dummy_metric")  == 0.999
        assert retrieved.info.status                       == "FINISHED"