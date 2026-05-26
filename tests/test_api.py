"""
Unit Tests for FastAPI Endpoints

WHY  : API bugs cause downstream failures in all client apps.
HOW  : TestClient from fastapi.testclient simulates HTTP requests.
WHEN : Every code push (CI/CD).
WHERE: Local dev + CI/CD runner (no real server needed).
WHAT : Tests /health, /model-info, /predict endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ── Mock the model so tests don't need a live MLflow server ───────────────────
@pytest.fixture(autouse=True)
def mock_model():
    mock = MagicMock()
    mock.predict.return_value      = np.array([0])
    mock.predict_proba.return_value = np.array([[0.95, 0.03, 0.02]])

    with patch('api.model', mock), \
         patch('api.mlflow.sklearn.load_model', return_value=mock):
        yield mock


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from api import app
    return TestClient(app)


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_contains_status(client):
    response = client.get("/health")
    assert "status" in response.json()


# ── Predict endpoint ──────────────────────────────────────────────────────────

VALID_INPUT = {
    "sepal_length": 5.1,
    "sepal_width":  3.5,
    "petal_length": 1.4,
    "petal_width":  0.2
}


def test_predict_returns_200(client):
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200


def test_predict_response_structure(client):
    response = client.post("/predict", json=VALID_INPUT)
    body = response.json()
    for field in ['prediction', 'class_name', 'confidence']:
        assert field in body, f"Missing field: {field}"


def test_predict_class_name_is_valid(client):
    response = client.post("/predict", json=VALID_INPUT)
    assert response.json()['class_name'] in ['setosa', 'versicolor', 'virginica']


def test_predict_confidence_in_range(client):
    response = client.post("/predict", json=VALID_INPUT)
    conf = response.json()['confidence']
    assert 0.0 <= conf <= 1.0


def test_predict_rejects_negative_values(client):
    bad_input = {**VALID_INPUT, "sepal_length": -1.0}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422   # Pydantic validation error


def test_predict_rejects_missing_fields(client):
    incomplete = {"sepal_length": 5.1}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_string_values(client):
    bad_input = {**VALID_INPUT, "sepal_length": "not_a_number"}
    response = client.post("/predict", json=bad_input)
    assert response.status_code == 422
