"""
Unit Tests for Training Pipeline

WHY  : Tests catch bugs before they reach production.
       A model with a bug can silently produce wrong predictions.
HOW  : pytest fixtures + assert statements
WHEN : Run on every git push via CI/CD (GitHub Actions)
WHERE: Local dev + CI/CD runner
WHAT : Tests data loading, model training, and evaluation functions
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import load_data, train_model, evaluate_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    """Load data once and reuse across tests in this module."""
    return load_data()


@pytest.fixture(scope="module")
def trained_model(data):
    """Train a model once and reuse across tests."""
    X_train, X_test, y_train, y_test, _, feature_names, target_names = data
    return train_model(X_train, y_train)


# ── Data Tests ────────────────────────────────────────────────────────────────

def test_load_data_returns_correct_structure(data):
    X_train, X_test, y_train, y_test, stats, feature_names, target_names = data
    assert X_train is not None
    assert X_test  is not None
    assert y_train is not None
    assert y_test  is not None


def test_data_shapes(data):
    X_train, X_test, y_train, y_test, _, _, _ = data
    # 150 samples total, 80/20 split → 120 train, 30 test
    assert len(X_train) == 120
    assert len(X_test)  == 30
    assert X_train.shape[1] == 4    # 4 features
    assert X_test.shape[1]  == 4


def test_data_has_correct_features(data):
    X_train, _, _, _, _, feature_names, _ = data
    assert len(feature_names) == 4
    assert 'sepal length (cm)' in list(X_train.columns)
    assert 'petal length (cm)' in list(X_train.columns)


def test_no_missing_values(data):
    X_train, X_test, y_train, y_test, _, _, _ = data
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum()  == 0


def test_class_balance_in_splits(data):
    """Stratified split should preserve class proportions."""
    _, _, y_train, y_test, _, _, _ = data
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist  = y_test.value_counts(normalize=True).sort_index()
    for cls in range(3):
        assert abs(train_dist[cls] - test_dist[cls]) < 0.05  # within 5%


# ── Model Training Tests ───────────────────────────────────────────────────────

def test_model_is_not_none(trained_model):
    assert trained_model is not None


def test_model_has_correct_n_features(trained_model):
    assert trained_model.n_features_in_ == 4


def test_model_has_correct_n_classes(trained_model):
    assert len(trained_model.classes_) == 3


def test_model_has_feature_importances(trained_model):
    assert len(trained_model.feature_importances_) == 4
    assert abs(sum(trained_model.feature_importances_) - 1.0) < 1e-6  # sum to 1


def test_different_params_produce_different_models(data):
    X_train, _, y_train, _, _, _, _ = data
    model_a = train_model(X_train, y_train, n_estimators=10, max_depth=2)
    model_b = train_model(X_train, y_train, n_estimators=200, max_depth=10)
    # Different models should produce at least slightly different importances
    assert not all(
        abs(a - b) < 1e-10
        for a, b in zip(model_a.feature_importances_, model_b.feature_importances_)
    )


# ── Evaluation Tests ──────────────────────────────────────────────────────────

def test_evaluate_returns_all_metrics(data, trained_model):
    _, X_test, _, y_test, _, _, target_names = data
    metrics, _, _, _ = evaluate_model(trained_model, X_test, y_test, target_names)
    for key in ['accuracy', 'f1_score', 'precision', 'recall']:
        assert key in metrics


def test_metrics_are_in_valid_range(data, trained_model):
    _, X_test, _, y_test, _, _, target_names = data
    metrics, _, _, _ = evaluate_model(trained_model, X_test, y_test, target_names)
    for key in ['accuracy', 'f1_score', 'precision', 'recall']:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_minimum_accuracy(data, trained_model):
    """Model must achieve at least 80% accuracy to pass the quality gate."""
    _, X_test, _, y_test, _, _, target_names = data
    metrics, _, _, _ = evaluate_model(trained_model, X_test, y_test, target_names)
    assert metrics['accuracy'] >= 0.80, (
        f"Accuracy {metrics['accuracy']:.4f} below minimum threshold 0.80"
    )


def test_predictions_have_correct_shape(data, trained_model):
    _, X_test, _, y_test, _, _, target_names = data
    _, y_pred, y_proba, _ = evaluate_model(trained_model, X_test, y_test, target_names)
    assert len(y_pred)      == len(X_test)
    assert y_proba.shape    == (len(X_test), 3)   # 3 classes
