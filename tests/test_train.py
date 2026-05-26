# tests/test_train.py
import pytest
from src.train import load_data, train_model, evaluate_model

def test_load_data():
    X_train, X_test, y_train, y_test = load_data()
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert X_train.shape[1] == 4

def test_model_training():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    assert model is not None
    
def test_model_performance():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics['accuracy'] > 0.8  # Minimum acceptable