"""
Dataset Versioning with MLflow

WHY  : "I don't know which data produced this model" is a production nightmare.
       Data versioning creates a permanent link: run ↔ dataset.
HOW  : mlflow.data.from_pandas() wraps a DataFrame with metadata.
       mlflow.log_input() attaches it to the active run.
WHEN : Called inside mlflow.start_run() before training starts.
WHERE: Training pipeline (train.py calls log_dataset).
WHAT : Creates a versioned dataset entry in the MLflow run.
"""

import mlflow
import pandas as pd


def log_dataset(X: pd.DataFrame, y: pd.Series, context: str = "training"):
    """
    Log a pandas DataFrame as a versioned MLflow dataset.

    Args:
        X       : Feature DataFrame
        y       : Target Series
        context : "training" or "testing"
    """
    df = X.copy()
    df['target'] = y.values

    dataset = mlflow.data.from_pandas(
        df,
        source="sklearn.datasets.load_iris",
        name=f"iris_{context}_data"
    )

    mlflow.log_input(dataset, context=context)


def log_train_test_datasets(X_train, X_test, y_train, y_test):
    """Convenience wrapper to log both splits in one call."""
    log_dataset(X_train, y_train, context="training")
    log_dataset(X_test,  y_test,  context="testing")
