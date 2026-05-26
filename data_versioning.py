# src/data_versioning.py
import mlflow
from mlflow.data.pandas_dataset import PandasDataset

def log_dataset(X, y, context="training"):
    """Version your datasets"""
    dataset = mlflow.data.from_pandas(
        pd.concat([X, y], axis=1),
        source="sklearn.datasets.load_iris",
        name=f"iris_{context}"
    )
    mlflow.log_input(dataset, context=context)