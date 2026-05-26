# test_mlflow.py
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("employee-attrition")

with mlflow.start_run():
    mlflow.log_param("model", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("f1_score", 0.83)
    print("Run logged to MLflow!")