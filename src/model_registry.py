# src/model_registry.py
def register_best_model():
    """Register and promote best model"""
    client = mlflow.tracking.MlflowClient()
    
    # Find best run
    experiment = mlflow.get_experiment_by_name("iris_classification")
    best_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    ).iloc[0]
    
    # Register model
    model_uri = f"runs:/{best_run.run_id}/model"
    mlflow.register_model(model_uri, "iris_classifier")
    
    # Promote to production
    client.transition_model_version_stage(
        name="iris_classifier",
        version=1,
        stage="Production"
    )