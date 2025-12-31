import mlflow
from train import load_data, train_model, evaluate_model
import os
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment("iris_classification")

def run_experiment():
    """Run multiple experiments with different parameters"""
    
    X_train, X_test, y_train, y_test = load_data()
    
    # Different parameter combinations
    experiments = [
        {'n_estimators': 50, 'max_depth': 3},
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 150, 'max_depth': 7},
        {'n_estimators': 200, 'max_depth': 10},
    ]
    
    for params in experiments:
        with mlflow.start_run(run_name=f"rf_n{params['n_estimators']}_d{params['max_depth']}"):
            
            mlflow.log_params(params)
            
            model = train_model(X_train, y_train, 
                              params['n_estimators'], 
                              params['max_depth'])
            
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Experiment: {params} | Accuracy: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    run_experiment()