import mlflow
import mlflow.sklearn
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

def get_latest_run():
    """Get the latest run from the experiment"""
    experiment = mlflow.get_experiment_by_name("iris_classification")
    if experiment is None:
        raise ValueError("Experiment 'iris_classification' not found. Run train.py first!")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["start_time DESC"], 
        max_results=1
    )
    
    if runs.empty:
        raise ValueError("No runs found. Run train.py first!")
    
    return runs.iloc[0]['run_id']

def load_model(run_id):
    """Load model from MLflow"""
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def predict(model, data):
    """Make predictions"""
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Get latest run automatically
    run_id = get_latest_run()
    print(f"Using run_id: {run_id}")
    
    # Sample data
    sample_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], 
                              columns=['sepal length (cm)', 'sepal width (cm)', 
                                      'petal length (cm)', 'petal width (cm)'])
    
    model = load_model(run_id)
    prediction = predict(model, sample_data)
    
    class_names = ['setosa', 'versicolor', 'virginica']
    print(f"Input: {sample_data.values[0]}")
    print(f"Prediction: Class {prediction[0]} ({class_names[prediction[0]]})")
        
        # mlflow.log_params(params)
        
        # # Train model
        # from train import load_data, train_model
        # X_train, X_test, y_train, y_test = load_data()
        # model = train_model(X_train, y_train, 
        #                     params['n_estimators'], 
        #                     params['max_depth'])
        
        # # Log model
        # mlflow.sklearn.log_model(model, "model")
        
        # print("Model trained and logged successfully.")