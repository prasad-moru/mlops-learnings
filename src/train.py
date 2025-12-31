import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_iris
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow.set_experiment("iris_classification")

def load_data():
    """Load and prepare data"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, n_estimators=100, max_depth=5):
    """Train model with given parameters"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    return metrics

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Start MLflow run
    with mlflow.start_run(run_name="rf_baseline"):
        
        # Parameters
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'model_type': 'RandomForest'
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = train_model(X_train, y_train, 
                          params['n_estimators'], 
                          params['max_depth'])
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained successfully!")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()