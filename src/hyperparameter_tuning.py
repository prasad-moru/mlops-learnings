# src/hyperparameter_tuning.py
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
    }
    
    with mlflow.start_run(nested=True):
        model = train_model(X_train, y_train, **params)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
    return metrics['accuracy']

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)