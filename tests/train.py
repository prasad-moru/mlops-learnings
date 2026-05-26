"""
Production-Ready Training Pipeline with MLOps Best Practices

MLOPS CONCEPTS COVERED:
======================
1. EXPERIMENT TRACKING: Using MLflow to log parameters, metrics, and models
2. REPRODUCIBILITY: Setting random seeds and logging data versions
3. MODEL VERSIONING: Saving models with metadata for future reference
4. ARTIFACT MANAGEMENT: Logging plots, feature importance, and model artifacts
5. METADATA LOGGING: Tags, descriptions, and environment information

WHY THIS MATTERS:
================
- Reproducibility: Anyone can recreate exact results
- Auditability: Full history of what was tried and results
- Collaboration: Team members can see and build on each other's work
- Production Readiness: Clear path from experiment to deployment

WHEN TO USE:
===========
- Initial model training
- Retraining on new data
- A/B testing different algorithms
- Baseline model creation
"""

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

# Setup logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# MLOPS CONCEPT: CENTRALIZED CONFIGURATION
# WHY: Single source of truth for all settings
# BENEFIT: Easy to change behavior without code modification
# ============================================================================
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', 'iris_classification')
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
logger.info(f"Experiment Name: {EXPERIMENT_NAME}")


def load_data(test_size=0.2, random_state=RANDOM_SEED):
    """
    Load and prepare data with proper logging
    
    MLOPS CONCEPT: DATA VERSIONING & LINEAGE
    =========================================
    WHY: Track which data was used for training
    HOW: Log data hash, shape, and statistics
    WHEN: Every time data is loaded
    WHERE: In MLflow as dataset metadata
    
    Args:
        test_size: Proportion of data for testing (0.2 = 20%)
        random_state: Seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Train/test splits
    """
    logger.info("Loading Iris dataset...")
    
    # Load data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    # ========================================================================
    # MLOPS BEST PRACTICE: Log data statistics
    # WHY: Understand data distribution and detect data drift later
    # ========================================================================
    data_stats = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'target_distribution': y.value_counts().to_dict(),
        'data_hash': str(hash(X.values.tobytes()))  # For change detection
    }
    
    logger.info(f"Dataset loaded: {data_stats['n_samples']} samples, "
                f"{data_stats['n_features']} features")
    
    # Split data with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # IMPORTANT: Maintains class distribution in splits
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, data_stats


def train_model(X_train, y_train, n_estimators=100, max_depth=5, 
                min_samples_split=2, min_samples_leaf=1, random_state=RANDOM_SEED):
    """
    Train Random Forest model with specified hyperparameters
    
    MLOPS CONCEPT: HYPERPARAMETER TRACKING
    =======================================
    WHY: Know exactly which parameters produced which results
    HOW: Log all parameters to MLflow
    BENEFIT: Easy comparison of different configurations
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees in forest
        max_depth: Maximum depth of each tree
        min_samples_split: Minimum samples required to split node
        min_samples_leaf: Minimum samples required at leaf node
        random_state: Random seed for reproducibility
        
    Returns:
        Trained model object
    """
    logger.info(f"Training Random Forest with n_estimators={n_estimators}, "
                f"max_depth={max_depth}")
    
    # ========================================================================
    # MLOPS BEST PRACTICE: Explicit parameter specification
    # WHY: No hidden defaults, full control over model behavior
    # ========================================================================
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    
    return model


def evaluate_model(model, X_test, y_test, target_names=None):
    """
    Comprehensive model evaluation with multiple metrics
    
    MLOPS CONCEPT: COMPREHENSIVE METRICS LOGGING
    ============================================
    WHY: Single metric isn't enough to understand model performance
    HOW: Track accuracy, precision, recall, F1, ROC-AUC
    WHEN: Every evaluation (training, validation, production)
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        target_names: Names of target classes
        
    Returns:
        Dictionary of metrics and predictions
    """
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # ========================================================================
    # MLOPS BEST PRACTICE: Multiple evaluation metrics
    # WHY: Different metrics reveal different aspects of performance
    # - Accuracy: Overall correctness
    # - Precision: How many predicted positives are actually positive
    # - Recall: How many actual positives were found
    # - F1: Harmonic mean of precision and recall
    # - ROC-AUC: Model's ability to distinguish between classes
    # ========================================================================
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted'),
    }
    
    # ROC-AUC for multi-class (one-vs-rest)
    try:
        metrics['roc_auc'] = roc_auc_score(
            y_test, y_pred_proba, 
            multi_class='ovr', 
            average='weighted'
        )
    except Exception as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        metrics['roc_auc'] = None
    
    # Per-class metrics
    class_report = classification_report(
        y_test, y_pred, 
        target_names=target_names,
        output_dict=True
    )
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics, y_pred, y_pred_proba, class_report


def create_visualizations(model, X_train, X_test, y_test, y_pred, 
                         feature_names, target_names):
    """
    Create and save model visualization artifacts
    
    MLOPS CONCEPT: ARTIFACT LOGGING
    ================================
    WHY: Visual inspection reveals insights metrics can't capture
    HOW: Generate plots and log to MLflow
    WHAT: Confusion matrix, feature importance, prediction distribution
    
    Args:
        model: Trained model
        X_train: Training features (for feature importance context)
        X_test: Test features
        y_test: True labels
        y_pred: Predicted labels
        feature_names: Names of features
        target_names: Names of target classes
        
    Returns:
        Dictionary of figure paths
    """
    logger.info("Creating visualizations...")
    
    figures = {}
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # ========================================================================
    # 1. CONFUSION MATRIX
    # WHY: Shows where model is making mistakes
    # INTERPRETATION: Diagonal = correct predictions, off-diagonal = errors
    # ========================================================================
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names,
                yticklabels=target_names,
                ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    confusion_matrix_path = 'confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()
    figures['confusion_matrix'] = confusion_matrix_path
    
    # ========================================================================
    # 2. FEATURE IMPORTANCE
    # WHY: Understand which features drive predictions
    # USE CASE: Feature selection, model interpretation, debugging
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    feature_importance_path = 'feature_importance.png'
    plt.tight_layout()
    plt.savefig(feature_importance_path, dpi=150, bbox_inches='tight')
    plt.close()
    figures['feature_importance'] = feature_importance_path
    
    # ========================================================================
    # 3. PREDICTION DISTRIBUTION
    # WHY: Check if predictions are balanced across classes
    # RED FLAG: Heavily imbalanced predictions might indicate bias
    # ========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # True distribution
    pd.Series(y_test).value_counts().sort_index().plot(
        kind='bar', ax=ax1, color='skyblue', edgecolor='black'
    )
    ax1.set_title('True Label Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=10)
    ax1.set_ylabel('Count', fontsize=10)
    ax1.set_xticklabels(target_names, rotation=45)
    
    # Predicted distribution
    pd.Series(y_pred).value_counts().sort_index().plot(
        kind='bar', ax=ax2, color='lightcoral', edgecolor='black'
    )
    ax2.set_title('Predicted Label Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_xticklabels(target_names, rotation=45)
    
    distribution_path = 'prediction_distribution.png'
    plt.tight_layout()
    plt.savefig(distribution_path, dpi=150, bbox_inches='tight')
    plt.close()
    figures['distribution'] = distribution_path
    
    logger.info(f"Created {len(figures)} visualizations")
    
    return figures


def main():
    """
    Main training pipeline with complete MLOps practices
    
    PIPELINE FLOW:
    ==============
    1. Load and version data
    2. Train model with specified parameters
    3. Evaluate with comprehensive metrics
    4. Create visualizations
    5. Log everything to MLflow
    6. Save model artifacts
    
    MLOPS PRINCIPLES DEMONSTRATED:
    ==============================
    ✓ Reproducibility (random seeds, versioning)
    ✓ Traceability (all parameters and metrics logged)
    ✓ Automation (single command to run entire pipeline)
    ✓ Monitoring (comprehensive metrics and visualizations)
    """
    
    logger.info("="*70)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*70)
    
    try:
        # ====================================================================
        # STEP 1: DATA LOADING & VERSIONING
        # ====================================================================
        X_train, X_test, y_train, y_test, data_stats = load_data()
        
        # Get iris metadata
        iris = load_iris()
        feature_names = list(iris.feature_names)
        target_names = list(iris.target_names)
        
        # ====================================================================
        # STEP 2: START MLFLOW RUN
        # MLOPS CONCEPT: Run Context Management
        # WHY: Groups all related artifacts, metrics, and parameters
        # ====================================================================
        with mlflow.start_run(run_name=f"rf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
            
            # ================================================================
            # STEP 3: LOG PARAMETERS
            # MLOPS CONCEPT: Hyperparameter Tracking
            # WHY: Reproduce results, compare experiments
            # ================================================================
            params = {
                'model_type': 'RandomForest',
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': RANDOM_SEED,
                'test_size': 0.2,
                'n_features': data_stats['n_features'],
                'n_training_samples': len(X_train)
            }
            
            mlflow.log_params(params)
            logger.info("Parameters logged to MLflow")
            
            # ================================================================
            # STEP 4: LOG DATASET INFORMATION
            # MLOPS CONCEPT: Data Versioning
            # ================================================================
            mlflow.log_dict(data_stats, "data_statistics.json")
            
            # ================================================================
            # STEP 5: TRAIN MODEL
            # ================================================================
            model = train_model(
                X_train, y_train,
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf']
            )
            
            # ================================================================
            # STEP 6: EVALUATE MODEL
            # ================================================================
            metrics, y_pred, y_pred_proba, class_report = evaluate_model(
                model, X_test, y_test, target_names
            )
            
            # ================================================================
            # STEP 7: LOG METRICS
            # MLOPS CONCEPT: Performance Tracking
            # WHY: Compare model versions, track improvements
            # ================================================================
            mlflow.log_metrics(metrics)
            mlflow.log_dict(class_report, "classification_report.json")
            logger.info("Metrics logged to MLflow")
            
            # ================================================================
            # STEP 8: CREATE AND LOG VISUALIZATIONS
            # MLOPS CONCEPT: Artifact Management
            # ================================================================
            figures = create_visualizations(
                model, X_train, X_test, y_test, y_pred,
                feature_names, target_names
            )
            
            for name, path in figures.items():
                mlflow.log_artifact(path)
                os.remove(path)  # Clean up local files
            
            logger.info("Visualizations logged to MLflow")
            
            # ================================================================
            # STEP 9: LOG MODEL
            # MLOPS CONCEPT: Model Versioning
            # WHY: Deploy specific versions, rollback if needed
            # ================================================================
            
            # Create model signature for input/output validation
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Log model with metadata
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                input_example=X_train.iloc[:5],  # Sample input for documentation
                registered_model_name=None  # Will register separately if needed
            )
            
            logger.info("Model logged to MLflow")
            
            # ================================================================
            # STEP 10: ADD TAGS FOR SEARCHABILITY
            # MLOPS CONCEPT: Metadata Tagging
            # WHY: Easy filtering and searching of experiments
            # ================================================================
            mlflow.set_tags({
                "team": "data-science",
                "project": "iris-classification",
                "framework": "scikit-learn",
                "algorithm": "RandomForest",
                "dataset": "iris",
                "environment": os.getenv("ENVIRONMENT", "development"),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "training_date": datetime.now().strftime("%Y-%m-%d")
            })
            
            logger.info("Tags added to MLflow run")
            
            # ================================================================
            # STEP 11: PRINT SUMMARY
            # ================================================================
            print("\n" + "="*70)
            print("TRAINING COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"\n📊 PERFORMANCE METRICS:")
            print(f"   ├── Accuracy:  {metrics['accuracy']:.4f}")
            print(f"   ├── Precision: {metrics['precision']:.4f}")
            print(f"   ├── Recall:    {metrics['recall']:.4f}")
            print(f"   ├── F1 Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc']:
                print(f"   └── ROC-AUC:   {metrics['roc_auc']:.4f}")
            
            print(f"\n🔧 MODEL CONFIGURATION:")
            print(f"   ├── Estimators: {params['n_estimators']}")
            print(f"   ├── Max Depth:  {params['max_depth']}")
            print(f"   └── Random Seed: {params['random_state']}")
            
            print(f"\n📁 ARTIFACTS LOGGED:")
            print(f"   ├── Model file")
            print(f"   ├── Confusion matrix")
            print(f"   ├── Feature importance plot")
            print(f"   ├── Prediction distribution")
            print(f"   └── Classification report")
            
            print(f"\n🔗 MLflow Run ID: {mlflow.active_run().info.run_id}")
            print(f"🌐 View in UI: {MLFLOW_TRACKING_URI}")
            print("="*70 + "\n")
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
