# src/monitoring.py
from scipy.stats import ks_2samp
import numpy as np

def detect_data_drift(reference_data, current_data, threshold=0.05):
    """Detect distribution drift using KS test"""
    drift_detected = {}
    
    for col in reference_data.columns:
        statistic, p_value = ks_2samp(
            reference_data[col], 
            current_data[col]
        )
        drift_detected[col] = p_value < threshold
        
        mlflow.log_metric(f"drift_pvalue_{col}", p_value)
    
    return drift_detected