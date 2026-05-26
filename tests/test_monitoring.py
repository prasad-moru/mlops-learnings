"""
Unit Tests for Monitoring / Drift Detection

WHY  : Broken drift detection means silent model degradation.
HOW  : Construct reference and production DataFrames; verify KS logic.
WHEN : Every push via CI/CD.
WHERE: Local + CI/CD.
WHAT : Tests drift detection with no-drift and high-drift scenarios.
"""

import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring import detect_data_drift

COLS = ['sepal length (cm)', 'sepal width (cm)',
        'petal length (cm)', 'petal width (cm)']

np.random.seed(42)
REFERENCE = pd.DataFrame(np.random.normal(0, 1, (100, 4)), columns=COLS)
SAME      = pd.DataFrame(np.random.normal(0, 1, (50,  4)), columns=COLS)  # no drift
SHIFTED   = pd.DataFrame(np.random.normal(5, 1, (50,  4)), columns=COLS)  # heavy drift


# ── No-drift scenario ─────────────────────────────────────────────────────────

def test_no_drift_detected_for_same_distribution():
    report = detect_data_drift(REFERENCE, SAME, threshold=0.05)
    assert report['drift_detected'] is False


def test_report_contains_all_features():
    report = detect_data_drift(REFERENCE, SAME)
    for col in COLS:
        assert col in report['features']


def test_p_values_are_in_valid_range():
    report = detect_data_drift(REFERENCE, SAME)
    for feat, vals in report['features'].items():
        assert 0.0 <= vals['p_value'] <= 1.0


# ── Drift scenario ────────────────────────────────────────────────────────────

def test_drift_detected_for_shifted_distribution():
    report = detect_data_drift(REFERENCE, SHIFTED, threshold=0.05)
    assert report['drift_detected'] is True


def test_drifted_features_have_low_p_values():
    report = detect_data_drift(REFERENCE, SHIFTED, threshold=0.05)
    for feat, vals in report['features'].items():
        assert vals['p_value'] < 0.05   # all features shifted → all drift


# ── Report structure ──────────────────────────────────────────────────────────

def test_report_has_required_keys():
    report = detect_data_drift(REFERENCE, SAME)
    for key in ['timestamp', 'threshold', 'n_reference', 'n_production',
                'features', 'drift_detected']:
        assert key in report


def test_report_sample_sizes():
    report = detect_data_drift(REFERENCE, SAME)
    assert report['n_reference']  == 100
    assert report['n_production'] == 50


def test_threshold_respected():
    """Higher threshold = easier to flag drift."""
    report_strict = detect_data_drift(REFERENCE, SAME, threshold=0.001)
    report_loose  = detect_data_drift(REFERENCE, SAME, threshold=0.999)
    # With threshold=0.999 almost everything should flag as drift
    assert report_loose['drift_detected'] is True
