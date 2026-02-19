"""
Metrics and Evaluation Utilities
"""

from typing import Tuple, List
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
    }
    
    # AUC-ROC
    try:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Classification report
    metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
    
    return metrics


def get_roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds

