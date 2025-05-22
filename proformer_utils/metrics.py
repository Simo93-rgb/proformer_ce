import torch
import math
from typing import Dict

def compute_classification_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute precision, recall, f1, accuracy and Matthews Correlation Coefficient.

    Args:
        preds (torch.Tensor): Predicted labels (0/1).
        labels (torch.Tensor): True labels (0/1).

    Returns:
        dict: Dictionary with precision, recall, f1, accuracy, mcc.
    """
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = labels.size(0)
    
    # Calcolo delle metriche standard
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # Calcolo dell'MCC
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0
    
    return dict(precision=precision, recall=recall, f1=f1, accuracy=accuracy, mcc=mcc)

