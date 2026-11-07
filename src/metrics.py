"""Classification metrics used for every model (and the human baseline)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """accuracy, precision, recall, F1 (positive class = hateful, zero_division=0),
    plus AUROC when probabilities are given (NaN if y_true has only one class)."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float).ravel()
        out["auroc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return out
