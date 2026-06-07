from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)
import numpy as np


def evaluate_model(model, X_test, y_test):
    """Evaluate model with metrics that handle binary and multiclass targets."""

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
    }

    # F1: use binary default when binary, otherwise weighted for multiclass
    labels = np.unique(y_test)
    if labels.shape[0] <= 2:
        metrics["f1"] = f1_score(y_test, predictions)
    else:
        metrics["f1"] = f1_score(y_test, predictions, average="weighted")

    # ROC AUC: prefer predict_proba, handle multiclass with 'ovr'
    roc_auc = None
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            # binary
            if probs.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, probs[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, probs, multi_class="ovr", average="weighted")
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, scores, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc = None

    metrics["roc_auc"] = roc_auc

    return metrics