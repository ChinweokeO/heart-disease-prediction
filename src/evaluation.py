from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, predictions)
    }

    return metrics