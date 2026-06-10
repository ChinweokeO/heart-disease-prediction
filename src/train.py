import sys
import yaml
import os
import mlflow

print("TRACKING URI:", mlflow.get_tracking_uri())

# allow local file store (MLflow 3.x requirement)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# use simple local folder (NO file://)
mlflow.set_tracking_uri("mlruns")

from sklearn.ensemble import RandomForestClassifier

src_root = os.path.dirname(os.path.abspath(__file__))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from preprocessing import (
    load_data,
    preprocess_data,
    split_data,
)
from evaluation import evaluate_model


def load_config(config_path):
    """Load the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def train_model(config):
    """Train the model using configuration values."""
    target_column = config["training"]["target_column"]
    df = load_data(config["data"]["raw_data_path"], target_column=target_column)
    numeric_columns = df.select_dtypes(include="number").columns.drop(target_column).tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df = preprocess_data(df, numeric_columns, categorical_columns)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column,
        config["data"]["test_size"],
        config["data"]["random_state"],
    )

    model = RandomForestClassifier(
        n_estimators=config["model"].get("n_estimators", 200),
        max_depth=config["model"].get("max_depth", None),
        min_samples_split=config["model"].get("min_samples_split", 5),
        min_samples_leaf=config["model"].get("min_samples_leaf", 2),
        class_weight=config["model"].get("class_weight", "balanced"),
        n_jobs=config["model"].get("n_jobs", -1),
        random_state=config["data"]["random_state"],
    )

    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics


if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "configs", "config.yaml")

    config = load_config(config_path)
    model, metrics = train_model(config)

    #try:
        #import mlflow
        #import mlflow.sklearn
    #except ImportError:
       # mlflow = None
        #print("Warning: mlflow is not installed. Skipping MLflow logging.")
    
    if mlflow is not None:
        
       with mlflow.start_run():
            mlflow.log_params(config["model"])
            mlflow.log_param("data_version", "heart_v1")
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            mlflow.sklearn.log_model(model, "model")

    print("Model metrics:")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  f1: {metrics['f1']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"  roc_auc: {metrics['roc_auc']:.4f}")
    else:
        print("  roc_auc: not available")

    if metrics["f1"] < config["metrics"]["minimum_f1"]:
        raise ValueError("Model failed minimum F1 threshold")