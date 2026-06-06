import yaml
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

from preprocess import (
    load_data,
    preprocess_data,
    split_data
)

from evaluation import evaluate_model

#Train Model
def load_config(config_path):

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def train_model(config):

    df = load_data(
        config["data"]["raw_data_path"]
    )

    numeric_columns = config["features"]["numeric"]
    categorical_columns = config["features"]["categorical"]

    df = preprocess_data(
        df,
        numeric_columns,
        categorical_columns
    )

    X_train, X_test, y_train, y_test = split_data(
        df,
        config["data"]["target_column"],
        config["split"]["test_size"],
        config["split"]["random_state"]
    )

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=42
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    return model, metrics

with mlflow.start_run():

    mlflow.log_params(config["model"])

    mlflow.log_param(
        "data_version",
        "heart_v1"
    )

    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        model,
        "model"
    )


if metrics["f1"] < config["metrics"]["minimum_f1"]:
    raise ValueError(
        "Model failed minimum F1 threshold"
    )