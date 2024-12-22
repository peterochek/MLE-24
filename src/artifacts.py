import mlflow
from mlflow import log_artifact, log_metric, log_param


def log_mlflow_experiment():
    mlflow.set_experiment("Titanic Experiment")
    with mlflow.start_run():
        log_param("model", "Logistic Regression")
        log_metric("accuracy", 0.78)
        log_artifact("models/logistic_regression.pkl")
