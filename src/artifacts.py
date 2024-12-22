import mlflow
from mlflow import log_artifact, log_metric, log_param


def log_mlflow_experiment(name, accuracy, model_output_path):
    with mlflow.start_run(run_name=name):
        log_param("model", name)
        log_metric("accuracy", accuracy)
        log_artifact(f"{model_output_path}/{name.replace(' ', '_').lower()}.pkl")
