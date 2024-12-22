from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.preprocess import preprocess_data
from src.train import train_models


def preprocess_task(input_path: str, output_path: str):
    preprocess_data(input_path, output_path)


def train_task(input_path: str, output_dir: str):
    train_models(input_path, output_dir)


def create_dag(dag_id: str, default_args: dict, schedule_interval: str):
    with DAG(
        dag_id=dag_id,
        default_args=default_args,
        schedule_interval=schedule_interval,
        catchup=False,
    ) as dag:
        preprocess = PythonOperator(
            task_id="preprocess",
            python_callable=preprocess_task,
            op_kwargs={
                "input_path": "data/titanic.csv",
                "output_path": "data/processed.csv",
            },
        )

        train = PythonOperator(
            task_id="train",
            python_callable=train_task,
            op_kwargs={
                "input_path": "data/processed.csv",
                "output_dir": "models/",
            },
        )

        preprocess >> train

    return dag


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 12, 22),
}

dag_id = "preprocess_and_train"
schedule_interval = None
globals()[dag_id] = create_dag(dag_id, default_args, schedule_interval)
