from datetime import datetime, timedelta, time
import keyword
from re import T
from textwrap import dedent
from functools import partial
from airflow import DAG
import pendulum
from airflow.operators.bash import BashOperator
from airflow.sensors.time_sensor import TimeSensor
from airflow.sensors.python import PythonSensor
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup
import mlflow

local_timezone = pendulum.timezone("Asia/Seoul")
default_args = {
    "owner": "choihyunwoo",
    "depends_on_past": False,
    "email": ["pica-git0@github.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}


def random_check():
    from random import randint

    return True


def dummy_success():
    return True


with DAG(
    "mlflow-v2",
    default_args=default_args,
    description="mlflow 플로우(매주)",
    # schedule_interval="0 5 * * 4",
    schedule_interval="@week",
    start_date=pendulum.datetime(2024, 10, 16, tz=local_timezone),
    catchup=False,
    tags=["mlflow"],
) as dag:
    with TaskGroup("data_processing") as data_processing:

        data_collect = PythonOperator(
            task_id="data_collect",
            python_callable=dummy_success,
        )
        data_preprocess = PythonOperator(
            task_id="data_preprocess",
            python_callable=dummy_success,
        )
        data_collect >> data_preprocess

    with TaskGroup("data_processing") as model_processing:
        model_train = PythonOperator(
            task_id="model_train",
            python_callable=dummy_success,
        )
        model_eval = PythonOperator(
            task_id="model_eval",
            python_callable=dummy_success,
        )
        model_deploy = PythonOperator(
            task_id="model_deploy",
            python_callable=dummy_success,
        )
        model_train >> model_eval >> model_deploy

    data_processing >> model_processing
