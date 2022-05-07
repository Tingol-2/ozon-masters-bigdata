import datetime

import pendulum

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.latest_only import LatestOnlyOperator
from airflow.utils.trigger_rule import TriggerRule

with DAG(
    dag_id='Classification',
    schedule_interval=None,
    start_date=pendulum.datetime(2022, 5, 7, tz="UTC"),
    catchup=False,
    #tags=['example3'],
) as dag:

base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'
