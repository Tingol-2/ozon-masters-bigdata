import datetime

import pendulum

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.latest_only import LatestOnlyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator

base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'

with DAG(
    dag_id='Classification',
    schedule_interval=None,
    #start_date=pendulum.datetime(2022, 5, 7, tz="UTC"),
    catchup=False,
    #tags=['example3'],
) as dag:
    feature_eng_task = SparkSubmitOperator(
        application=f"{base_dir}preprocess.py"\
        , task_id="feature_eng_task"\
        , application_args = ['--path-in', '/datasets/amazon/all_reviews_5_core_train_extra_small_sentiment.json', '--path-out', 'Tingol-2_train_out']\
        ,spark_binary="/usr/bin/spark-submit"\
        ,env_vars={"PYSPARK_PYTHON": dsenv}
    )
    
    

