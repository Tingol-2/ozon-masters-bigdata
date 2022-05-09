import datetime

import pendulum

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.latest_only import LatestOnlyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.contrib.sensors.file_sensor import FileSensor

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
        ,env_vars={"PYSPARK_PYTHON": '/opt/conda/envs/dsenv/bin/python'}
    )
    
    train_download_task = BashOperator(
                            task_id='train_download_task',
                            bash_command="hdfs dfs -getmerge Tingol-2_train_out {base_dir}Tingol-2_train_out_local",
     )
    
    
    train_task = SparkSubmitOperator(
        application=f"{base_dir}train.py"\
        , task_id="train_task"\
        , application_args = ['--train-in', '{base_dir}Tingol-2_train_out_local', '--sklearn-model-out', f'{base_dir}6.joblib']\
        ,spark_binary="/usr/bin/spark-submit"\
        ,env_vars={"PYSPARK_PYTHON": '/opt/conda/envs/dsenv/bin/python'}
    )
    
    model_sensor = FileSensor( task_id= "model_sensor", poke_interval= 30,  filepath= f'{base_dir}6.joblib' )
    
    feature_eng_task = SparkSubmitOperator(
        application=f"{base_dir}preprocess.py"\
        , task_id="feature_eng_task"\
        , application_args = ['--path-in', '/datasets/amazon/all_reviews_5_core_test_extra_small_features.json', '--path-out', 'Tingol-2_test_out']\
        ,spark_binary="/usr/bin/spark-submit"\
        ,env_vars={"PYSPARK_PYTHON": '/opt/conda/envs/dsenv/bin/python'}
    )

    predict_task = SparkSubmitOperator(
        application=f"{base_dir}predict.py"\
        , task_id="predict_task"\
        , application_args = ['--train-in', 'Tingol-2_test_out', '--pred-out', 'Tingol-2_hw6_prediction', '--sklearn-model-in', f'{base_dir}6.joblib']\
        ,spark_binary="/usr/bin/spark-submit"\
        ,env_vars={"PYSPARK_PYTHON": '/opt/conda/envs/dsenv/bin/python'}
    ) 
    

feature_eng_task >>  train_download_task >> train_task >> model_sensor >> feature_eng_task >> predict_task
