#!/opt/conda/envs/dsenv/bin/python
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkConf
from pyspark.sq.functions 
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()

import pandas as pd
from joblib import load

path_in = sys.argv[1] 
path_out = sys.argv[2] 
model_path = sys.argv[3] 

model = load(model_path)

est_broadcast = spark.sparkContext.broadcast(model)

@F.pandas_udf(FloatType())
def predict(series):
    predictions = est_broadcast.value.predict(series.tolist())
    return pd.Series(predictions)

dataset = spark.read.json(path_in)

predicts = dataset.withColumn('predictions', predict(vector_to_array('features')))

predicts.select("id", "prediction").write.save(path_out)

spark.stop()
