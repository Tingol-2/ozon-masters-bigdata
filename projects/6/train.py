#!/opt/conda/envs/dsenv/bin/python
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkConf

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

conf = SparkConf()
spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()

from sklearn.ensemble import GradientBoostingClassifier
from pyspark.ml.functions import vector_to_array
import pandas as pd
from joblib import dump

path_in = sys.argv[1] 
path_out = sys.argv[2] 

dataset = spark.read.json(path_in)

features = (dataset.withColumn("f", vector_to_array("features"))
            .select(['id'] + [col("f")[i] for i in range(100)])).toPandas()

target = dataset.select('label').toPandas()

feats = list(features.columns)[1:]

model = GradientBoostingClassifier()
model.fit(features[feats], target['label'])

dump(model, path_out)

spark.stop()
