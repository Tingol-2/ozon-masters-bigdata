#!/opt/conda/envs/dsenv/bin/python
from joblib import load
import os, sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import StructType, StructField, DoubleType, StringType, BooleanType, TimestampType
from pyspark.ml.linalg import VectorUDT


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

path_in = sys.argv[2] 
path_out = sys.argv[4] 
model_path = sys.argv[6] 

model = load(model_path)

est_broadcast = spark.sparkContext.broadcast(model)

@F.pandas_udf(FloatType())
def predict(series):
    predictions = est_broadcast.value.predict(series.tolist())
    return pd.Series(predictions)

schema = StructType([
    StructField("label", DoubleType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("id", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType()),
    StructField("words", VectorUDT()),
    StructField("words_filtered", VectorUDT()),
    StructField("word_vec", VectorUDT()),
    StructField("features", VectorUDT())
])


dataset = spark.read.json(path_in, schema=schema)

predicts = dataset.withColumn('predictions', predict(vector_to_array('features')))

predicts.select("id", "predictions").write.mode('overwrite').save(path_out, header='false', format='csv' )

spark.stop()
