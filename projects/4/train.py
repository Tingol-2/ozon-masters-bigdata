#!/opt/conda/envs/dsenv/bin/python
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

path_dataset = sys.argv[1]
path_model = sys.argv[2]


schema = StructType([
    StructField("overall", DoubleType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("id", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", TimestampType())
])

dataset = spark.read.json(path_dataset, schema=schema)

dataset = dataset.fillna({'summary':''})

dataset.cache()

pipeline_model = pipeline.fit(dataset)

pipeline_model.write().overwrite().save(path_model)

spark.stop()
