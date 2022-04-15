#!/opt/conda/envs/dsenv/bin/python
import os, sys

from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import Pipeline, PipelineModel

path_model = sys.argv[1]
path_dataset = sys.argv[2]
path_inference = sys.argv[3]


schema = StructType([
    StructField("overall", DoubleType()),
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

model = PipelineModel.load(path_model)

result = spark.read.json(path_dataset, schema=schema)
result = test.fillna({'summary':''})

predictions = model.transform(result)

predictions.select("id", "prediction").write.save(path_inference)

spark.stop()
