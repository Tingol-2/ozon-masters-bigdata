#!/opt/conda/envs/dsenv/bin/python
import os, sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

path_dataset = sys.argv[1]
path_out = sys.argv[2]

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
    StructField("unixReviewTime", TimestampType())
])

dataset = spark.read.json(path_dataset, schema=schema)

dataset = dataset.fillna({'summary':''})

tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")


stop_words = StopWordsRemover.loadDefaultStopWords("english")

swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                          outputCol="words_filtered", stopWords=stop_words)


hasher = HashingTF(numFeatures=100, binary=True,
                    inputCol=swr.getOutputCol(), outputCol="word_vec")

vas = VectorAssembler(inputCols=['word_vec'], outputCol="features")

x = tokenizer.transform(dataset)
x = swr.transform(x)
x = hasher.transform(x)
x = vas.transform(x)

x.write().overwrite().save(path_out)
