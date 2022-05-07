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

def main():
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

if __name__ == "__main__":
    main()
    
spark.stop()
