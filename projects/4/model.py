#!/opt/conda/envs/dsenv/bin/python
from pyspark.ml.feature import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline

tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="\\W")


stop_words = StopWordsRemover.loadDefaultStopWords("english")

swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(),
                          outputCol="words_filtered", stopWords=stop_words)


hasher = HashingTF(numFeatures=100, binary=True,
                    inputCol=swr.getOutputCol(), outputCol="word_vec")

vas = VectorAssembler(inputCols=['word_vec'], outputCol="features")

gbt = GBTRegressor(featuresCol="features", labelCol="overall", maxIter=12)

pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    hasher,
    vas,
    gbt
])
