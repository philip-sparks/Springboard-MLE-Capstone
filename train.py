"""
Contains the code handling how the model is trained
"""

# external imports
import pyspark.sql.functions as sqlf
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)

# local imoprts
import get_config
import utilities


def train_model(data_in, model_type='LogisticRegression'):
    """
    Train a model based on the provided data

    Args:
        data_in (SPARK dataframe) - data to be used for training 
            Has to contain the columns 'tag' and 'Body'
    Returns:
        An trained machine learning model!
    """
    regexTokenizer = RegexTokenizer(inputCol="Body", outputCol="words", pattern="\\W")
    add_stopwords = ["http", "https", "amp", "rt", "t", "c", "the"]
    stopwordsRemover = StopWordsRemover(
        inputCol="words", outputCol="filtered"
    ).setStopWords(add_stopwords)
    countVectors = CountVectorizer(
        inputCol="filtered", outputCol="features", vocabSize=1000, minDF=1.0
    )
    label_stringIdx = StringIndexer(inputCol="tag", outputCol="label")
    ml_model = utilities.get_model(model_type)

    pipeline = Pipeline(
        stages=[
            regexTokenizer,
            stopwordsRemover,
            countVectors,
            label_stringIdx,
            ml_model,
        ]
    )

    # Fit the pipeline to training documents.

    pipelineModel = pipeline.fit(data_in)

    return pipelineModel


def train_and_eval_model(conf, spark_in, sc_in):
    """
    Train a model based on the path provided in config.yaml. 
    Save the model to the path defined in onfig.yaml

    Args:
        conf (dict) - contains the configuration
        spark_in - SparkSession object
        sc_in - SparkContext object
    Returns:
    """

    data = spark_in.read.parquet(conf["tagged_emails_pipeline_path"])
    data = data.where(~(sqlf.col("Body").isNull()))

    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=100)

    pipelineModel = train_model(trainingData, conf["model_type"])

    preds = pipelineModel.transform(testData)

    eval_model(preds, pipelineModel)

    try:
        pipelineModel.save(conf["model_path"])
    except:
        pipelineModel.write().overwrite().save(conf["model_path"])


def eval_model(test_preds, model):
    """
    Evaluate the ml model given the predictions and test data

    Args:
        test_preds - a list of transformed prediction data
        model - the ml pipelined model
    Returns:
    A confusion matrix, along with the precision, recall and F1 score of the currently trained model
    """
    metrics = MulticlassMetrics(test_preds.select("prediction", "label").rdd)

    # Overall statistics
    precision = metrics.precision()
    recall = metrics.recall()
    f1Score = metrics.fMeasure()
    print("Confusion matrix")
    print(metrics.confusionMatrix())
    print("Summary Stats")
    print("Precision = %s" % precision)
    print("Recall = %s" % recall)
    print("F1 Score = %s" % f1Score)


if __name__ == "__main__":
    conf = get_config.get_config()
    spark, sc = utilities.get_spark_session("preprocessing")
    train_and_eval_model(conf, spark, sc)
    print("Done.")
