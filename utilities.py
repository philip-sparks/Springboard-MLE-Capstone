"""
Contains utilities and helper functions
"""

# external imports
import findspark

findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import NaiveBayes


def get_spark_session(app_name):
    """
    Create a spark session and spark context

    Args:
    app_name - spark application name

    Returns:
    SparkSession, SparkContext
    """

    conf = SparkConf().setAppName(app_name).set("spark.ui.showConsoleProgress", True)
    #conf.set('spark.jars.packages', 'org.apache.hadoop:hadoop-aws:2.7.2')
    sc = SparkContext(conf=conf)
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    return spark, sc


def get_model(model_string='LogisticRegression'):
    """
    Get the desired model object for training and classification

    Args:
    Returns:
    model object from pyspark.ml.classification
    """
    models_dict = {
        'LogisticRegression': LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(numTrees=10),
        # Deep Learning note: the number on neurons in the last layer needs to equal
        # the number of categories. The number of neurons in the first layer
        # needs to be equal to the vocabulary of count vectorizer
        'MultilayerPerceptronClassifier': MultilayerPerceptronClassifier(tol=1e-3, maxIter=10000, layers = [500, 100, 20, 6], blockSize=128, seed=1234),
        'NaiveBayes': NaiveBayes()
    }
    return models_dict[model_string]