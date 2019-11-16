"""
Contains the codes that load a model from S3 and allow for classification
of a single data point
"""

# external imports
import pyspark.sql.functions as sqlf
import pyspark.sql.types as sqlt
from pyspark.ml import PipelineModel
from pyspark.ml.feature import (
    RegexTokenizer,
    StopWordsRemover,
    CountVectorizer,
    OneHotEncoder,
    StringIndexer,
    VectorAssembler,
)

# local imports
import utilities
import get_config


class StoredModelPredictor(object):
    """
    Classify an email without the need to initialize SparkSession and
    load the model 
    """

    def __init__(self):
        """
        Load the model and read the config
        """

        self.conf = get_config.get_config()

        self.spark, self.sc = utilities.get_spark_session("classification_app")
        self.pipelineModel = PipelineModel.load(self.conf["model_path"])
        self.label_list = self.pipelineModel.stages[3].labels

    def predict(self, data_in):
        """
        Predict the class of an email
        
        Args:
            data_in (dict) - A dict containing the key 'Body'
        Returns:
            Spark ML prediction, probabilities for different classes and the original email
        """

        df = self.sc.parallelize([data_in]).toDF()

        pred = self.pipelineModel.transform(df).select(
            "Body", "probability", "prediction"
        )

        return [
            {
                "Body": Row[0],
                "probability": list(Row[1]),
                "prediction": self.label_list[int(Row[2])],
            }
            for Row in pred.collect()
        ]
