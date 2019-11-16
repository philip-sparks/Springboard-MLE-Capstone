"""
Contains several unit tests that validate critical functions
"""

# standard imports
import unittest

# external imports
import pyspark.sql.functions as sqlf

# local imports
import utilities
import preprocessing
import train


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_str = """
            Date: Mon, 23 Oct 2000 06:13:00 -0700 (PDT)
            From: phillip.allen@enron.com
            To: randall.gay@enron.com
            Subject: test email
            Mime-Version: 1.0
            Content-Type: text/plain; charset=us-ascii
            Content-Transfer-Encoding: 7bit
            X-From: Phillip K Allen
            X-To: Randall L Gay
            X-cc: 
            X-bcc: 
            X-Folder: \Phillip_Allen_Dec2000Notes Folders\'sent mail
            X-Origin: Allen-P
            X-FileName: pallen.nsf

            Hi, this is a test
            """

    def test_preprocessing(self):

        my_col = sqlf.col("message")

        df1 = sc.parallelize([{"message": self.test_str}]).toDF()

        exp = preprocessing.parse_message(my_col)

        df2 = df1.select(exp)

        self.assertEqual(
            df2.select("Date").collect()[0][0], "Mon, 23 Oct 2000 06:13:00 -0700 (PDT)"
        )

        self.assertEqual(df2.select("From").collect()[0][0], "phillip.allen@enron.com")

        self.assertEqual(df2.select("To").collect()[0][0], "randall.gay@enron.com")

        self.assertEqual(df2.select("Subject").collect()[0][0], "test email")

        self.assertEqual(df2.select("Mime-Version").collect()[0][0], "1.0")

        self.assertEqual(
            df2.select("Content-Type").collect()[0][0], "text/plain; charset=us-ascii"
        )

        self.assertEqual(
            df2.select("Content-Transfer-Encoding").collect()[0][0], "7bit"
        )

        self.assertEqual(df2.select("X-From").collect()[0][0], "Phillip K Allen")

        self.assertEqual(df2.select("X-To").collect()[0][0], "Randall L Gay")

        self.assertEqual(df2.select("X-cc").collect()[0][0], "")

        self.assertEqual(df2.select("X-bcc").collect()[0][0], "")

        self.assertEqual(
            df2.select("X-Folder").collect()[0][0],
            "\Phillip_Allen_Dec2000Notes Folders'sent mail",
        )

        self.assertEqual(df2.select("X-Origin").collect()[0][0], "Allen-P")

        self.assertEqual(df2.select("X-FileName").collect()[0][0], "pallen.nsf")

        self.assertEqual(
            df2.select("Body").collect()[0][0],
            "\n\n            Hi, this is a test\n            ",
        )


class TestTraining(unittest.TestCase):
    def setUp(self):
        self.data_dict_train = [
            {"Body": "apple is great", "tag": "apple"},
            {"Body": "apple is fruit", "tag": "apple"},
            {"Body": "apple is red", "tag": "apple"},
            {"Body": "pear is great", "tag": "pear"},
            {"Body": "pear is fruit", "tag": "pear"},
            {"Body": "pear is yellow", "tag": "pear"},
        ]

        self.data_dict_test = [
            {"Body": "What fruit is apple", "tag": "apple"},
            {"Body": "What fruit is pear", "tag": "pear"},
        ]

    def test_training(self):

        df_train = sc.parallelize(self.data_dict_train).toDF()

        model = train.train_model(df_train)
        label_list = model.stages[3].labels
        df_test = sc.parallelize(self.data_dict_test).toDF()
        preds = model.transform(df_test)

        self.assertEqual(
            label_list[int(preds.select("label").collect()[0][0])],
            self.data_dict_test[0]["tag"],
        )
        self.assertEqual(
            label_list[int(preds.select("label").collect()[0][0])],
            self.data_dict_test[0]["tag"],
        )


if __name__ == "__main__":
    spark, sc = utilities.get_spark_session("unittests")

    unittest.main()
