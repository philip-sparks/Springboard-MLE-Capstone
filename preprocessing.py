"""
Contains the preprocessing steps of the raw csv files.
"""

# external imports
import pyspark.sql.functions as sqlf
import pyspark.sql.types as sqlt

# local imports
import get_config
import utilities


def parse_message(col, eol="\n"):
    """
    Generate the expression that parses the email message into From, Subject, Body etc.

    Args:
        col - sqlf.col() column object
        eol - end of line chatacter to use when parsing the email
    Returns:
        List pyspark.sql.functions to be passed to select()
    """

    out_dict = [
        "Message-ID",
        "Date",
        "From",
        "To",
        "Subject",
        "Mime-Version",
        "Content-Type",
        "Content-Transfer-Encoding",
        "X-From",
        "X-To",
        "X-cc",
        "X-bcc",
        "X-Folder",
        "X-Origin",
        "X-FileName",
        eol,
    ]
    expr = []
    for i in range(0, len(out_dict) - 1):
        expr.append(
            sqlf.ltrim(
                sqlf.rtrim(sqlf.split(sqlf.split(col, out_dict[i] + ":")[1], eol)[0])
            ).alias(out_dict[i])
        )

    expr.append(sqlf.split(sqlf.split(col, "X-FileName:")[1], "nsf")[1].alias("Body"))

    return expr


def parse_email_csv(spark_in, conf):
    """
    Parse the untagged rows within each line.

    Args:
        spark_in - SparkSession object
        conf (dict) - contains the configurations and constants
    Returns:
        spark dataframe containing the email data
    """

    path = conf["nontagged_emails_raw_path"]

    df_emails_raw = spark_in.read.csv(
        path + "/*", sep=",", quote='"', escape='"', multiLine=True, header=True
    )

    email_info_parsed = df_emails_raw.select(*parse_message(sqlf.col("message")))

    email_info_parsed = email_info_parsed.withColumn("tag", sqlf.lit(""))

    return email_info_parsed


def read_tagged_emails(spark_in, sc_in, conf):
    """
    Read the tagged email data where each email is stored as a separate file.

    Args:
        spark_in - SparkSession object
        sc_in - SparkContext object
        conf (dict) - contains the configurations and constants
    Returns:
        park dataframe containing the email data
    """

    base_path = conf["tagged_emails_raw_path"]

    df = sc_in.wholeTextFiles(base_path + "/*").toDF(["file", "message"])
    for i in range(2, 5):
        df = df.union(sc.wholeTextFiles(base_path + "/*" * i).toDF(["file", "message"]))

    df_parsed = df.select(
        *(parse_message(sqlf.col("message"), eol="\r") + [sqlf.col("file")])
    )

    category_types = conf["category_types"]

    expr = sqlf.when(sqlf.col("file").contains(category_types[0]), category_types[0])
    for cat in category_types[1:]:
        expr = expr.when(sqlf.col("file").contains(cat), cat)

    expr = expr.otherwise(sqlf.lit(None))

    df_parsed = df_parsed.withColumn("tag", expr)

    return df_parsed


if __name__ == "__main__":

    conf = get_config.get_config()

    spark, sc = utilities.get_spark_session("preprocessing")

    emails_parsed_nontagged = parse_email_csv(spark, conf)
    emails_tagged_parsed = read_tagged_emails(spark, sc, conf)
    emails_parsed_nontagged.write.parquet(
        conf["nontagged_emails_pipeline_path"], mode="overwrite"
    )
    emails_tagged_parsed.write.parquet(
        conf["tagged_emails_pipeline_path"], mode="overwrite"
    )
