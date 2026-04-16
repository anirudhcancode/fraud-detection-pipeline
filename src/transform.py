from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import os

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FraudDetectionETL") \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

def extract(filepath: str):
    print(f"Reading data from {filepath}...")
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"Extracted {df.count()} rows")
    return df

def transform(df):
    print("Transforming data...")

    # Rename Class column to label
    df = df.withColumnRenamed("Class", "label")

    # Drop nulls
    df = df.dropna()

    # Feature 1: normalize Amount using log transform
    df = df.withColumn("amount_log", F.log1p(F.col("Amount")))

    # Feature 2: transaction hour from Time
    df = df.withColumn("hour", (F.col("Time") / 3600 % 24).cast("int"))

    # Feature 3: rolling transaction count per hour (window)
    window = Window.orderBy("Time").rowsBetween(-100, 0)
    df = df.withColumn("tx_count_rolling", F.count("Time").over(window))

    # Feature 4: rolling average amount
    df = df.withColumn("avg_amount_rolling", F.avg("Amount").over(window))

    # Feature 5: flag high value transactions
    df = df.withColumn("high_value_flag", (F.col("Amount") > 1000).cast("int"))

    print(f"Transformed data has {df.count()} rows and {len(df.columns)} columns")
    return df

def load(df, output_path: str):
    print(f"Writing Parquet to {output_path}...")
    df.write.mode("overwrite").parquet(output_path)
    print("Done writing Parquet")

if __name__ == "__main__":
    input_path = "data/creditcard.csv"
    output_path = "data/processed"

    df = extract(input_path)
    df_transformed = transform(df)
    load(df_transformed, output_path)

    print("ETL pipeline complete!")
    spark.stop()