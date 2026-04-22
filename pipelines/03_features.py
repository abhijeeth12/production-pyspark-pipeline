#!/usr/bin/env python3
"""
Script 03: Feature Engineering
- Extract hour/period-of-day from CRSDepTime
- Create route feature (Origin + Dest)
- Encode seasonal features
- Build MLlib feature vector using StringIndexer + OneHotEncoder + VectorAssembler
- Save transformed Parquet (with label + features column)
"""
import os, sys, json, time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from src.io_utils import get_base_path, configure_spark_for_s3
from src.logger import get_logger
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    FeatureHasher, VectorAssembler, StandardScaler
)

PROC_DIR = get_base_path("processed")
OUTPUT_DIR = get_base_path("output")

log = get_logger("03_features")

def main():
    log("=" * 70)
    log("STEP 03 — Feature Engineering")
    log("=" * 70)

    builder = SparkSession.builder \
        .appName("FlightDelay-Features") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.enabled", "false") \
        
    spark = configure_spark_for_s3(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    clean_path = os.path.join(PROC_DIR, "flights_clean.parquet")
    log(f"Loading cleaned data from {clean_path}...")
    df = spark.read.parquet(clean_path)
    log(f"Rows: {df.count():,}  Columns: {df.columns}")

    # ── New derived features ──────────────────────────────────────────────

    # 1. Departure hour (0–23)
    df = df.withColumn("DepHour", (F.col("CRSDepTime") / 100).cast("int"))

    # 2. Period of day (categorical)
    df = df.withColumn("DayPeriod",
        F.when(F.col("DepHour").between(5, 11), "Morning")
         .when(F.col("DepHour").between(12, 16), "Afternoon")
         .when(F.col("DepHour").between(17, 20), "Evening")
         .otherwise("Night"))

    # 3. Is Weekend?
    df = df.withColumn("IsWeekend",
        F.when(F.col("DayOfWeek").isin(6, 7), 1).otherwise(0))

    # 4. Route = Origin + Dest concatenated
    df = df.withColumn("Route",
        F.concat_ws("-", F.col("Origin"), F.col("Dest")))

    # 5. Season from Month
    df = df.withColumn("Season",
        F.when(F.col("Month").isin(12, 1, 2), "Winter")
         .when(F.col("Month").isin(3, 4, 5), "Spring")
         .when(F.col("Month").isin(6, 7, 8), "Summer")
         .otherwise("Fall"))

    # 6. Arr hour
    df = df.withColumn("ArrHour", (F.col("CRSArrTime") / 100).cast("int"))

    # 7. Is long-haul (> 1500 miles)?
    df = df.withColumn("IsLongHaul",
        F.when(F.col("Distance") > 1500, 1).otherwise(0))

    log(f"Derived features added. Schema: {[f.name for f in df.schema]}")

    # ── String columns to index ───────────────────────────────────────────
    cat_cols = ["Reporting_Airline", "Origin", "Dest", "Route",
                "DayPeriod", "Season"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    # Numerical feature columns
    num_cols = ["Month", "DayofMonth", "DayOfWeek", "DepHour", "ArrHour",
                "CRSElapsedTime", "Distance", "IsWeekend", "IsLongHaul"]
    if "DistanceGroup" in df.columns:
        num_cols.append("DistanceGroup")
    num_cols = [c for c in num_cols if c in df.columns]

    log(f"Categorical cols: {cat_cols}")
    log(f"Numerical cols:   {num_cols}")

    hasher = FeatureHasher(inputCols=cat_cols, outputCol="cat_features", numFeatures=300)
    assembler  = VectorAssembler(inputCols=num_cols + ["cat_features"],
                                 outputCol="features_raw",
                                 handleInvalid="keep")
    scaler     = StandardScaler(inputCol="features_raw", outputCol="features",
                                withStd=True, withMean=False)

    pipeline = Pipeline(stages=[hasher, assembler, scaler])
    log("Fitting feature pipeline...")
    model = pipeline.fit(df)
    log("Pipeline fitted. Transforming dataset...")
    df_feat = model.transform(df)

    # Keep only what we need for training
    df_final = df_feat.select("features", F.col("ArrDel15").alias("label"))
    df_final = df_final.withColumn("label", F.col("label").cast("double"))

    log(f"Feature dataset: {df_final.count():,} rows")
    log(f"Feature vector size: {df_final.first()['features'].size}")

    # ── Save ─────────────────────────────────────────────────────────────
    feat_path = os.path.join(PROC_DIR, "flights_features.parquet")
    df_final.write.mode("overwrite").parquet(feat_path)
    log(f"Feature data saved to {feat_path}")

    # Save feature names for report
    feature_names = num_cols + ["hashed_categorical"] * 300
    with open(os.path.join(get_base_path("output"), "03_feature_names.json"), "w") as f:
        json.dump({"numerical": num_cols, "categorical_hashed": cat_cols,
                   "all_features": feature_names}, f, indent=2)

    spark.stop()
    log("Step 03 complete.\n")

if __name__ == "__main__":
    main()
