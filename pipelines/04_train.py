#!/usr/bin/env python3
"""
Script 04: Model Training
- Trains two Spark MLlib classifiers:
    1. Logistic Regression (baseline)
    2. Random Forest Classifier (primary model)
    3. Gradient Boosted Trees (advanced model)
- Uses 80/20 train/test split (stratified by label)
- Saves trained models to output/models/
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
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)

PROC_DIR = get_base_path("processed")
OUTPUT_DIR = get_base_path("output")
MODELS_DIR  = os.path.join(get_base_path("output"), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

log = get_logger("04_train")

def main():
    log("=" * 70)
    log("STEP 04 — Model Training")
    log("=" * 70)

    builder = SparkSession.builder \
        .appName("FlightDelay-Train") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.enabled", "false") \
        
    spark = configure_spark_for_s3(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    feat_path = os.path.join(PROC_DIR, "flights_features.parquet")
    log(f"Loading feature data from {feat_path}...")
    df = spark.read.parquet(feat_path)
    total = df.count()
    log(f"Total samples: {total:,}")

    # ── Train / Test split ───────────────────────────────────────────────
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # ── Handle Imbalance (Undersampling) ──────────────────────────────────
    major_df = train_df.filter(F.col("label") == 0)
    minor_df = train_df.filter(F.col("label") == 1)
    maj_ct = major_df.count()
    min_ct = minor_df.count()
    ratio = min(1.0, float(min_ct) / float(max(1, maj_ct)))
    train_df = major_df.sample(False, ratio, seed=42).unionAll(minor_df)

    train_count = train_df.count()
    test_count  = test_df.count()
    log(f"Train (balanced): {train_count:,}  Test: {test_count:,}")

    # Cache for repeated use
    train_df.cache()
    test_df.cache()

    results = {}

    # ── Model 1: Logistic Regression (baseline) ───────────────────────────
    log("\n--- Training Logistic Regression (baseline) ---")
    t0 = time.time()
    lr = LogisticRegression(maxIter=20, regParam=0.01, elasticNetParam=0.0,
                            featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_df)
    lr_time  = time.time() - t0
    log(f"  LR trained in {lr_time:.1f}s")

    lr_path = os.path.join(MODELS_DIR, "logistic_regression")
    lr_model.write().overwrite().save(lr_path)
    log(f"  LR model saved to {lr_path}")
    results["logistic_regression"] = {"training_time_sec": round(lr_time, 2)}

    # ── Model 2: Random Forest ────────────────────────────────────────────
    log("\n--- Training Random Forest Classifier ---")
    t0 = time.time()
    rf = RandomForestClassifier(
        numTrees=50, maxDepth=10, minInstancesPerNode=5, seed=42,
        featuresCol="features", labelCol="label",
        featureSubsetStrategy="sqrt"
    )
    rf_model = rf.fit(train_df)
    rf_time  = time.time() - t0
    log(f"  RF trained in {rf_time:.1f}s")

    rf_path = os.path.join(MODELS_DIR, "random_forest")
    rf_model.write().overwrite().save(rf_path)
    log(f"  RF model saved to {rf_path}")
    results["random_forest"] = {"training_time_sec": round(rf_time, 2)}

    # ── Model 3: GBT ─────────────────────────────────────────────────────
    log("\n--- Training Gradient Boosted Trees ---")
    t0 = time.time()
    gbt = GBTClassifier(
        maxIter=3, maxDepth=4, stepSize=0.1, seed=42,
        featuresCol="features", labelCol="label"
    )
    gbt_model = gbt.fit(train_df)
    gbt_time  = time.time() - t0
    log(f"  GBT trained in {gbt_time:.1f}s")

    gbt_path = os.path.join(MODELS_DIR, "gbt")
    gbt_model.write().overwrite().save(gbt_path)
    log(f"  GBT model saved to {gbt_path}")
    results["gbt"] = {"training_time_sec": round(gbt_time, 2)}

    # ── Save test set for evaluation ──────────────────────────────────────
    test_path = os.path.join(PROC_DIR, "test_set.parquet")
    test_df.write.mode("overwrite").parquet(test_path)
    log(f"\nTest set saved to {test_path}")

    # Save training results
    with open(os.path.join(get_base_path("output"), "04_train_results.json"), "w") as f:
        json.dump({
            "train_count": train_count,
            "test_count": test_count,
            "models": results
        }, f, indent=2)

    train_df.unpersist()
    test_df.unpersist()
    spark.stop()
    log("Step 04 complete.\n")

if __name__ == "__main__":
    main()
