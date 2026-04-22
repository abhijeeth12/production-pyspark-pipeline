#!/usr/bin/env python3
"""
Script 02: Data Preprocessing & Cleaning
- Drop cancelled/diverted flights (not predictable at departure time)  
- Filter to key columns relevant to delay prediction
- Handle nulls, outliers, and data type issues
- Save cleaned Parquet
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
from pyspark.sql.types import IntegerType, DoubleType

PROC_DIR = get_base_path("processed")
OUTPUT_DIR = get_base_path("output")

log = get_logger("02_preprocess")

# Columns we will KEEP for model training  
# (all knowable at or before departure time, except target)
FEATURE_COLS_RAW = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline",      # may be UniqueCarrier in older files
    "Origin", "Dest",
    "CRSDepTime",             # scheduled dep time (HHMM)
    "CRSArrTime",             # scheduled arr time
    "CRSElapsedTime",         # scheduled elapsed time
    "Distance",
    "DistanceGroup",
]
TARGET_COL = "ArrDel15"       # 1 = arrival delayed ≥ 15 min, 0 = on-time

def main():
    log("=" * 70)
    log("STEP 02 — Data Preprocessing")
    log("=" * 70)

    builder = SparkSession.builder \
        .appName("FlightDelay-Preprocess") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.enabled", "false") \
        
    spark = configure_spark_for_s3(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    raw_path = os.path.join(PROC_DIR, "flights_raw.parquet")
    log(f"Loading raw Parquet from {raw_path}...")
    df = spark.read.parquet(raw_path)
    log(f"Raw rows: {df.count():,}")

    # ── Standardise carrier column name ──────────────────────────────────
    if "Reporting_Airline" not in df.columns and "UniqueCarrier" in df.columns:
        df = df.withColumnRenamed("UniqueCarrier", "Reporting_Airline")

    # ── Step 1: Drop cancelled / diverted flights ─────────────────────────
    if "Cancelled" in df.columns:
        before = df.count()
        df = df.filter(F.col("Cancelled") == 0)
        log(f"After dropping cancelled: {df.count():,} (removed {before-df.count():,})")
    if "Diverted" in df.columns:
        before = df.count()
        df = df.filter(F.col("Diverted") == 0)
        log(f"After dropping diverted:  {df.count():,} (removed {before-df.count():,})")

    # ── Step 2: Drop rows where target is null ────────────────────────────
    before = df.count()
    df = df.filter(F.col(TARGET_COL).isNotNull())
    log(f"After dropping null target: {df.count():,} (removed {before-df.count():,})")

    # ── Step 3: Select relevant columns ──────────────────────────────────
    available_features = [c for c in FEATURE_COLS_RAW if c in df.columns]
    all_cols = available_features + [TARGET_COL]
    df = df.select(*all_cols)
    log(f"Selected {len(all_cols)} columns: {all_cols}")

    # ── Step 4: Cast types ────────────────────────────────────────────────
    int_cols = ["Year","Quarter","Month","DayofMonth","DayOfWeek",
                "CRSDepTime","CRSArrTime","DistanceGroup"]
    dbl_cols = ["CRSElapsedTime","Distance"]
    for c in int_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(IntegerType()))
    for c in dbl_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
    df = df.withColumn(TARGET_COL, F.col(TARGET_COL).cast(IntegerType()))

    # ── Step 5: Drop remaining nulls ──────────────────────────────────────
    before = df.count()
    df = df.dropna()
    after  = df.count()
    log(f"After dropna: {after:,} rows (removed {before-after:,})")

    # ── Step 6: Filter outliers ───────────────────────────────────────────
    if "Distance" in df.columns:
        df = df.filter((F.col("Distance") > 0) & (F.col("Distance") < 5000))
    if "CRSElapsedTime" in df.columns:
        df = df.filter((F.col("CRSElapsedTime") > 0) & (F.col("CRSElapsedTime") < 1000))

    final_count = df.count()
    log(f"Final clean dataset: {final_count:,} rows")

    # ── Step 7: Class balance after cleaning ──────────────────────────────
    dist = df.groupBy(TARGET_COL).count().orderBy(TARGET_COL).collect()
    for row in dist:
        pct = row["count"] * 100 / final_count
        label = "delayed" if row[TARGET_COL] == 1 else "on-time"
        log(f"  {label} ({row[TARGET_COL]}): {row['count']:,} ({pct:.1f}%)")

    # ── Save cleaned parquet ──────────────────────────────────────────────
    clean_path = os.path.join(PROC_DIR, "flights_clean.parquet")
    df.write.mode("overwrite").parquet(clean_path)
    log(f"\nCleaned data saved to {clean_path}")

    # ── Save stats ────────────────────────────────────────────────────────
    stats = {
        "final_rows": final_count,
        "feature_columns": available_features,
        "target": TARGET_COL,
        "class_distribution": {str(row[TARGET_COL]): row["count"] for row in dist}
    }
    with open(os.path.join(get_base_path("output"), "02_preprocessing_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    spark.stop()
    log("Step 02 complete.\n")

if __name__ == "__main__":
    main()
