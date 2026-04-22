#!/usr/bin/env python3
"""
Script 01: Data Ingestion & Exploration
Loads the flight CSV files into Spark, performs initial exploration,
and saves a summary to output/.
"""
import os, sys, json, time

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-17-openjdk")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.io_utils import get_base_path, configure_spark_for_s3
from src.logger import get_logger

DATA_DIR   = get_base_path("raw")
PROC_DIR   = get_base_path("processed")
OUTPUT_DIR = get_base_path("output")

log = get_logger("01_ingest")

def main():
    log("=" * 70)
    log("STEP 01 — Data Ingestion & Exploration")
    log("=" * 70)

    # ── Spark Session ─────────────────────────────────────────────────────
    builder = SparkSession.builder \
        .appName("FlightDelay-Ingest") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.enabled", "false")
    spark = configure_spark_for_s3(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Spark version: {spark.version}")

    # ── Load CSVs ─────────────────────────────────────────────────────────
    if DATA_DIR.startswith("s3a://"):
        csv_files = [
            f"{DATA_DIR}/flights_2015_01.csv",
            f"{DATA_DIR}/flights_2015_02.csv",
            f"{DATA_DIR}/flights_2015_03.csv"
        ]
    else:
        csv_files = sorted([
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.endswith(".csv")
        ])
    if not csv_files:
        log("ERROR: No CSV files found in data/raw/")
        spark.stop()
        sys.exit(1)

    log(f"Found {len(csv_files)} CSV file(s): {[os.path.basename(f) for f in csv_files]}")

    df = spark.read.csv(csv_files, header=True, inferSchema=True, mode="PERMISSIVE")
    total_rows = df.count()
    total_cols = len(df.columns)
    log(f"Dataset shape: {total_rows:,} rows × {total_cols} columns")

    # ── Schema ────────────────────────────────────────────────────────────
    log("\n--- Schema ---")
    schema_lines = []
    for field in df.schema.fields:
        line = f"  {field.name:<40} {str(field.dataType)}"
        print(line)
        schema_lines.append(line)

    # ── Basic stats ───────────────────────────────────────────────────────
    # Key columns: ArrDelay, DepDelay, ArrDel15, Cancelled, etc.
    key_cols = ["Year", "Month", "DayofMonth", "DayOfWeek",
                "ArrDel15", "DepDel15", "ArrDelay", "DepDelay",
                "Distance", "AirTime", "ActualElapsedTime",
                "Cancelled", "Diverted"]
    key_cols = [c for c in key_cols if c in df.columns]
    log(f"\nKey columns available: {key_cols}")

    # Count nulls
    log("\n--- Null Counts (key columns) ---")
    null_counts = {}
    for col in key_cols:
        nc = df.filter(F.col(col).isNull()).count()
        null_counts[col] = nc
        log(f"  {col:<30}: {nc:,} nulls ({nc*100/total_rows:.2f}%)")

    # Class distribution
    if "ArrDel15" in df.columns:
        log("\n--- Target (ArrDel15) Distribution ---")
        dist = df.groupBy("ArrDel15").count().orderBy("ArrDel15").collect()
        for row in dist:
            pct = row["count"] * 100 / total_rows
            log(f"  ArrDel15={row['ArrDel15']}: {row['count']:,} ({pct:.1f}%)")

    # Monthly distribution
    if "Month" in df.columns:
        log("\n--- Monthly Distribution ---")
        monthly = df.groupBy("Month").count().orderBy("Month").collect()
        for row in monthly:
            log(f"  Month {row['Month']:2}: {row['count']:,} flights")

    # Top carriers
    if "Reporting_Airline" in df.columns or "UniqueCarrier" in df.columns:
        carrier_col = "Reporting_Airline" if "Reporting_Airline" in df.columns else "UniqueCarrier"
        log(f"\n--- Top 10 Airlines (by {carrier_col}) ---")
        carriers = df.groupBy(carrier_col).count().orderBy(F.desc("count")).limit(10).collect()
        for row in carriers:
            log(f"  {row[carrier_col]}: {row['count']:,}")

    # Save summary JSON
    summary = {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "csv_files": [os.path.basename(f) for f in csv_files],
        "key_columns": key_cols,
        "null_counts": null_counts,
        "schema": {f.name: str(f.dataType) for f in df.schema.fields}
    }
    summary_path = os.path.join(OUTPUT_DIR, "01_dataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nSummary saved to {summary_path}")

    # Save raw Parquet for faster downstream processing
    parquet_path = os.path.join(PROC_DIR, "flights_raw.parquet")
    log(f"\nSaving raw data as Parquet to {parquet_path}...")
    df.write.mode("overwrite").parquet(parquet_path)
    log("Parquet saved.")

    spark.stop()
    log("Step 01 complete.\n")

if __name__ == "__main__":
    main()
