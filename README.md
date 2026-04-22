# Flight Delay Prediction (PySpark Pipeline)

This project demonstrates a fully functional Big Data Analytics pipeline using **Apache Spark 3.5.1 (PySpark)** to predict commercial flight delays. It processes the 2015 US Domestic Flight On-Time Performance dataset and generates a comprehensive PDF report analyzing model performances.

The project is structured with production-level principles in mind, allowing the pipeline to seamlessly switch between local execution and cloud execution via AWS S3. It also includes a GitHub Actions Continuous Integration (CI) pipeline.

## Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI for pipeline validation
├── configs/
│   └── pipeline_config.py     # Configuration (S3 toggles, Paths)
├── data/                      # Local datasets
│   ├── raw/                   # Raw CSVs should be placed here
│   └── processed/             # Clean Parquet files are generated here
├── pipelines/
│   ├── 00_download_data.py    # Fetches data from BTS
│   ├── 01_ingest.py           # Ingestion into Spark
│   ├── 02_preprocess.py       # Data cleaning
│   ├── 03_features.py         # Feature engineering (One-hot, Scaling)
│   ├── 04_train.py            # Train Models (LogisticRegression, RandomForest, GBT)
│   ├── 05_evaluate.py         # Metrics calculations
│   └── 06_report.py           # Auto-generates the final PDF report
├── src/
│   ├── io_utils.py            # I/O utilities supporting transparent Local vs S3 switching
│   └── logger.py              # Centralized cross-pipeline logging
├── requirements.txt           # Minimal pip dependencies
└── README.md                  # Project documentation
```

## How to Run Locally

If you have Python installed and the `.venv` activated alongside the project-specific Java bundle, you can simply stream through the scripts:

```bash
# 1. Ensure you have the virtual env ready
source .venv/bin/activate
pip install -r requirements.txt

# 2. Add raw data (if you don't use 00_download_data.py, put CSVs in data/raw)
python pipelines/00_download_data.py

# 3. Run the pipeline sequentially
python pipelines/01_ingest.py
python pipelines/02_preprocess.py
python pipelines/03_features.py
python pipelines/04_train.py
python pipelines/05_evaluate.py
python pipelines/06_report.py
```

The output artifacts, model checkpoints, and datasets will reside in the `output/`, `report/`, and `data/` directories respectively. A local log detailing each step will be generated in `output/`.

## Enabling AWS S3 Storage

This pipeline supports direct operations on AWS S3, seamlessly pulling raw CSVs, writing parquets, and storing artifacts directly to a bucket without needing intermediate local files.

To enable S3 integration, update your bucket location in `configs/pipeline_config.py` (or inject an environment variable) and supply your AWS credentials:

```bash
export USE_S3="True"
export S3_BUCKET_URI="s3a://your-bucket-name/flight-project"
export AWS_ACCESS_KEY_ID="<your-access-key>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"

python pipelines/01_ingest.py
```

## CI Pipeline Integration

A basic GitHub Actions configuration exists in `.github/workflows/ci.yml`. Whenever code is pushed to `main`, the action ensures syntax correctness using `flake8` and dry-runs the configuration loader. This is a lightweight validation step simulating a real-world repository checking pipeline integrity before deploying code to an isolated cluster.
# production-pyspark-pipeline
