import os

# To enable S3, set the environment variable USE_S3=true before running.
USE_S3 = os.getenv("USE_S3", "False").lower() == "true"

# Define your S3 bucket URI here
S3_BUCKET_URI = "s3a://pyspark-pipeline-abhi"
