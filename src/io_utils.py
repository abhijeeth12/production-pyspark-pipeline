import os

# Append configs folder dynamically if needed, or assume running from project root
try:
    from configs.pipeline_config import USE_S3, S3_BUCKET_URI
except ModuleNotFoundError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs.pipeline_config import USE_S3, S3_BUCKET_URI

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_base_path(folder_type="raw"):
    """
    Returns local path or S3 URI based on configuration.
    folder_type can be 'raw', 'processed', 'output', 'models'.
    """
    if USE_S3:
        return f"{S3_BUCKET_URI}/data/{folder_type}"
    
    path = os.path.join(PROJECT_DIR, "data", folder_type)
    if folder_type in ["output", "report"]:
        path = os.path.join(PROJECT_DIR, folder_type)
        
    os.makedirs(path, exist_ok=True)
    return path

def configure_spark_for_s3(spark_builder):
    if not USE_S3:
        return spark_builder

    spark_builder = spark_builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4") \
        .config("spark.hadoop.fs.s3a.access.key", os.getenv("AWS_ACCESS_KEY_ID", "")) \
        .config("spark.hadoop.fs.s3a.secret.key", os.getenv("AWS_SECRET_ACCESS_KEY", "")) \
        .config("spark.hadoop.fs.s3a.endpoint", "s3.ap-south-1.amazonaws.com") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.timeout", "60000") \
        .config("spark.hadoop.fs.s3a.socket.timeout", "60000") \
        .config("spark.hadoop.fs.s3a.connection.maximum", "50") \
        .config("spark.hadoop.fs.s3a.attempts.maximum", "3")

    return spark_builder
