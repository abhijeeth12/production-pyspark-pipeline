import os
import time

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_logger(step_name):
    """
    Returns a simple logger function for the specified pipeline step.
    The log writes both to console and to output/<step_name>_log.txt.
    """
    log_file_path = os.path.join(OUTPUT_DIR, f"{step_name}_log.txt")
    
    def log_function(msg):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        # We only guarantee local filesystem logging to keep it simple,
        # but could upload these to S3 at the end of the script if requested.
        with open(log_file_path, "a") as f:
            f.write(line + "\n")
            
    return log_function
