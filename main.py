import pandas as pd
import os
from tabulate import tabulate
from datetime import datetime

LOG_FILE = "artifacts/run_log.txt"
CLEANED_DATA_FILE = "artifacts/cleaned_data.csv"
INSIGHTS_DOC = "artifacts/insights.md"
DATASET_FILE = "/content/consumption_detail_07_2021_domestic.csv"


class Logger:
    """A simple logger to record all agent actions and system outputs."""
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write(f"RTGS Run Log - {datetime.now().isoformat()}\n")
            f.write("----------------------------------------\n\n")

    def log(self, message):
        """Appends a timestamped message to the log file and prints to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_message)
        print(log_message, end="")