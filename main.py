import pandas as pd
import os
from tabulate import tabulate
from datetime import datetime

# --- Agentic Architecture ---
# 1. Ingestion Agent: Loads data from a local file.
# 2. DataPrep Agent: Standardizes, cleans, and transforms the data.
# 3. Insights Agent: Generates key metrics and insights for policymakers.
# 4. Output Agent: Renders CLI-friendly output and saves documentation.

LOG_FILE = "artifacts/run_log.txt"
CLEANED_DATA_FILE = "artifacts/cleaned_data.csv"
INSIGHTS_DOC = "artifacts/insights.md"
DATASET_FILE = "C:/Users/Hemalatha P/Desktop/RTGS/consumption_detail_06_2021_domestic.csv"

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

class IngestAgent:
    """Agent responsible for ingesting the raw data from a local file."""
    def __init__(self, logger):
        self.logger = logger
        self.data_path = DATASET_FILE
        self.raw_data = None

    def execute(self):
        """Loads the dataset from a local CSV file into a pandas DataFrame."""
        self.logger.log(f"INGESTION: Starting data ingestion from local file '{self.data_path}'.")
        if not os.path.exists(self.data_path):
            self.logger.log(f"INGESTION: ERROR: Dataset file not found at '{self.data_path}'.")
            self.logger.log("Please place your target CSV file in the project root directory and update the DATASET_FILE variable.")
            return None
        
        try:
            self.raw_data = pd.read_csv(self.data_path)
            self.logger.log(f"INGESTION: Successfully loaded data. Initial shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            self.logger.log(f"INGESTION: Failed to load data. Error: {e}")
            return None

def main():
    """Main function to run the sequential agents."""
    logger = Logger(LOG_FILE)
    
    # Run the Ingestion Agent
    ingest_agent = IngestAgent(logger)
    df = ingest_agent.execute()
    if df is None:
        return

if __name__ == "__main__":
    main()
