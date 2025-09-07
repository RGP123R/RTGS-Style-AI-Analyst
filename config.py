import argparse
import os

# Default paths
DEFAULT_DATASET_FILE = "C:/Users/Hemalatha P/Desktop/techb/consumption_detail_05_2022_temporary_supply.csv"
ARTIFACTS_DIR = "artifacts"
LOG_FILE = os.path.join(ARTIFACTS_DIR, "run_log.txt")
CLEANED_DATA_FILE = os.path.join(ARTIFACTS_DIR, "cleaned_data.csv")
INSIGHTS_DOC = os.path.join(ARTIFACTS_DIR, "insights.md")

def get_config():
    """Parses command-line arguments and returns configuration."""
    parser = argparse.ArgumentParser(description="RTGS Data Analysis Pipeline")
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET_FILE,
                        help='Path to the input CSV dataset file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    # Validate dataset path for security
    if not os.path.isabs(args.dataset):
        args.dataset = os.path.abspath(args.dataset)
    if '..' in args.dataset or not args.dataset.startswith(os.getcwd()):
        raise ValueError("Invalid dataset path: must be within current directory")

    return args

# Global config - only parse if not imported for testing
if __name__ == "__main__":
    config = get_config()
    DATASET_FILE = config.dataset
else:
    # For testing, use defaults
    class MockConfig:
        dataset = DEFAULT_DATASET_FILE
        log_level = 'INFO'
    config = MockConfig()
    DATASET_FILE = config.dataset
