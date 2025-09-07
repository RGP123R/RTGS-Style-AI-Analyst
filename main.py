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

class DataPrepAgent:
    """Agent for cleaning and standardizing the raw data."""
    def __init__(self, logger, df):
        self.logger = logger
        self.df = df

    def _find_columns(self):
        """Finds potential geographic and numeric columns by type and name patterns."""
        geographic_cols = []
        numeric_cols = []
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if column values look like district names (e.g., strings with spaces)
                if self.df[col].nunique() < 50: # Heuristic: If it has few unique string values, it might be a geographic identifier.
                    geographic_cols.append(col)
            elif self.df[col].dtype in ['int64', 'float64']:
                numeric_cols.append(col)
        
        return geographic_cols, numeric_cols

    def execute(self):
        """Performs cleaning and standardization tasks by auto-detecting columns."""
        self.logger.log("CLEANING & STANDARDIZATION: Starting data preparation with auto-detection.")
        
        # Auto-detect key columns
        geographic_cols, numeric_cols = self._find_columns()
        
        if not geographic_cols or not numeric_cols:
            self.logger.log("CLEANING: ERROR: Could not find essential columns for analysis (geographic and/or numeric).")
            return None
            
        self.logger.log(f"CLEANING: Detected geographic columns: {geographic_cols}")
        self.logger.log(f"CLEANING: Detected numeric columns: {numeric_cols}")

        # Use the first detected geographic column as the district name for this prototype
        district_col = geographic_cols[0]
        self.df.rename(columns={district_col: 'district_name'}, inplace=True)
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        self.logger.log("CLEANING: Renamed a geographic column to 'district_name'.")
        
        # 1. Handle duplicates
        if 'district_name' in self.df.columns:
            self.df.drop_duplicates(subset=['district_name'], inplace=True)
            self.logger.log("CLEANING: Dropped duplicate district entries.")

        # 2. Data type conversion for all detected numeric columns
        for col in numeric_cols:
            standardized_col = col.lower().replace(' ', '_')
            if standardized_col in self.df.columns:
                self.df[standardized_col] = pd.to_numeric(self.df[standardized_col], errors='coerce').fillna(0).astype(int)
        self.logger.log("CLEANING: Converted detected numerical columns to integer types.")
        
        self.logger.log(f"CLEANING: Data is now analysis-ready. Final shape: {self.df.shape}")
        
        # Save cleaned data as an artifact
        if not os.path.exists('artifacts'):
            os.makedirs('artifacts')
        self.df.to_csv(CLEANED_DATA_FILE, index=False)
        self.logger.log(f"CLEANING: Cleaned data saved to {CLEANED_DATA_FILE}")

        return self.df

class InsightsAgent:
    """Agent for generating key insights and analysis."""
    def __init__(self, logger, df):
        self.logger = logger
        self.df = df
        self.insights = {}

    def execute(self):
        """Calculates key metrics and stores insights."""
        self.logger.log("INSIGHTS: Beginning data analysis to find patterns and gaps.")
        
        # NOTE: The Insights Agent is still tailored to the education dataset.
        # This is a limitation that would need further work to make it generic.
        
        # Placeholder for flexible insights generation
        self.logger.log("INSIGHTS: Running generic analysis on available numeric columns.")
        
        # Find the top 3 and bottom 3 districts for a sample numeric column
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) > 1 and 'district_name' in self.df.columns:
            # Pick a representative numeric column for a sample insight
            sample_col = numeric_cols[1]
            self.insights['sample_insight'] = {
                'column': sample_col,
                'highest': self.df.sort_values(by=sample_col, ascending=False).head(3)[['district_name', sample_col]].to_dict('records'),
                'lowest': self.df.sort_values(by=sample_col, ascending=True).head(3)[['district_name', sample_col]].to_dict('records')
            }
            self.logger.log(f"INSIGHTS: Generated a sample insight for column '{sample_col}'.")
            
        else:
            self.logger.log("INSIGHTS: Not enough numeric data columns to generate insights.")
        
        self.logger.log("INSIGHTS: All key metrics and insights have been generated.")
        return self.insights

class OutputAgent:
    """Agent for generating CLI and documentation outputs."""
    def __init__(self, logger, insights):
        self.logger = logger
        self.insights = insights

    def execute(self):
        """Formats and prints insights to the CLI and saves to files."""
        self.logger.log("OUTPUT: Formatting and generating CLI and documentation artifacts.")
        
        # 1. Generate CLI Output (ASCII charts/tables)
        print("\n" + "="*80)
        print("REAL-TIME GOVERNANCE SYSTEM (RTGS) - GENERIC INSIGHTS")
        print("="*80)
        
        if 'sample_insight' in self.insights:
            sample_insight = self.insights['sample_insight']
            print(f"\n--- Insights for '{sample_insight['column']}' ---\n")
            
            highest_df = pd.DataFrame(sample_insight['highest'])
            highest_df.columns = ['District', sample_insight['column'].replace('_', ' ').title()]
            print("Districts with the Highest Values:")
            print(tabulate(highest_df, headers='keys', tablefmt='psql', showindex=False))
            
            lowest_df = pd.DataFrame(sample_insight['lowest'])
            lowest_df.columns = ['District', sample_insight['column'].replace('_', ' ').title()]
            print("\nDistricts with the Lowest Values:")
            print(tabulate(lowest_df, headers='keys', tablefmt='psql', showindex=False))
        else:
            print("\nNo insights to display. The system could not find enough relevant data columns.")

        # 2. Generate and save human-readable documentation
        self.logger.log("DOCUMENTATION: Generating human-readable documentation.")
        with open(INSIGHTS_DOC, "w") as f:
            f.write("# RTGS Generic Insights Report\n\n")
            f.write("### Executive Summary\n")
            f.write("This report provides a data-driven overview based on the provided dataset. The system automatically identified key geographic and numeric columns to generate a preliminary analysis, highlighting significant disparities and trends.\n\n")
            f.write("### Key Findings\n\n")
            if 'sample_insight' in self.insights:
                sample_insight = self.insights['sample_insight']
                f.write(f"#### 1. Analysis of '{sample_insight['column']}'\n")
                f.write(f"The analysis of the '{sample_insight['column']}' metric reveals significant variations across districts.\n\n")
                f.write("```\n")
                f.write(tabulate(pd.DataFrame(sample_insight['highest']), headers='keys', tablefmt='psql', showindex=False))
                f.write("\n")
                f.write(tabulate(pd.DataFrame(sample_insight['lowest']), headers='keys', tablefmt='psql', showindex=False))
                f.write("\n```\n")
            else:
                f.write("No insights could be generated due to a lack of identifiable geographic or numeric columns in the provided dataset.\n")

        self.logger.log(f"OUTPUT: All artifacts (logs, cleaned data, insights) saved in the 'artifacts' directory.")
        
def main():
    """Main function to run the sequential agents."""
    logger = Logger(LOG_FILE)
    
    # Run the Ingestion Agent
    ingest_agent = IngestAgent(logger)
    df = ingest_agent.execute()
    if df is None:
        return
    
    # Run the Data Preparation Agent
    prep_agent = DataPrepAgent(logger, df)
    cleaned_df = prep_agent.execute()
    if cleaned_df is None:
        return
    
    # Run the Insights Agent
    insights_agent = InsightsAgent(logger, cleaned_df)
    insights = insights_agent.execute()
    
    # Run the Output Agent
    output_agent = OutputAgent(logger, insights)
    output_agent.execute()

if __name__ == "__main__":
    main()
