import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import re
import numpy as np

from config import DATASET_FILE, CLEANED_DATA_FILE, INSIGHTS_DOC, LOG_FILE

class IngestAgent:
    """Agent responsible for ingesting the raw data from a local file."""
    def __init__(self, logger: logging.Logger, data_path: str = DATASET_FILE):
        self.logger = logger
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None

    def execute(self) -> Optional[pd.DataFrame]:
        """Loads the dataset from a local CSV file into a pandas DataFrame."""
        self.logger.info(f"INGESTION: Starting data ingestion from local file '{self.data_path}'.")
        if not os.path.exists(self.data_path):
            self.logger.error(f"INGESTION: Dataset file not found at '{self.data_path}'.")
            self.logger.error("Please place your target CSV file in the project root directory or update the DATASET_FILE variable.")
            return None

        try:
            # For large files, chunked reading can be implemented here if needed
            self.raw_data = pd.read_csv(self.data_path)
            self.logger.info(f"INGESTION: Successfully loaded data. Initial shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            self.logger.error(f"INGESTION: Failed to load data. Error: {e}")
            return None

class DataPrepAgent:
    """Comprehensive agent for advanced data cleaning and preprocessing using Pandas, NumPy, Scikit-learn, and re."""
    def __init__(self, logger: logging.Logger, df: pd.DataFrame):
        self.logger = logger
        self.df = df.copy()  # Work on a copy to preserve original data
        self.original_shape = df.shape

    def _identify_missing_values(self) -> Dict[str, int]:
        """Identify missing values per column using isnull() or isna()."""
        self.logger.info("CLEANING: Identifying missing values per column.")
        missing_counts = self.df.isnull().sum() + self.df.isna().sum()
        missing_dict = missing_counts[missing_counts > 0].to_dict()

        for col, count in missing_dict.items():
            self.logger.info(f"CLEANING: Column '{col}' has {count} missing values ({count/len(self.df)*100:.1f}%)")

        return missing_dict

    def _impute_missing_values(self, strategy: str = 'auto'):
        """Advanced missing value imputation using multiple strategies."""
        self.logger.info(f"CLEANING: Imputing missing values using '{strategy}' strategy.")

        # For numeric columns
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                if strategy == 'mean':
                    fill_val = self.df[col].mean()
                elif strategy == 'median':
                    fill_val = self.df[col].median()
                elif strategy == 'mode':
                    mode_val = self.df[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else 0
                else:  # auto strategy
                    # Use median for skewed data, mean for normal
                    if abs(self.df[col].skew()) > 1:
                        fill_val = self.df[col].median()
                    else:
                        fill_val = self.df[col].mean()

                self.df[col].fillna(fill_val, inplace=True)
                self.logger.info(f"CLEANING: Filled {missing_count} missing values in '{col}' with {fill_val:.2f}")

        # For categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                mode_val = self.df[col].mode()
                fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                self.df[col].fillna(fill_val, inplace=True)
                self.logger.info(f"CLEANING: Filled {missing_count} missing values in '{col}' with '{fill_val}'")

    def _remove_missing_data(self, threshold: float = 0.5):
        """Remove rows or columns with excessive missing data."""
        self.logger.info(f"CLEANING: Removing data with >{threshold*100}% missing values.")

        # Remove columns with too many missing values
        col_missing_ratio = self.df.isnull().mean()
        cols_to_drop = col_missing_ratio[col_missing_ratio > threshold].index.tolist()
        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.logger.info(f"CLEANING: Dropped columns with excessive missing data: {cols_to_drop}")

        # Remove rows with too many missing values
        row_missing_ratio = self.df.isnull().mean(axis=1)
        rows_to_drop = row_missing_ratio[row_missing_ratio > threshold].index
        if len(rows_to_drop) > 0:
            self.df.drop(index=rows_to_drop, inplace=True)
            self.logger.info(f"CLEANING: Dropped {len(rows_to_drop)} rows with excessive missing data")

    def _identify_duplicates(self) -> int:
        """Identify duplicate rows using duplicated()."""
        self.logger.info("CLEANING: Identifying duplicate rows.")
        duplicate_count = self.df.duplicated().sum()
        self.logger.info(f"CLEANING: Found {duplicate_count} duplicate rows")
        return duplicate_count

    def _remove_duplicates(self, subset: Optional[List[str]] = None):
        """Remove duplicate entries using drop_duplicates()."""
        initial_count = len(self.df)
        self.df.drop_duplicates(subset=subset, inplace=True, keep='first')
        removed = initial_count - len(self.df)
        if removed > 0:
            self.logger.info(f"CLEANING: Removed {removed} duplicate rows")

    def _standardize_text(self):
        """Standardize text data using string methods and regular expressions."""
        self.logger.info("CLEANING: Standardizing text data.")

        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in text_cols:
            # Convert to string type first
            self.df[col] = self.df[col].astype(str)

            # Standardize case (convert to title case for names, lower for others)
            if 'name' in col.lower():
                self.df[col] = self.df[col].str.title()
            else:
                self.df[col] = self.df[col].str.lower()

            # Remove extra whitespace
            self.df[col] = self.df[col].str.strip()

            # Remove multiple spaces
            self.df[col] = self.df[col].apply(lambda x: re.sub(r'\s+', ' ', x))

            # Remove special characters (keep only alphanumeric, spaces, and basic punctuation)
            self.df[col] = self.df[col].apply(lambda x: re.sub(r'[^\w\s\.\-\(\)]', '', x))

            self.logger.info(f"CLEANING: Standardized text in column '{col}'")

    def _convert_data_types(self):
        """Convert columns to appropriate data types using astype() and to_datetime()."""
        self.logger.info("CLEANING: Converting data types.")

        for col in self.df.columns:
            # Try to convert to numeric
            try:
                # Check if column looks like it should be numeric
                if self.df[col].dtype == 'object':
                    # Try converting to numeric
                    temp_series = pd.to_numeric(self.df[col], errors='coerce')
                    if temp_series.notna().sum() > len(self.df) * 0.8:  # If >80% can be converted
                        self.df[col] = temp_series
                        self.logger.info(f"CLEANING: Converted '{col}' to numeric type")
            except:
                pass

            # Try to convert to datetime
            try:
                if self.df[col].dtype == 'object':
                    # Check for date-like patterns
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'year']):
                        temp_series = pd.to_datetime(self.df[col], errors='coerce')
                        if temp_series.notna().sum() > len(self.df) * 0.8:
                            self.df[col] = temp_series
                            self.logger.info(f"CLEANING: Converted '{col}' to datetime type")
            except:
                pass

    def _detect_outliers_zscore(self, threshold: float = 3.0) -> Dict[str, int]:
        """Detect outliers using Z-score method."""
        from scipy import stats
        import numpy as np

        self.logger.info(f"CLEANING: Detecting outliers using Z-score (threshold={threshold})")
        outlier_counts = {}

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outliers = (z_scores > threshold).sum()
            outlier_counts[col] = outliers
            if outliers > 0:
                self.logger.info(f"CLEANING: Found {outliers} outliers in '{col}' using Z-score")

        return outlier_counts

    def _detect_outliers_iqr(self) -> Dict[str, int]:
        """Detect outliers using IQR method."""
        self.logger.info("CLEANING: Detecting outliers using IQR method")
        outlier_counts = {}

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
            if outliers > 0:
                self.logger.info(f"CLEANING: Found {outliers} outliers in '{col}' using IQR")

        return outlier_counts

    def _handle_outliers(self, method: str = 'cap', detection_method: str = 'iqr'):
        """Handle outliers by removing, capping, or transforming."""
        self.logger.info(f"CLEANING: Handling outliers using {detection_method} detection and {method} method")

        if detection_method == 'iqr':
            outlier_counts = self._detect_outliers_iqr()
        elif detection_method == 'zscore':
            outlier_counts = self._detect_outliers_zscore()
        else:
            self.logger.warning(f"CLEANING: Unknown detection method '{detection_method}', using IQR")
            outlier_counts = self._detect_outliers_iqr()

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if outlier_counts.get(col, 0) > 0:
                if method == 'cap':
                    # Cap outliers
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                    self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                    self.logger.info(f"CLEANING: Capped outliers in '{col}'")

                elif method == 'remove':
                    # Remove outlier rows
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    initial_count = len(self.df)
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                    removed = initial_count - len(self.df)
                    self.logger.info(f"CLEANING: Removed {removed} outlier rows from '{col}'")

    def _standardize_column_names(self):
        """Standardize column names for consistency."""
        self.logger.info("CLEANING: Standardizing column names")

        # Convert to lowercase, replace spaces with underscores, remove special characters
        new_columns = {}
        for col in self.df.columns:
            new_col = str(col).lower()
            new_col = re.sub(r'[^\w\s]', '', new_col)  # Remove special characters
            new_col = re.sub(r'\s+', '_', new_col)  # Replace spaces with underscores
            new_col = re.sub(r'_+', '_', new_col)  # Remove multiple underscores
            new_col = new_col.strip('_')  # Remove leading/trailing underscores
            new_columns[col] = new_col

        self.df.rename(columns=new_columns, inplace=True)
        self.logger.info("CLEANING: Column names standardized")

    def _validate_data_quality(self) -> bool:
        """Validate data quality and completeness."""
        self.logger.info("CLEANING: Validating data quality")

        # Check for remaining missing values
        missing_total = self.df.isnull().sum().sum()
        if missing_total > 0:
            self.logger.warning(f"VALIDATION: Dataset still contains {missing_total} missing values")
            return False

        # Check if dataset is empty
        if self.df.empty:
            self.logger.error("VALIDATION: Dataset is empty after cleaning")
            return False

        # Check for infinite values
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                self.logger.warning(f"VALIDATION: Column '{col}' contains {inf_count} infinite values")

        self.logger.info("VALIDATION: Data quality validation completed")
        return True

    def execute(self) -> Optional[pd.DataFrame]:
        """Execute comprehensive data cleaning pipeline."""
        self.logger.info("CLEANING: Starting comprehensive data cleaning pipeline")
        self.logger.info(f"CLEANING: Initial dataset shape: {self.original_shape}")

        try:
            # Step 1: Identify missing values
            missing_info = self._identify_missing_values()

            # Step 2: Handle missing values (impute or remove)
            if missing_info:
                self._impute_missing_values(strategy='auto')
                self._remove_missing_data(threshold=0.8)  # Remove columns/rows with >80% missing

            # Step 3: Remove duplicates
            duplicate_count = self._identify_duplicates()
            if duplicate_count > 0:
                self._remove_duplicates()

            # Step 4: Standardize text and data types
            self._standardize_text()
            self._convert_data_types()

            # Step 5: Handle outliers
            self._handle_outliers(method='cap', detection_method='iqr')

            # Step 6: Standardize column names
            self._standardize_column_names()

            # Step 7: Final validation
            if not self._validate_data_quality():
                self.logger.warning("CLEANING: Data quality validation failed")

            # Convert numeric columns to integer if possible
            numeric_cols = self.df.select_dtypes(include=['float64']).columns
            for col in numeric_cols:
                if (self.df[col] % 1 == 0).all():
                    self.df[col] = self.df[col].astype(int)
                    self.logger.info(f"CLEANING: Converted column '{col}' to integer type")

            self.logger.info(f"CLEANING: Final dataset shape: {self.df.shape}")
            self.logger.info(f"CLEANING: Cleaning completed successfully")

            # Save cleaned data
            os.makedirs(os.path.dirname(CLEANED_DATA_FILE), exist_ok=True)
            self.df.to_csv(CLEANED_DATA_FILE, index=False)
            self.logger.info(f"CLEANING: Cleaned data saved to {CLEANED_DATA_FILE}")

            return self.df

        except Exception as e:
            self.logger.error(f"CLEANING: Error during data cleaning: {str(e)}")
            return None

class InsightsAgent:
    """Agent for generating key insights and analysis."""
    def __init__(self, logger: logging.Logger, df: pd.DataFrame):
        self.logger = logger
        self.df = df
        self.insights: Dict[str, Any] = {}

    def _detect_geographic_column(self) -> Optional[str]:
        """Detect the geographic column in the dataset."""
        possible_geo_cols = ['district_name', 'district', 'circle', 'division', 'subdivision', 'section', 'area', 'region', 'state', 'province']
        for col in self.df.columns:
            if col.lower() in possible_geo_cols:
                return col
        # If no exact match, check if column name contains geo keywords
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['district', 'circle', 'division', 'area', 'region']):
                return col
        return None

    def execute(self) -> Dict[str, Any]:
        """Calculates key metrics and stores insights."""
        self.logger.info("INSIGHTS: Beginning data analysis to find patterns and gaps.")

        numeric_cols = self.df.select_dtypes(include='number').columns.tolist()

        geo_col = self._detect_geographic_column()
        if not geo_col or not numeric_cols:
            self.logger.warning("INSIGHTS: Not enough data to generate insights. No geographic column or numeric columns found.")
            return self.insights

        self.logger.info(f"INSIGHTS: Using '{geo_col}' as the geographic column.")

        # Generate insights for all numeric columns
        for col in numeric_cols:
            # Skip columns where all values are the same
            if self.df[col].nunique() <= 1:
                self.logger.info(f"INSIGHTS: Skipping column '{col}' as all values are identical.")
                continue
            highest = self.df.sort_values(by=col, ascending=False).head(3)[[geo_col, col]].to_dict('records')
            lowest = self.df.sort_values(by=col, ascending=True).head(3)[[geo_col, col]].to_dict('records')
            self.insights[col] = {
                'highest': highest,
                'lowest': lowest
            }
            self.logger.info(f"INSIGHTS: Generated insights for column '{col}'.")

        self.logger.info("INSIGHTS: All key metrics and insights have been generated.")
        return self.insights

class OutputAgent:
    """Agent for generating CLI and documentation outputs."""
    def __init__(self, logger: logging.Logger, insights: Dict[str, Any]):
        self.logger = logger
        self.insights = insights

    def _detect_geographic_column(self, df_sample: pd.DataFrame) -> str:
        """Detect the geographic column from a sample DataFrame."""
        possible_geo_cols = ['district_name', 'district', 'circle', 'division', 'subdivision', 'section', 'area', 'region', 'state', 'province']
        for col in df_sample.columns:
            if col.lower() in possible_geo_cols:
                return col
        # If no exact match, check if column name contains geo keywords
        for col in df_sample.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['district', 'circle', 'division', 'area', 'region']):
                return col
        return 'Location'  # Default

    def execute(self) -> None:
        """Formats and prints insights to the CLI and saves to files."""
        self.logger.info("OUTPUT: Formatting and generating CLI and documentation artifacts.")

        from tabulate import tabulate
        import pandas as pd

        print("\n" + "="*80)
        print("REAL-TIME GOVERNANCE SYSTEM (RTGS) - GENERIC INSIGHTS")
        print("="*80)

        if not self.insights:
            print("\nNo insights to display. The system could not find enough relevant data columns.")
        else:
            # Detect geo column from the first insight
            geo_col_name = 'Location'
            if self.insights:
                first_col = next(iter(self.insights))
                first_data = self.insights[first_col]['highest']
                if first_data:
                    geo_col_name = list(first_data[0].keys())[0]

            for col, data in self.insights.items():
                print(f"\n--- Insights for '{col}' ---\n")

                highest_df = pd.DataFrame(data['highest'])
                highest_df.columns = [geo_col_name.replace('_', ' ').title(), col.replace('_', ' ').title()]
                print(f"{geo_col_name.replace('_', ' ').title()}s with the Highest Values:")
                print(tabulate(highest_df, headers='keys', tablefmt='psql', showindex=False))

                lowest_df = pd.DataFrame(data['lowest'])
                lowest_df.columns = [geo_col_name.replace('_', ' ').title(), col.replace('_', ' ').title()]
                print(f"\n{geo_col_name.replace('_', ' ').title()}s with the Lowest Values:")
                print(tabulate(lowest_df, headers='keys', tablefmt='psql', showindex=False))

        self.logger.info("DOCUMENTATION: Generating human-readable documentation.")
        with open(INSIGHTS_DOC, "w") as f:
            f.write("# RTGS Generic Insights Report\n\n")
            f.write("### Executive Summary\n")
            f.write("This report provides a data-driven overview based on the provided dataset. The system automatically identified key geographic and numeric columns to generate a preliminary analysis, highlighting significant disparities and trends.\n\n")
            f.write("### Key Findings\n\n")

            if self.insights:
                geo_col_name = list(self.insights[next(iter(self.insights))]['highest'][0].keys())[0]
                for col, data in self.insights.items():
                    f.write(f"#### Analysis of '{col}'\n")
                    f.write(f"The analysis of the '{col}' metric reveals significant variations across {geo_col_name.replace('_', ' ')}s.\n\n")
                    f.write("```\n")
                    f.write(tabulate(pd.DataFrame(data['highest']), headers='keys', tablefmt='psql', showindex=False))
                    f.write("\n")
                    f.write(tabulate(pd.DataFrame(data['lowest']), headers='keys', tablefmt='psql', showindex=False))
                    f.write("\n```\n")
            else:
                f.write("No insights could be generated due to a lack of identifiable geographic or numeric columns in the provided dataset.\n")

        self.logger.info(f"OUTPUT: All artifacts (logs, cleaned data, insights) saved in the 'artifacts' directory.")
        self.logger.info(f"OUTPUT: All artifacts (logs, cleaned data, insights) saved in the 'artifacts' directory.")
