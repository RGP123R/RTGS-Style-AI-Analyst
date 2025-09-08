# RTGS-Style AI Analyst for Telangana Open Data

---

## Mission

Prototype a **Real-Time Governance System (RTGS)** for policymakers. Using datasets from the [Telangana Open Data Portal](https://data.telangana.gov.in/), build a **terminal-based agentic system** that turns raw public data into **standardized, cleaned, trustworthy evidence** and communicates insights clearly **from the command line**.

---

## Objective

Create a **CLI-first agentic system** that ingests open government data, makes it **analysis-ready**, and surfaces insights a busy decision-maker can grasp at a glance.

If datasets are very large or messy, you may **start with a small slice** (e.g., one district or one year) and state that scope clearly.

---

## What This Project Delivers

- **Dataset Selection & Scope**
  Uses datasets from the Telangana Open Data Portal, focusing on electricity consumption data for Telangana. The scope includes data cleaning, standardization, and insights generation relevant for governance.

- **Ingestion & Standardization Output**
  Loads raw datasets and outputs analysis-ready versions.

- **Cleaning & Transformation Output**
  Produces cleaned and transformed datasets resolving data quality issues.

- **Insights Output**
  Generates clear insights revealing patterns, trends, gaps, or imbalances relevant to policymakers.

- **Logs & Documentation Output**
  Provides human-readable logs and generated documentation recording changes, reasons, and implications.

- **CLI-Accessible Results**
  All results accessible via terminal tables and ASCII charts, and/or saved image files.

---

## Features

- **Agentic Architecture**: Modular design with separate agents for ingestion, data preparation, insights generation, and output.
- **Auto-Detection**: Automatically detects geographic and numeric columns in CSV datasets.
- **Flexible Configuration**: Command-line arguments for dataset path and logging level.
- **Comprehensive Logging**: Built-in Python logging with file and console output.
- **Insights Generation**: Generates top/bottom insights for numeric columns with varying values.
- **CLI Output**: Formatted tables for easy reading.
- **Documentation**: Saves insights to Markdown reports.
- **Interactive Web UI**: Streamlit-based web application for visual data exploration.

---

## Installation

1. Clone or download the project.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

The RTGS Data Analysis Pipeline can be run in two ways:

### Command-Line Interface (CLI) - Primary Method

Run the pipeline with a CSV dataset:

```bash
python main.py --dataset path/to/your/dataset.csv
```

Optional arguments:

- `--dataset`: Path to the input CSV file (default: configured in config.py)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO)

Example:

```bash
python main.py --dataset consumption_detail_05_2022_temporary_supply.csv --log-level DEBUG
```

### Streamlit Web Application - Interactive Method

For an interactive web-based experience:

```bash
streamlit run streamlit_app.py
```

This launches a web interface at http://localhost:8501 where you can:

- Upload CSV files through a web interface
- View real-time processing progress
- Explore interactive data visualizations
- Download processed results and reports

Alternative launcher script:

```bash
python run_streamlit.py
```

---

## Output

The pipeline generates:

- **artifacts/run_log.txt**: Detailed log of all operations.
- **artifacts/cleaned_data.csv**: Processed and cleaned dataset.
- **artifacts/insights.md**: Human-readable insights report.
- **Console Output**: Formatted tables of insights.
- **Interactive Visualizations**: Charts and graphs in the Streamlit web interface.

---

## Project Structure

### Core Files

- `main.py`: CLI entry point, orchestrates the agents.
- `agents.py`: Agent classes for data processing (IngestAgent, DataPrepAgent, InsightsAgent, OutputAgent).
- `config.py`: Configuration and command-line argument parsing.
- `streamlit_app.py`: Interactive web application using Streamlit.
- `run_streamlit.py`: Convenience script to launch the Streamlit app.
- `generate_insights.py`: Standalone script for insight generation from cleaned data.

### Data and Configuration

- `requirements.txt`: Python dependencies (pandas, streamlit, matplotlib, seaborn, plotly, etc.).
- `sample_data.csv`: Sample dataset for testing.
- `consumption_detail_05_2022_temporary_supply.csv`: Example Telangana electricity consumption data.

### Documentation

- `README.md`: This documentation file.

### Output Directory

- `artifacts/`: Output directory containing:
  - `run_log.txt`: Detailed processing logs.
  - `cleaned_data.csv`: Processed and cleaned dataset.
  - `insights.md`: Human-readable insights report.

---

## Agents

1. **IngestAgent**: Loads CSV data into pandas DataFrame.
2. **DataPrepAgent**: Cleans, standardizes, and auto-detects columns.
3. **InsightsAgent**: Generates insights for numeric columns with varying values.
4. **OutputAgent**: Formats and saves CLI output and documentation.

---

## Requirements

- Python 3.7+
- pandas
- tabulate
- streamlit
- matplotlib
- seaborn
- plotly
- pytest (for testing)

---

---

## Security

- Validates dataset file paths to prevent directory traversal.
- Uses secure path handling for Windows and Unix systems.

---

## Dataset Manifest

- **Primary Dataset**: `consumption_detail_05_2022_temporary_supply.csv`

  - Source: Telangana Open Data Portal
  - Description: Electricity consumption data for Telangana districts
  - Time Range: May 2022
  - Key Fields: District names, consumption metrics, geographic identifiers
  - Purpose: Governance insights for energy distribution and policy making

- **Sample Dataset**: `sample_data.csv`
  - Purpose: Testing and demonstration

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Add tests for new features.
4. Ensure all tests pass.
5. Submit a pull request.

---

## License

This project is open-source. See LICENSE file for details.

---

## Video Demo

[https://drive.google.com/drive/folders/1ANRparSWVuzyq5PmGTZ9SGs4IFW-3dM_](https://drive.google.com/file/d/1i0hNHq3vkVfGVwqXi2n7bZ1auXZU28mT/view?usp=drive_link)

The demo video shows:

- Repository layout and file structure
- Installation and setup process
- Full pipeline execution on sample data
- Output locations and formats
- Interactive Streamlit web interface demonstration
