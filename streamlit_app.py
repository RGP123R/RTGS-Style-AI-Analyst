import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys
import io
import os
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any
import base64
from datetime import datetime
import json
from fpdf import FPDF
import markdown
import tempfile

# Import existing agents
from agents import IngestAgent, DataPrepAgent, InsightsAgent, OutputAgent
from config import LOG_FILE, CLEANED_DATA_FILE, INSIGHTS_DOC

# Set page configuration
st.set_page_config(
    page_title="RTGS Data Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging for Streamlit
class StreamlitLogger:
    def __init__(self):
        self.logs = []
        self.logger = logging.getLogger("rtgs_streamlit")
        self.logger.setLevel(logging.INFO)

        # Create a custom handler that captures logs
        class StreamlitHandler(logging.Handler):
            def __init__(self, log_list):
                super().__init__()
                self.log_list = log_list

            def emit(self, record):
                log_entry = self.format(record)
                self.log_list.append(log_entry)

        handler = StreamlitHandler(self.logs)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_logs(self):
        return "\n".join(self.logs)

    def clear_logs(self):
        self.logs = []

# Initialize session state
if 'logger' not in st.session_state:
    st.session_state.logger = StreamlitLogger()
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'insights' not in st.session_state:
    st.session_state.insights = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def capture_agent_output(func, *args, **kwargs):
    """Capture stdout and stderr from agent execution"""
    log_capture = io.StringIO()
    with redirect_stdout(log_capture), redirect_stderr(log_capture):
        result = func(*args, **kwargs)
    captured_output = log_capture.getvalue()
    return result, captured_output

def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    """Create a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def create_text_download_link(text: str, filename: str, link_text: str) -> str:
    """Create a download link for text content"""
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def generate_comprehensive_report(raw_data: pd.DataFrame, cleaned_data: pd.DataFrame,
                                insights: Dict[str, Any], logs: str) -> str:
    """Generate a comprehensive HTML report"""
    report_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Calculate data quality metrics
    raw_missing = raw_data.isnull().sum().sum()
    cleaned_missing = cleaned_data.isnull().sum().sum()
    duplicates_removed = len(raw_data) - len(cleaned_data.drop_duplicates())

    # Detect geographic column
    geo_col = None
    possible_geo_cols = ['district_name', 'district', 'circle', 'division', 'subdivision', 'section', 'area', 'region', 'state', 'province']
    for col in cleaned_data.columns:
        if col.lower() in possible_geo_cols:
            geo_col = col
            break

    html_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RTGS Data Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .section {{
                background: white;
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #667eea;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
            }}
            .insights-table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            .insights-table th, .insights-table td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            .insights-table th {{
                background-color: #f8f9fa;
                font-weight: 600;
            }}
            .status-good {{ color: #28a745; }}
            .status-warning {{ color: #ffc107; }}
            .status-danger {{ color: #dc3545; }}
            .recommendations {{
                background: #e8f4fd;
                border-left: 4px solid #2196F3;
                padding: 20px;
                margin: 20px 0;
            }}
            .footer {{
                text-align: center;
                color: #666;
                font-size: 0.9em;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 RTGS Data Analysis Report</h1>
            <p>Comprehensive Analysis & Insights</p>
            <p><strong>Generated on:</strong> {report_date}</p>
        </div>

        <div class="section">
            <h2>📈 Executive Summary</h2>
            <p>This report presents a comprehensive analysis of the uploaded dataset using the RTGS (Real-Time Governance System) data processing pipeline. The analysis includes data quality assessment, cleaning operations, and key insights generation.</p>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(raw_data):,}</div>
                    <div class="metric-label">Original Records</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(cleaned_data):,}</div>
                    <div class="metric-label">Cleaned Records</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(cleaned_data.columns)}</div>
                    <div class="metric-label">Data Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(insights)}</div>
                    <div class="metric-label">Insights Generated</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔍 Data Quality Assessment</h2>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value {'status-danger' if raw_missing > 0 else 'status-good'}">{raw_missing:,}</div>
                    <div class="metric-label">Missing Values (Raw)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'status-good' if cleaned_missing == 0 else 'status-warning'}">{cleaned_missing:,}</div>
                    <div class="metric-label">Missing Values (Cleaned)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'status-good' if duplicates_removed == 0 else 'status-warning'}">{duplicates_removed:,}</div>
                    <div class="metric-label">Duplicates Removed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{'✅' if geo_col else '❌'}</div>
                    <div class="metric-label">Geographic Column Detected</div>
                </div>
            </div>

            <h3>Data Types Distribution</h3>
            <ul>
    """

    # Add data types information
    dtype_counts = cleaned_data.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        html_report += f"<li><strong>{dtype}:</strong> {count} columns</li>"

    html_report += """
            </ul>
        </div>

        <div class="section">
            <h2>📊 Key Insights</h2>
    """

    if insights:
        for col_name, data in insights.items():
            html_report += f"""
            <h3>Analysis of {col_name.replace('_', ' ').title()}</h3>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                <div>
                    <h4 style="color: #28a745;">🏆 Highest Values</h4>
                    <table class="insights-table">
                        <thead>
                            <tr>
                                <th>{geo_col or 'Location'}</th>
                                <th>{col_name.replace('_', ' ').title()}</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            for item in data['highest'][:5]:  # Show top 5
                location = list(item.keys())[0]
                value = list(item.values())[0]
                html_report += f"""
                            <tr>
                                <td>{location}</td>
                                <td>{value:,.2f}</td>
                            </tr>
                """

            html_report += """
                        </tbody>
                    </table>
                </div>

                <div>
                    <h4 style="color: #dc3545;">📉 Lowest Values</h4>
                    <table class="insights-table">
                        <thead>
                            <tr>
                                <th>{geo_col or 'Location'}</th>
                                <th>{col_name.replace('_', ' ').title()}</th>
                            </tr>
                        </thead>
                        <tbody>
            """

            for item in data['lowest'][:5]:  # Show bottom 5
                location = list(item.keys())[0]
                value = list(item.values())[0]
                html_report += f"""
                            <tr>
                                <td>{location}</td>
                                <td>{value:,.2f}</td>
                            </tr>
                """

            html_report += """
                        </tbody>
                    </table>
                </div>
            </div>
            """
    else:
        html_report += "<p>No insights were generated from the dataset.</p>"

    html_report += """
        </div>

        <div class="section">
            <h2>💡 Recommendations</h2>
            <div class="recommendations">
                <h3>Data Quality Improvements</h3>
                <ul>
    """

    if raw_missing > 0:
        html_report += f"<li><strong>Missing Data:</strong> {raw_missing:,} missing values were identified and handled using advanced imputation techniques.</li>"

    if duplicates_removed > 0:
        html_report += f"<li><strong>Duplicate Records:</strong> {duplicates_removed:,} duplicate records were removed to ensure data integrity.</li>"

    if not geo_col:
        html_report += "<li><strong>Geographic Analysis:</strong> Consider adding geographic columns (district, region, etc.) for enhanced spatial analysis.</li>"

    html_report += """
                    <li><strong>Data Validation:</strong> Regular data quality checks are recommended to maintain high data standards.</li>
                    <li><strong>Trend Analysis:</strong> Consider time-series analysis if temporal data is available.</li>
                </ul>

                <h3>Next Steps</h3>
                <ul>
                    <li>Review the insights to identify areas requiring attention</li>
                    <li>Implement automated data quality monitoring</li>
                    <li>Consider predictive modeling based on the cleaned dataset</li>
                    <li>Share findings with relevant stakeholders</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>📋 Processing Summary</h2>
            <p>The data processing pipeline completed successfully with the following operations:</p>
            <ol>
                <li><strong>Data Ingestion:</strong> Successfully loaded and validated the input dataset</li>
                <li><strong>Data Cleaning:</strong> Applied comprehensive cleaning including missing value imputation, duplicate removal, and data type standardization</li>
                <li><strong>Insights Generation:</strong> Analyzed numeric columns to identify highest and lowest values by geographic regions</li>
                <li><strong>Report Generation:</strong> Created this comprehensive analysis report</li>
            </ol>
        </div>

        <div class="footer">
            <p>Report generated by RTGS Data Analysis Platform</p>
            <p>For questions or support, please contact the data analysis team.</p>
        </div>
    </body>
    </html>
    """

    return html_report

def generate_pdf_report(raw_data: pd.DataFrame, cleaned_data: pd.DataFrame,
                       insights: Dict[str, Any], logs: str) -> bytes:
    """Generate a PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)

    # Title
    pdf.cell(0, 10, "RTGS Data Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # Executive Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Executive Summary", ln=True)
    pdf.set_font("Arial", "", 10)
    summary_text = f"""This report presents a comprehensive analysis of the uploaded dataset.
Original Records: {len(raw_data):,}
Cleaned Records: {len(cleaned_data):,}
Data Columns: {len(cleaned_data.columns)}
Insights Generated: {len(insights)}"""

    # Split summary into lines to fit PDF width
    for line in summary_text.split('\n'):
        pdf.cell(0, 5, line, ln=True)
    pdf.ln(5)

    # Data Quality Metrics
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Data Quality Assessment", ln=True)
    pdf.set_font("Arial", "", 10)

    raw_missing = raw_data.isnull().sum().sum()
    cleaned_missing = cleaned_data.isnull().sum().sum()
    duplicates_removed = len(raw_data) - len(cleaned_data.drop_duplicates())

    metrics = [
        f"Missing Values (Raw): {raw_missing:,}",
        f"Missing Values (Cleaned): {cleaned_missing:,}",
        f"Duplicates Removed: {duplicates_removed:,}",
        f"Data Types: {', '.join([f'{k}: {v}' for k, v in cleaned_data.dtypes.value_counts().items()])}"
    ]

    for metric in metrics:
        pdf.cell(0, 5, metric, ln=True)
    pdf.ln(5)

    # Key Insights
    if insights:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Key Insights", ln=True)
        pdf.set_font("Arial", "", 10)

        for col_name, data in insights.items():
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, f"Analysis of {col_name.replace('_', ' ').title()}", ln=True)
            pdf.set_font("Arial", "", 10)

            # Highest values
            pdf.cell(0, 5, "Highest Values:", ln=True)
            for item in data['highest'][:3]:
                location = list(item.keys())[0]
                value = list(item.values())[0]
                pdf.cell(0, 5, f"  {location}: {value:,.2f}", ln=True)

            # Lowest values
            pdf.cell(0, 5, "Lowest Values:", ln=True)
            for item in data['lowest'][:3]:
                location = list(item.keys())[0]
                value = list(item.values())[0]
                pdf.cell(0, 5, f"  {location}: {value:,.2f}", ln=True)
            pdf.ln(5)

    # Recommendations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)

    recommendations = [
        "Review the insights to identify areas requiring attention",
        "Implement automated data quality monitoring",
        "Consider predictive modeling based on the cleaned dataset",
        "Share findings with relevant stakeholders"
    ]

    for rec in recommendations:
        pdf.cell(0, 5, f"• {rec}", ln=True)

    return pdf.output(dest='S').encode('latin-1')

def plot_data_overview(df: pd.DataFrame, title_suffix: str = "", chart_key: str = "data_overview"):
    """Create overview plots for the dataset"""
    st.subheader(f"📊 Data Overview {title_suffix}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")
        st.metric("Total Columns", f"{len(df.columns):,}")

    with col2:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        st.metric("Numeric Columns", f"{len(numeric_cols):,}")

    with col3:
        missing_total = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_total:,}")

    # Data types distribution
    st.subheader("Data Types Distribution")
    dtype_counts = df.dtypes.value_counts()
    fig = px.pie(values=dtype_counts.values, names=dtype_counts.index.astype(str),
                 title="Column Data Types")
    st.plotly_chart(fig, use_container_width=True, key=f"{chart_key}_pie")

def plot_insights_visualization(insights: Dict[str, Any], df: pd.DataFrame):
    """Create visualizations for insights"""
    st.subheader("📈 Insights Visualization")

    if not insights:
        st.warning("No insights available to visualize.")
        return

    # Detect geographic column
    geo_col = None
    possible_geo_cols = ['district_name', 'district', 'circle', 'division', 'subdivision', 'section', 'area', 'region', 'state', 'province']
    for col in df.columns:
        if col.lower() in possible_geo_cols:
            geo_col = col
            break
    if not geo_col:
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['district', 'circle', 'division', 'area', 'region']):
                geo_col = col
                break

    if not geo_col:
        st.warning("Could not detect geographic column for visualization.")
        return

    # Create tabs for different insights
    tabs = st.tabs(list(insights.keys()))

    for i, (col_name, data) in enumerate(insights.items()):
        with tabs[i]:
            st.subheader(f"Analysis of {col_name.replace('_', ' ').title()}")

            # Prepare data for plotting
            highest_df = pd.DataFrame(data['highest'])
            lowest_df = pd.DataFrame(data['lowest'])

            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Highest Values", "Lowest Values"),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )

            # Add highest values bar chart
            fig.add_trace(
                go.Bar(
                    x=highest_df[geo_col],
                    y=highest_df[col_name],
                    name="Highest",
                    marker_color='lightgreen'
                ),
                row=1, col=1
            )

            # Add lowest values bar chart
            fig.add_trace(
                go.Bar(
                    x=lowest_df[geo_col],
                    y=lowest_df[col_name],
                    name="Lowest",
                    marker_color='lightcoral'
                ),
                row=1, col=2
            )

            fig.update_layout(
                title=f"{col_name.replace('_', ' ').title()} by {geo_col.replace('_', ' ').title()}",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display tables
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Highest Values")
                st.dataframe(highest_df, use_container_width=True)

            with col2:
                st.subheader("Lowest Values")
                st.dataframe(lowest_df, use_container_width=True)

def main():
    st.title("📊 RTGS Data Analysis Platform")
    st.markdown("---")

    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("🔧 Controls")

        # File upload
        uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])

        # Processing button
        process_button = st.button("🚀 Process Data", type="primary", use_container_width=True)

        # Clear data button
        if st.button("🗑️ Clear All Data", use_container_width=True):
            st.session_state.raw_data = None
            st.session_state.cleaned_data = None
            st.session_state.insights = None
            st.session_state.processing_complete = False
            st.session_state.logger.clear_logs()
            st.rerun()

        # Download section
        if st.session_state.processing_complete:
            st.header("📥 Downloads")

            if st.session_state.cleaned_data is not None:
                st.markdown(create_download_link(
                    st.session_state.cleaned_data,
                    "cleaned_data.csv",
                    "📄 Download Cleaned Data"
                ), unsafe_allow_html=True)

            if st.session_state.insights:
                # Create insights text
                insights_text = "# RTGS Insights Report\n\n"
                for col, data in st.session_state.insights.items():
                    insights_text += f"## {col}\n\n"
                    insights_text += "### Highest Values\n"
                    insights_text += pd.DataFrame(data['highest']).to_markdown() + "\n\n"
                    insights_text += "### Lowest Values\n"
                    insights_text += pd.DataFrame(data['lowest']).to_markdown() + "\n\n"

                st.markdown(create_text_download_link(
                    insights_text,
                    "insights_report.md",
                    "📋 Download Insights Report"
                ), unsafe_allow_html=True)

    # Main content area
    if uploaded_file is not None and not st.session_state.processing_complete:
        # Display file info
        st.success(f"📁 File uploaded: {uploaded_file.name}")
        st.info(f"File size: {uploaded_file.size:,} bytes")

        # Load data preview
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.subheader("👀 Data Preview")
            st.dataframe(df_preview.head(10), use_container_width=True)

            # Basic stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{len(df_preview):,}")
            with col2:
                st.metric("Columns", f"{len(df_preview.columns):,}")
            with col3:
                missing = df_preview.isnull().sum().sum()
                st.metric("Missing Values", f"{missing:,}")

        except Exception as e:
            st.error(f"Error reading file: {e}")
            return

    # Process data when button is clicked
    if process_button and uploaded_file is not None:
        with st.spinner("🔄 Processing data... This may take a few moments."):

            # Reset file pointer
            uploaded_file.seek(0)

            # Step 1: Data Ingestion
            st.subheader("📥 Step 1: Data Ingestion")
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading data...")
            try:
                # Save uploaded file temporarily
                temp_path = "temp_uploaded_data.csv"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Create ingest agent
                ingest_agent = IngestAgent(st.session_state.logger.logger, temp_path)
                raw_data, ingest_logs = capture_agent_output(ingest_agent.execute)

                if raw_data is None:
                    st.error("Failed to load data. Check the logs for details.")
                    return

                st.session_state.raw_data = raw_data
                st.success("✅ Data ingestion completed!")
                progress_bar.progress(25)

            except Exception as e:
                st.error(f"Error during data ingestion: {e}")
                return
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            # Step 2: Data Cleaning
            status_text.text("Cleaning data...")
            st.subheader("🧹 Step 2: Data Cleaning")

            try:
                prep_agent = DataPrepAgent(st.session_state.logger.logger, st.session_state.raw_data)
                cleaned_data, cleaning_logs = capture_agent_output(prep_agent.execute)

                if cleaned_data is None:
                    st.error("Data cleaning failed. Check the logs for details.")
                    return

                st.session_state.cleaned_data = cleaned_data
                st.success("✅ Data cleaning completed!")
                progress_bar.progress(50)

            except Exception as e:
                st.error(f"Error during data cleaning: {e}")
                return

            # Step 3: Insights Generation
            status_text.text("Generating insights...")
            st.subheader("🔍 Step 3: Insights Generation")

            try:
                insights_agent = InsightsAgent(st.session_state.logger.logger, st.session_state.cleaned_data)
                insights, insights_logs = capture_agent_output(insights_agent.execute)

                st.session_state.insights = insights
                st.success("✅ Insights generation completed!")
                progress_bar.progress(75)

            except Exception as e:
                st.error(f"Error during insights generation: {e}")
                return

            # Step 4: Output Generation
            status_text.text("Generating outputs...")
            st.subheader("📋 Step 4: Output Generation")

            try:
                output_agent = OutputAgent(st.session_state.logger.logger, st.session_state.insights)
                output_logs = capture_agent_output(output_agent.execute)[1]

                st.success("✅ All processing completed!")
                progress_bar.progress(100)
                st.session_state.processing_complete = True

            except Exception as e:
                st.error(f"Error during output generation: {e}")
                return

            status_text.empty()
            progress_bar.empty()

    # Display results if processing is complete
    if st.session_state.processing_complete:
        st.success("🎉 Data processing pipeline completed successfully!")

        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Overview",
            "🔧 Processing Logs",
            "📈 Insights",
            "📋 Raw Insights",
            "📥 Downloads"
        ])

        with tab1:
            if st.session_state.raw_data is not None:
                st.subheader("Raw Data Overview")
                plot_data_overview(st.session_state.raw_data, "(Raw)", "raw")

            if st.session_state.cleaned_data is not None:
                st.subheader("Cleaned Data Overview")
                plot_data_overview(st.session_state.cleaned_data, "(Cleaned)", "cleaned")

                # Show before/after comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Raw Data Shape", f"{st.session_state.raw_data.shape[0]:,} × {st.session_state.raw_data.shape[1]:,}")
                with col2:
                    st.metric("Cleaned Data Shape", f"{st.session_state.cleaned_data.shape[0]:,} × {st.session_state.cleaned_data.shape[1]:,}")

        with tab2:
            st.subheader("Processing Logs")
            logs = st.session_state.logger.get_logs()
            if logs:
                st.code(logs, language="text")
            else:
                st.info("No logs available.")

        with tab3:
            if st.session_state.insights and st.session_state.cleaned_data is not None:
                plot_insights_visualization(st.session_state.insights, st.session_state.cleaned_data)
            else:
                st.warning("No insights available.")

        with tab4:
            st.subheader("Raw Insights Data")
            if st.session_state.insights:
                for col_name, data in st.session_state.insights.items():
                    st.subheader(f"📊 {col_name.replace('_', ' ').title()}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Highest Values:**")
                        st.json(data['highest'])

                    with col2:
                        st.write("**Lowest Values:**")
                        st.json(data['lowest'])
            else:
                st.warning("No insights available.")

        with tab5:
            st.subheader("Download Options")
            st.markdown("### Processed Data & Reports")

            if st.session_state.cleaned_data is not None:
                st.markdown(create_download_link(
                    st.session_state.cleaned_data,
                    "cleaned_data.csv",
                    "📄 Download Cleaned Data (CSV)"
                ), unsafe_allow_html=True)

            if st.session_state.insights:
                insights_text = "# RTGS Insights Report\n\n"
                insights_text += f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                for col, data in st.session_state.insights.items():
                    insights_text += f"## {col.replace('_', ' ').title()}\n\n"
                    insights_text += "### Highest Values\n"
                    insights_text += pd.DataFrame(data['highest']).to_markdown() + "\n\n"
                    insights_text += "### Lowest Values\n"
                    insights_text += pd.DataFrame(data['lowest']).to_markdown() + "\n\n"

                st.markdown(create_text_download_link(
                    insights_text,
                    "insights_report.md",
                    "📋 Download Insights Report (Markdown)"
                ), unsafe_allow_html=True)

                # Also offer JSON format
                st.markdown(create_text_download_link(
                    str(st.session_state.insights).replace("'", '"'),
                    "insights_report.json",
                    "📋 Download Insights Report (JSON)"
                ), unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and RTGS Agent Architecture")

if __name__ == "__main__":
    main()
