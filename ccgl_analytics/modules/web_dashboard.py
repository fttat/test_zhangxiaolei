"""
CCGL Analytics - Web Dashboard Module
Interactive web dashboard using Streamlit and FastAPI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from ..utils.logger import get_logger, LoggerMixin

class ChartGenerator(LoggerMixin):
    """Generate various types of charts and visualizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize chart generator.
        
        Args:
            config: Chart configuration
        """
        self.config = config or {}
        self.chart_config = self.config.get('web', {}).get('charts', {})
        self.default_width = self.chart_config.get('width', 800)
        self.default_height = self.chart_config.get('height', 600)
    
    def create_data_overview_chart(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create data overview visualization.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Chart configuration
        """
        # Data shape and types
        shape_data = pd.DataFrame({
            'Metric': ['Rows', 'Columns'],
            'Count': [data.shape[0], data.shape[1]]
        })
        
        fig = px.bar(
            shape_data, 
            x='Metric', 
            y='Count',
            title='Dataset Overview',
            width=self.default_width//2,
            height=self.default_height//2
        )
        
        return {
            'type': 'data_overview',
            'chart': fig,
            'summary': {
                'total_rows': data.shape[0],
                'total_columns': data.shape[1],
                'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
        }
    
    def create_missing_values_chart(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create missing values visualization.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Chart configuration
        """
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        
        if missing_data.empty:
            return {
                'type': 'missing_values',
                'chart': None,
                'message': 'No missing values found in the dataset'
            }
        
        fig = px.bar(
            x=missing_data.values,
            y=missing_data.index,
            orientation='h',
            title='Missing Values by Column',
            labels={'x': 'Number of Missing Values', 'y': 'Column'},
            width=self.default_width,
            height=max(400, len(missing_data) * 30)
        )
        
        return {
            'type': 'missing_values',
            'chart': fig,
            'summary': {
                'total_missing': missing_data.sum(),
                'columns_affected': len(missing_data),
                'worst_column': missing_data.index[-1] if not missing_data.empty else None
            }
        }
    
    def create_distribution_charts(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Create distribution charts for numeric columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of chart configurations
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        charts = []
        
        for column in numeric_columns[:6]:  # Limit to first 6 columns
            fig = px.histogram(
                data,
                x=column,
                title=f'Distribution of {column}',
                marginal='box',
                width=self.default_width//2,
                height=self.default_height//2
            )
            
            charts.append({
                'type': 'distribution',
                'column': column,
                'chart': fig,
                'statistics': {
                    'mean': data[column].mean(),
                    'median': data[column].median(),
                    'std': data[column].std(),
                    'min': data[column].min(),
                    'max': data[column].max()
                }
            })
        
        return charts
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation heatmap.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Chart configuration
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {
                'type': 'correlation',
                'chart': None,
                'message': 'Need at least 2 numeric columns for correlation analysis'
            }
        
        correlation_matrix = numeric_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title='Correlation Heatmap',
            aspect='auto',
            width=self.default_width,
            height=self.default_height
        )
        
        return {
            'type': 'correlation',
            'chart': fig,
            'summary': {
                'strongest_positive': correlation_matrix.unstack().sort_values(ascending=False).iloc[1],
                'strongest_negative': correlation_matrix.unstack().sort_values(ascending=True).iloc[0],
                'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            }
        }
    
    def create_clustering_visualization(self, data: pd.DataFrame, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create clustering visualization.
        
        Args:
            data: Original data
            clustering_results: Results from clustering analysis
            
        Returns:
            Chart configuration
        """
        if 'error' in clustering_results:
            return {
                'type': 'clustering',
                'chart': None,
                'message': f"Clustering visualization unavailable: {clustering_results['error']}"
            }
        
        # Get first two numeric columns for 2D visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns[:2]
        
        if len(numeric_columns) < 2:
            return {
                'type': 'clustering',
                'chart': None,
                'message': 'Need at least 2 numeric columns for clustering visualization'
            }
        
        # Create visualization data
        plot_data = data[numeric_columns].copy()
        plot_data['Cluster'] = clustering_results.get('cluster_labels', [0] * len(data))
        
        fig = px.scatter(
            plot_data,
            x=numeric_columns[0],
            y=numeric_columns[1],
            color='Cluster',
            title=f'Clustering Results ({clustering_results.get("algorithm", "Unknown")})',
            width=self.default_width,
            height=self.default_height
        )
        
        # Add cluster centers if available
        if 'cluster_centers' in clustering_results and clustering_results['cluster_centers']:
            centers = np.array(clustering_results['cluster_centers'])
            if centers.shape[1] >= 2:
                fig.add_trace(go.Scatter(
                    x=centers[:, 0],
                    y=centers[:, 1],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=15,
                        color='black',
                        line=dict(width=2, color='white')
                    ),
                    name='Cluster Centers'
                ))
        
        return {
            'type': 'clustering',
            'chart': fig,
            'summary': {
                'n_clusters': clustering_results.get('n_clusters', 0),
                'algorithm': clustering_results.get('algorithm', 'Unknown'),
                'silhouette_score': clustering_results.get('metrics', {}).get('silhouette_score')
            }
        }
    
    def create_anomaly_visualization(self, data: pd.DataFrame, anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create anomaly detection visualization.
        
        Args:
            data: Original data
            anomaly_results: Results from anomaly detection
            
        Returns:
            Chart configuration
        """
        if 'error' in anomaly_results:
            return {
                'type': 'anomaly',
                'chart': None,
                'message': f"Anomaly visualization unavailable: {anomaly_results['error']}"
            }
        
        # Get first two numeric columns for 2D visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns[:2]
        
        if len(numeric_columns) < 2:
            return {
                'type': 'anomaly',
                'chart': None,
                'message': 'Need at least 2 numeric columns for anomaly visualization'
            }
        
        # Create visualization data
        plot_data = data[numeric_columns].copy()
        anomaly_labels = anomaly_results.get('anomaly_labels', [False] * len(data))
        plot_data['Anomaly'] = ['Anomaly' if label else 'Normal' for label in anomaly_labels]
        
        fig = px.scatter(
            plot_data,
            x=numeric_columns[0],
            y=numeric_columns[1],
            color='Anomaly',
            title=f'Anomaly Detection Results ({anomaly_results.get("algorithm", "Unknown")})',
            color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
            width=self.default_width,
            height=self.default_height
        )
        
        return {
            'type': 'anomaly',
            'chart': fig,
            'summary': {
                'total_anomalies': anomaly_results.get('anomaly_count', 0),
                'anomaly_percentage': anomaly_results.get('anomaly_percentage', 0),
                'algorithm': anomaly_results.get('algorithm', 'Unknown')
            }
        }

class StreamlitDashboard(LoggerMixin):
    """Streamlit-based interactive dashboard."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Streamlit dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or {}
        self.chart_generator = ChartGenerator(config)
        
    def create_main_dashboard(self):
        """Create main dashboard interface."""
        st.set_page_config(
            page_title="CCGL Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main header
        st.title("🚀 CCGL Analytics System")
        st.markdown("**Centralized Control and Group Learning - Advanced Analytics Platform**")
        
        # Sidebar for navigation
        self._create_sidebar()
        
        # Main content area
        self._create_main_content()
    
    def _create_sidebar(self):
        """Create sidebar with navigation and controls."""
        st.sidebar.title("🔧 Dashboard Controls")
        
        # Data upload section
        st.sidebar.header("📁 Data Input")
        data_source = st.sidebar.selectbox(
            "Data Source",
            ["Upload File", "Database Query", "Sample Data"]
        )
        
        if data_source == "Upload File":
            uploaded_file = st.sidebar.file_uploader(
                "Choose a data file",
                type=['csv', 'xlsx', 'json']
            )
            
            if uploaded_file is not None:
                st.session_state['uploaded_file'] = uploaded_file
        
        elif data_source == "Sample Data":
            if st.sidebar.button("Load Sample Data"):
                self._load_sample_data()
        
        # Analysis options
        st.sidebar.header("📊 Analysis Options")
        analysis_types = st.sidebar.multiselect(
            "Select Analysis Types",
            ["Data Overview", "Quality Assessment", "Clustering", "Anomaly Detection", "Correlation Analysis"],
            default=["Data Overview", "Quality Assessment"]
        )
        st.session_state['analysis_types'] = analysis_types
        
        # Run analysis button
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            st.session_state['run_analysis'] = True
    
    def _create_main_content(self):
        """Create main content area."""
        # Check if data is available
        if 'data' not in st.session_state:
            self._show_welcome_page()
            return
        
        data = st.session_state['data']
        
        # Data overview section
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        
        with col2:
            st.metric("Total Columns", f"{data.shape[1]:,}")
        
        with col3:
            missing_count = data.isnull().sum().sum()
            st.metric("Missing Values", f"{missing_count:,}")
        
        with col4:
            memory_mb = data.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Data Overview", 
            "🔍 Quality Assessment", 
            "🎯 Clustering", 
            "⚠️ Anomaly Detection", 
            "📈 Advanced Analytics"
        ])
        
        with tab1:
            self._show_data_overview(data)
        
        with tab2:
            self._show_quality_assessment(data)
        
        with tab3:
            self._show_clustering_analysis(data)
        
        with tab4:
            self._show_anomaly_detection(data)
        
        with tab5:
            self._show_advanced_analytics(data)
    
    def _show_welcome_page(self):
        """Show welcome page when no data is loaded."""
        st.markdown("""
        ## Welcome to CCGL Analytics Dashboard! 👋
        
        ### Getting Started:
        1. **Upload your data** using the sidebar file uploader
        2. **Or load sample data** to explore the features
        3. **Select analysis types** you want to perform
        4. **Click "Run Analysis"** to generate insights
        
        ### Features:
        - 📊 **Interactive Data Visualization**
        - 🤖 **Machine Learning Analysis**
        - 🔍 **Data Quality Assessment**
        - 📈 **Real-time Analytics**
        - 🌐 **Multi-language Support**
        
        ### Supported File Formats:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        - JSON (.json)
        """)
        
        # Sample data preview
        if st.button("🎯 Try with Sample Data"):
            self._load_sample_data()
            st.rerun()
    
    def _load_sample_data(self):
        """Load sample data for demonstration."""
        np.random.seed(42)
        
        # Generate sample sales data
        n_samples = 1000
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        
        sample_data = pd.DataFrame({
            'date': dates,
            'sales_amount': np.random.normal(1000, 300, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], n_samples),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'discount_rate': np.random.uniform(0, 0.3, n_samples),
            'customer_satisfaction': np.random.normal(4.0, 0.8, n_samples)
        })
        
        # Add some missing values
        sample_data.loc[sample_data.index[:50], 'customer_satisfaction'] = np.nan
        sample_data.loc[sample_data.index[100:110], 'sales_amount'] = np.nan
        
        st.session_state['data'] = sample_data
        st.success("Sample data loaded successfully!")
    
    def _show_data_overview(self, data: pd.DataFrame):
        """Show data overview tab."""
        st.header("📋 Data Overview")
        
        # Basic information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            info_df = pd.DataFrame({
                'Property': ['Shape', 'Memory Usage', 'Numeric Columns', 'Text Columns'],
                'Value': [
                    f"{data.shape[0]} rows × {data.shape[1]} columns",
                    f"{data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                    len(data.select_dtypes(include=[np.number]).columns),
                    len(data.select_dtypes(include=['object']).columns)
                ]
            })
            st.dataframe(info_df, use_container_width=True)
        
        with col2:
            st.subheader("Column Data Types")
            dtype_counts = data.dtypes.value_counts()
            fig = px.pie(
                values=dtype_counts.values,
                names=dtype_counts.index,
                title="Distribution of Data Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data sample
        st.subheader("Data Sample")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Column details
        st.subheader("Column Details")
        column_info = []
        for col in data.columns:
            column_info.append({
                'Column': col,
                'Data Type': str(data[col].dtype),
                'Non-Null Count': data[col].count(),
                'Null Count': data[col].isnull().sum(),
                'Unique Values': data[col].nunique()
            })
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)
    
    def _show_quality_assessment(self, data: pd.DataFrame):
        """Show data quality assessment tab."""
        st.header("🔍 Data Quality Assessment")
        
        # Missing values analysis
        missing_chart = self.chart_generator.create_missing_values_chart(data)
        
        if missing_chart['chart'] is not None:
            st.subheader("Missing Values Analysis")
            st.plotly_chart(missing_chart['chart'], use_container_width=True)
            
            # Missing values summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Missing", missing_chart['summary']['total_missing'])
            with col2:
                st.metric("Columns Affected", missing_chart['summary']['columns_affected'])
            with col3:
                st.metric("Worst Column", missing_chart['summary']['worst_column'] or "None")
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Data distributions
        st.subheader("Data Distributions")
        distribution_charts = self.chart_generator.create_distribution_charts(data)
        
        if distribution_charts:
            cols = st.columns(2)
            for i, chart_data in enumerate(distribution_charts):
                with cols[i % 2]:
                    st.plotly_chart(chart_data['chart'], use_container_width=True)
                    
                    # Statistics
                    stats = chart_data['statistics']
                    st.write(f"**{chart_data['column']} Statistics:**")
                    stat_cols = st.columns(3)
                    with stat_cols[0]:
                        st.metric("Mean", f"{stats['mean']:.2f}")
                    with stat_cols[1]:
                        st.metric("Median", f"{stats['median']:.2f}")
                    with stat_cols[2]:
                        st.metric("Std Dev", f"{stats['std']:.2f}")
        
        # Correlation analysis
        correlation_chart = self.chart_generator.create_correlation_heatmap(data)
        
        if correlation_chart['chart'] is not None:
            st.subheader("Correlation Analysis")
            st.plotly_chart(correlation_chart['chart'], use_container_width=True)
            
            # Correlation summary
            summary = correlation_chart['summary']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Strongest Positive", f"{summary['strongest_positive']:.3f}")
            with col2:
                st.metric("Strongest Negative", f"{summary['strongest_negative']:.3f}")
            with col3:
                st.metric("Average Correlation", f"{summary['avg_correlation']:.3f}")
    
    def _show_clustering_analysis(self, data: pd.DataFrame):
        """Show clustering analysis tab."""
        st.header("🎯 Clustering Analysis")
        
        # Clustering options
        col1, col2 = st.columns(2)
        
        with col1:
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["kmeans", "dbscan", "hierarchical"]
            )
        
        with col2:
            if algorithm in ["kmeans", "hierarchical"]:
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            else:
                n_clusters = 'auto'
        
        if st.button("Run Clustering Analysis"):
            with st.spinner("Running clustering analysis..."):
                # Simulate clustering results
                clustering_results = self._simulate_clustering_results(data, algorithm, n_clusters)
                
                # Create visualization
                cluster_chart = self.chart_generator.create_clustering_visualization(data, clustering_results)
                
                if cluster_chart['chart'] is not None:
                    st.plotly_chart(cluster_chart['chart'], use_container_width=True)
                    
                    # Clustering summary
                    summary = cluster_chart['summary']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Number of Clusters", summary['n_clusters'])
                    with col2:
                        st.metric("Algorithm", summary['algorithm'])
                    with col3:
                        silhouette = summary.get('silhouette_score')
                        if silhouette:
                            st.metric("Silhouette Score", f"{silhouette:.3f}")
                else:
                    st.error(cluster_chart['message'])
    
    def _show_anomaly_detection(self, data: pd.DataFrame):
        """Show anomaly detection tab."""
        st.header("⚠️ Anomaly Detection")
        
        # Anomaly detection options
        col1, col2 = st.columns(2)
        
        with col1:
            algorithm = st.selectbox(
                "Detection Algorithm",
                ["isolation_forest", "local_outlier_factor", "one_class_svm"]
            )
        
        with col2:
            contamination = st.slider("Expected Contamination", 0.01, 0.5, 0.1, 0.01)
        
        if st.button("Run Anomaly Detection"):
            with st.spinner("Running anomaly detection..."):
                # Simulate anomaly detection results
                anomaly_results = self._simulate_anomaly_results(data, algorithm, contamination)
                
                # Create visualization
                anomaly_chart = self.chart_generator.create_anomaly_visualization(data, anomaly_results)
                
                if anomaly_chart['chart'] is not None:
                    st.plotly_chart(anomaly_chart['chart'], use_container_width=True)
                    
                    # Anomaly summary
                    summary = anomaly_chart['summary']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Anomalies", summary['total_anomalies'])
                    with col2:
                        st.metric("Anomaly Percentage", f"{summary['anomaly_percentage']:.2f}%")
                    with col3:
                        st.metric("Algorithm", summary['algorithm'])
                else:
                    st.error(anomaly_chart['message'])
    
    def _show_advanced_analytics(self, data: pd.DataFrame):
        """Show advanced analytics tab."""
        st.header("📈 Advanced Analytics")
        
        # Time series analysis if date column exists
        date_columns = data.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            st.subheader("📅 Time Series Analysis")
            date_col = st.selectbox("Select Date Column", date_columns)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                value_col = st.selectbox("Select Value Column", numeric_cols)
                
                # Create time series plot
                fig = px.line(
                    data,
                    x=date_col,
                    y=value_col,
                    title=f"Time Series: {value_col} over {date_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("📊 Statistical Summary")
        numeric_data = data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe(), use_container_width=True)
        
        # Export options
        st.subheader("💾 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="ccgl_analysis_results.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = data.to_json(orient='records', indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_data,
                file_name="ccgl_analysis_results.json",
                mime="application/json"
            )
    
    def _simulate_clustering_results(self, data: pd.DataFrame, algorithm: str, n_clusters: Union[int, str]) -> Dict[str, Any]:
        """Simulate clustering results for demonstration."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No numeric data available'}
        
        n = len(data)
        if isinstance(n_clusters, str):
            n_clusters = 3
        
        # Generate random cluster labels
        cluster_labels = np.random.randint(0, n_clusters, n)
        
        # Generate random cluster centers
        if numeric_data.shape[1] >= 2:
            cluster_centers = []
            for _ in range(n_clusters):
                center = [
                    np.random.uniform(numeric_data.iloc[:, 0].min(), numeric_data.iloc[:, 0].max()),
                    np.random.uniform(numeric_data.iloc[:, 1].min(), numeric_data.iloc[:, 1].max())
                ]
                cluster_centers.append(center)
        else:
            cluster_centers = []
        
        return {
            'algorithm': algorithm,
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': cluster_centers,
            'metrics': {
                'silhouette_score': np.random.uniform(0.3, 0.8)
            }
        }
    
    def _simulate_anomaly_results(self, data: pd.DataFrame, algorithm: str, contamination: float) -> Dict[str, Any]:
        """Simulate anomaly detection results for demonstration."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return {'error': 'No numeric data available'}
        
        n = len(data)
        n_anomalies = int(n * contamination)
        
        # Generate random anomaly labels
        anomaly_labels = [False] * n
        anomaly_indices = np.random.choice(n, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_labels[idx] = True
        
        return {
            'algorithm': algorithm,
            'anomaly_labels': anomaly_labels,
            'anomaly_count': n_anomalies,
            'anomaly_percentage': contamination * 100,
            'contamination': contamination
        }

class WebDashboard(LoggerMixin):
    """Main web dashboard class combining Streamlit and FastAPI."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize web dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or {}
        self.streamlit_dashboard = StreamlitDashboard(config)
        self.fastapi_app = self._create_fastapi_app() if FASTAPI_AVAILABLE else None
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application for API endpoints."""
        app = FastAPI(
            title="CCGL Analytics API",
            description="API for CCGL Analytics System",
            version="1.0.0"
        )
        
        @app.get("/")
        async def root():
            return {"message": "CCGL Analytics API", "status": "running"}
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @app.post("/analyze")
        async def analyze_data(request: Dict[str, Any]):
            # This would integrate with the actual analysis modules
            return {
                "status": "completed",
                "results": {
                    "message": "Analysis completed",
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        return app
    
    def run_streamlit_dashboard(self, port: int = 8501, host: str = "localhost"):
        """Run Streamlit dashboard.
        
        Args:
            port: Port number
            host: Host address
        """
        self.logger.info(f"Starting Streamlit dashboard on {host}:{port}")
        self.streamlit_dashboard.create_main_dashboard()
    
    def run_fastapi_server(self, port: int = 8000, host: str = "0.0.0.0"):
        """Run FastAPI server.
        
        Args:
            port: Port number
            host: Host address
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required to run the API server")
        
        self.logger.info(f"Starting FastAPI server on {host}:{port}")
        uvicorn.run(self.fastapi_app, host=host, port=port)
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive analysis report.
        
        Args:
            analysis_results: Results from analysis
            
        Returns:
            Generated report
        """
        report = {
            'title': 'CCGL Analytics Report',
            'generated_at': datetime.now().isoformat(),
            'summary': self._generate_report_summary(analysis_results),
            'sections': self._generate_report_sections(analysis_results),
            'recommendations': self._generate_recommendations(analysis_results)
        }
        
        return report
    
    def _generate_report_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report summary."""
        return {
            'total_analyses': len(results),
            'successful_analyses': len([r for r in results.values() if 'error' not in r]),
            'data_quality_score': 0.85,  # Placeholder
            'key_findings_count': 3
        }
    
    def _generate_report_sections(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate report sections."""
        sections = []
        
        for analysis_type, result in results.items():
            if 'error' not in result:
                sections.append({
                    'title': analysis_type.replace('_', ' ').title(),
                    'type': analysis_type,
                    'content': self._format_analysis_content(analysis_type, result),
                    'charts': self._get_chart_references(analysis_type, result)
                })
        
        return sections
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        recommendations.append("Implement data quality monitoring for continuous improvement")
        
        # Analysis-specific recommendations
        if 'clustering' in results and 'error' not in results['clustering']:
            recommendations.append("Consider customer segmentation strategies based on clustering results")
        
        if 'anomaly_detection' in results and 'error' not in results['anomaly_detection']:
            recommendations.append("Investigate detected anomalies for potential business opportunities")
        
        recommendations.append("Establish regular analysis schedules for ongoing insights")
        
        return recommendations
    
    def _format_analysis_content(self, analysis_type: str, result: Dict[str, Any]) -> str:
        """Format analysis content for reports."""
        if analysis_type == 'clustering':
            return f"Identified {result.get('n_clusters', 0)} distinct clusters using {result.get('algorithm', 'unknown')} algorithm."
        
        elif analysis_type == 'anomaly_detection':
            return f"Detected {result.get('anomaly_count', 0)} anomalies ({result.get('anomaly_percentage', 0):.1f}%) using {result.get('algorithm', 'unknown')} method."
        
        else:
            return f"Analysis completed successfully for {analysis_type}."
    
    def _get_chart_references(self, analysis_type: str, result: Dict[str, Any]) -> List[str]:
        """Get chart references for the analysis."""
        # Placeholder for chart references
        return [f"{analysis_type}_visualization.png"]