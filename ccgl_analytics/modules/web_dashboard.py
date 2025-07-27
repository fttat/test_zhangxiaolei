"""
Web Dashboard Module for CCGL Analytics
Generates HTML dashboards and visualizations
"""

import pandas as pd
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import base64

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from ..utils.logger import get_logger


class WebDashboardGenerator:
    """
    Web Dashboard Generator for CCGL Analytics
    Creates interactive HTML dashboards and reports
    """
    
    def __init__(self, template_dir: str = "templates", output_dir: str = "dashboard"):
        """
        Initialize dashboard generator
        
        Args:
            template_dir: Directory containing HTML templates
            output_dir: Directory to save generated dashboards
        """
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.logger = get_logger("web_dashboard")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart configurations
        self.chart_config = {
            'theme': 'plotly_white',
            'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }
        
        self.logger.info("Initialized Web Dashboard Generator")
    
    def generate_comprehensive_dashboard(self, analysis_results: Dict[str, Any], 
                                       dataset_info: Dict[str, Any]) -> str:
        """
        Generate comprehensive analysis dashboard
        
        Args:
            analysis_results: Results from comprehensive analysis
            dataset_info: Dataset information
            
        Returns:
            Path to generated dashboard HTML file
        """
        self.logger.info("Generating comprehensive dashboard")
        
        # Generate individual components
        charts = []
        
        # Dataset overview chart
        overview_chart = self._create_dataset_overview_chart(dataset_info)
        if overview_chart:
            charts.append(overview_chart)
        
        # Clustering charts
        if 'clustering' in analysis_results:
            clustering_charts = self._create_clustering_charts(analysis_results['clustering'])
            charts.extend(clustering_charts)
        
        # Anomaly detection charts
        if 'anomaly_detection' in analysis_results:
            anomaly_charts = self._create_anomaly_charts(analysis_results['anomaly_detection'])
            charts.extend(anomaly_charts)
        
        # Dimensionality reduction charts
        if 'dimensionality_reduction' in analysis_results:
            dim_charts = self._create_dimensionality_charts(analysis_results['dimensionality_reduction'])
            charts.extend(dim_charts)
        
        # Generate HTML dashboard
        dashboard_html = self._generate_dashboard_html(charts, analysis_results, dataset_info)
        
        # Save dashboard
        dashboard_file = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Dashboard saved to: {dashboard_file}")
        return str(dashboard_file)
    
    def _create_dataset_overview_chart(self, dataset_info: Dict[str, Any]) -> Optional[str]:
        """Create dataset overview chart"""
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            # Create subplots for overview
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Data Types Distribution', 'Missing Values', 
                               'Memory Usage', 'Column Count by Type'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Data types pie chart
            if 'dtypes' in dataset_info:
                dtypes_data = dataset_info['dtypes']
                dtypes_counts = {}
                for dtype in dtypes_data.values():
                    dtype_category = self._categorize_dtype(dtype)
                    dtypes_counts[dtype_category] = dtypes_counts.get(dtype_category, 0) + 1
                
                fig.add_trace(
                    go.Pie(labels=list(dtypes_counts.keys()), 
                          values=list(dtypes_counts.values()),
                          name="Data Types"),
                    row=1, col=1
                )
            
            # Missing values bar chart
            if 'missing_values' in dataset_info:
                missing_data = dataset_info['missing_values']
                missing_cols = [(k, v) for k, v in missing_data.items() if v > 0]
                if missing_cols:
                    cols, values = zip(*missing_cols[:10])  # Top 10
                    fig.add_trace(
                        go.Bar(x=list(cols), y=list(values), name="Missing Values"),
                        row=1, col=2
                    )
            
            # Memory usage
            if 'memory_usage_mb' in dataset_info:
                memory_mb = dataset_info['memory_usage_mb']
                fig.add_trace(
                    go.Bar(x=['Dataset'], y=[memory_mb], name="Memory (MB)"),
                    row=2, col=1
                )
            
            # Column counts
            if 'numeric_columns' in dataset_info and 'categorical_columns' in dataset_info:
                numeric_count = len(dataset_info['numeric_columns'])
                categorical_count = len(dataset_info['categorical_columns'])
                
                fig.add_trace(
                    go.Bar(x=['Numeric', 'Categorical'], 
                          y=[numeric_count, categorical_count],
                          name="Column Types"),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="Dataset Overview",
                showlegend=False,
                height=600
            )
            
            return fig.to_html(include_plotlyjs='cdn', div_id="dataset-overview")
            
        except Exception as e:
            self.logger.error(f"Failed to create dataset overview chart: {str(e)}")
            return None
    
    def _create_clustering_charts(self, clustering_results: Dict[str, Any]) -> List[str]:
        """Create clustering visualization charts"""
        charts = []
        
        if not PLOTLY_AVAILABLE:
            return charts
        
        try:
            for method, results in clustering_results.items():
                if 'clustered_data' not in results:
                    continue
                
                df = results['clustered_data']
                
                # Create clustering scatter plot if we have numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if len(numeric_cols) >= 2 and 'cluster' in df.columns:
                    fig = px.scatter(
                        df, 
                        x=numeric_cols[0], 
                        y=numeric_cols[1],
                        color='cluster',
                        title=f"{method.upper()} Clustering Results",
                        labels={'cluster': 'Cluster'},
                        color_discrete_sequence=self.chart_config['color_palette']
                    )
                    
                    # Add cluster centers if available
                    if 'cluster_centers' in results:
                        centers = results['cluster_centers']
                        fig.add_scatter(
                            x=centers[:, 0], y=centers[:, 1],
                            mode='markers',
                            marker=dict(symbol='x', size=15, color='black'),
                            name='Cluster Centers'
                        )
                    
                    fig.update_layout(height=500)
                    charts.append(fig.to_html(include_plotlyjs='cdn', div_id=f"clustering-{method}"))
                
                # Cluster size distribution
                if 'cluster' in df.columns:
                    cluster_counts = df['cluster'].value_counts()
                    fig = go.Figure(data=[
                        go.Bar(x=cluster_counts.index, y=cluster_counts.values)
                    ])
                    fig.update_layout(
                        title=f"{method.upper()} Cluster Size Distribution",
                        xaxis_title="Cluster",
                        yaxis_title="Number of Points",
                        height=400
                    )
                    charts.append(fig.to_html(include_plotlyjs='cdn', div_id=f"cluster-sizes-{method}"))
        
        except Exception as e:
            self.logger.error(f"Failed to create clustering charts: {str(e)}")
        
        return charts
    
    def _create_anomaly_charts(self, anomaly_results: Dict[str, Any]) -> List[str]:
        """Create anomaly detection visualization charts"""
        charts = []
        
        if not PLOTLY_AVAILABLE:
            return charts
        
        try:
            for method, results in anomaly_results.items():
                if 'full_data' not in results:
                    continue
                
                df = results['full_data']
                
                # Anomaly distribution
                if 'is_anomaly' in df.columns:
                    anomaly_counts = df['is_anomaly'].value_counts()
                    labels = ['Normal', 'Anomaly']
                    values = [anomaly_counts.get(False, 0), anomaly_counts.get(True, 0)]
                    
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                    fig.update_layout(
                        title=f"{method.replace('_', ' ').title()} - Anomaly Distribution",
                        height=400
                    )
                    charts.append(fig.to_html(include_plotlyjs='cdn', div_id=f"anomaly-dist-{method}"))
                
                # Anomaly scores (if available)
                if 'anomaly_score' in df.columns:
                    fig = go.Figure()
                    fig.add_histogram(
                        x=df[df['is_anomaly'] == False]['anomaly_score'],
                        name='Normal',
                        opacity=0.7
                    )
                    fig.add_histogram(
                        x=df[df['is_anomaly'] == True]['anomaly_score'],
                        name='Anomaly',
                        opacity=0.7
                    )
                    fig.update_layout(
                        title=f"{method.replace('_', ' ').title()} - Anomaly Scores",
                        xaxis_title="Anomaly Score",
                        yaxis_title="Count",
                        barmode='overlay',
                        height=400
                    )
                    charts.append(fig.to_html(include_plotlyjs='cdn', div_id=f"anomaly-scores-{method}"))
        
        except Exception as e:
            self.logger.error(f"Failed to create anomaly charts: {str(e)}")
        
        return charts
    
    def _create_dimensionality_charts(self, dim_results: Dict[str, Any]) -> List[str]:
        """Create dimensionality reduction visualization charts"""
        charts = []
        
        if not PLOTLY_AVAILABLE:
            return charts
        
        try:
            # PCA charts
            if 'pca' in dim_results:
                pca_data = dim_results['pca']
                
                # Explained variance chart
                if 'explained_variance_ratio' in pca_data:
                    variance_ratio = pca_data['explained_variance_ratio']
                    cumulative_variance = pca_data.get('cumulative_variance', [])
                    
                    fig = go.Figure()
                    fig.add_bar(
                        x=[f'PC{i+1}' for i in range(len(variance_ratio))],
                        y=variance_ratio,
                        name='Individual Variance'
                    )
                    
                    if cumulative_variance:
                        fig.add_scatter(
                            x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                            y=cumulative_variance,
                            mode='lines+markers',
                            name='Cumulative Variance',
                            yaxis='y2'
                        )
                    
                    fig.update_layout(
                        title="PCA Explained Variance",
                        xaxis_title="Principal Component",
                        yaxis_title="Explained Variance Ratio",
                        yaxis2=dict(title="Cumulative Variance", overlaying='y', side='right'),
                        height=400
                    )
                    charts.append(fig.to_html(include_plotlyjs='cdn', div_id="pca-variance"))
                
                # PCA scatter plot
                if 'transformed_data' in pca_data:
                    df = pca_data['transformed_data']
                    if 'PC1' in df.columns and 'PC2' in df.columns:
                        fig = px.scatter(
                            df, x='PC1', y='PC2',
                            title="PCA - First Two Components",
                            labels={'PC1': 'First Principal Component', 
                                   'PC2': 'Second Principal Component'}
                        )
                        fig.update_layout(height=500)
                        charts.append(fig.to_html(include_plotlyjs='cdn', div_id="pca-scatter"))
            
            # t-SNE charts
            if 'tsne' in dim_results:
                tsne_data = dim_results['tsne']
                if 'transformed_data' in tsne_data:
                    df = tsne_data['transformed_data']
                    if 'tSNE1' in df.columns and 'tSNE2' in df.columns:
                        fig = px.scatter(
                            df, x='tSNE1', y='tSNE2',
                            title="t-SNE Visualization",
                            labels={'tSNE1': 't-SNE Component 1', 
                                   'tSNE2': 't-SNE Component 2'}
                        )
                        fig.update_layout(height=500)
                        charts.append(fig.to_html(include_plotlyjs='cdn', div_id="tsne-scatter"))
        
        except Exception as e:
            self.logger.error(f"Failed to create dimensionality charts: {str(e)}")
        
        return charts
    
    def _generate_dashboard_html(self, charts: List[str], analysis_results: Dict[str, Any], 
                               dataset_info: Dict[str, Any]) -> str:
        """Generate complete HTML dashboard"""
        
        # Extract summary statistics
        summary_stats = self._extract_summary_stats(analysis_results, dataset_info)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCGL Analytics Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #666;
            font-size: 1em;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏭 CCGL Analytics Dashboard</h1>
        <p>Enterprise Warehouse Management Data Analysis</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats-grid">
        {self._generate_stats_cards(summary_stats)}
    </div>
    
    <div class="section-title">📊 Data Analysis Results</div>
    
    {chr(10).join(f'<div class="chart-container">{chart}</div>' for chart in charts)}
    
    <div class="footer">
        <p>Generated by CCGL Analytics Platform | 
           <a href="https://github.com/fttat/test_zhangxiaolei">GitHub</a></p>
    </div>
    
    <script>
        // Auto-resize charts on window resize
        window.addEventListener('resize', function() {{
            var charts = document.querySelectorAll('.plotly-graph-div');
            charts.forEach(function(chart) {{
                Plotly.Plots.resize(chart);
            }});
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _extract_summary_stats(self, analysis_results: Dict[str, Any], 
                              dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key statistics for dashboard summary"""
        stats = {}
        
        # Dataset stats
        if 'shape' in dataset_info:
            stats['total_records'] = dataset_info['shape'][0]
            stats['total_columns'] = dataset_info['shape'][1]
        
        # Quality score
        if 'preprocessing_summary' in analysis_results:
            preprocessing = analysis_results['preprocessing_summary']
            if 'detailed_steps' in preprocessing:
                for step_name, step_data in preprocessing['detailed_steps']:
                    if step_name == 'quality_check' and isinstance(step_data, dict):
                        if 'overall_score' in step_data:
                            stats['quality_score'] = f"{step_data['overall_score']:.3f}"
        
        # Clustering stats
        if 'clustering' in analysis_results:
            clustering = analysis_results['clustering']
            if 'kmeans' in clustering and 'n_clusters' in clustering['kmeans']:
                stats['clusters_found'] = clustering['kmeans']['n_clusters']
        
        # Anomaly stats
        if 'anomaly_detection' in analysis_results:
            anomaly = analysis_results['anomaly_detection']
            if 'isolation_forest' in anomaly and 'anomalies_detected' in anomaly['isolation_forest']:
                stats['anomalies_detected'] = anomaly['isolation_forest']['anomalies_detected']
        
        return stats
    
    def _generate_stats_cards(self, stats: Dict[str, Any]) -> str:
        """Generate HTML for statistics cards"""
        cards = []
        
        card_configs = [
            ('total_records', 'Total Records', '📊'),
            ('total_columns', 'Columns', '📋'),
            ('quality_score', 'Data Quality', '✅'),
            ('clusters_found', 'Clusters', '🎯'),
            ('anomalies_detected', 'Anomalies', '⚠️')
        ]
        
        for key, label, icon in card_configs:
            if key in stats:
                value = stats[key]
                cards.append(f"""
                <div class="stat-card">
                    <div class="stat-value">{icon} {value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """)
        
        return ''.join(cards)
    
    def _categorize_dtype(self, dtype: str) -> str:
        """Categorize pandas dtype into broader categories"""
        dtype = str(dtype).lower()
        if 'int' in dtype or 'float' in dtype:
            return 'Numeric'
        elif 'object' in dtype or 'string' in dtype:
            return 'Text'
        elif 'datetime' in dtype:
            return 'DateTime'
        elif 'bool' in dtype:
            return 'Boolean'
        else:
            return 'Other'
    
    def create_quickchart_url(self, chart_type: str, data: Dict[str, Any]) -> str:
        """
        Create QuickChart URL for simple charts
        
        Args:
            chart_type: Type of chart (bar, pie, line, etc.)
            data: Chart data
            
        Returns:
            QuickChart URL
        """
        base_url = "https://quickchart.io/chart"
        
        chart_config = {
            'type': chart_type,
            'data': data,
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': data.get('title', 'Chart')
                    }
                }
            }
        }
        
        # Encode chart configuration
        chart_json = json.dumps(chart_config)
        encoded_chart = base64.b64encode(chart_json.encode()).decode()
        
        return f"{base_url}?c={encoded_chart}"
    
    def generate_simple_dashboard(self, data: Dict[str, Any]) -> str:
        """
        Generate simple dashboard using QuickChart
        
        Args:
            data: Analysis data
            
        Returns:
            Path to generated dashboard HTML file
        """
        self.logger.info("Generating simple dashboard with QuickChart")
        
        # Create simple charts using QuickChart
        charts_html = []
        
        # Example bar chart
        if 'categories' in data:
            chart_url = self.create_quickchart_url('bar', {
                'labels': list(data['categories'].keys()),
                'datasets': [{
                    'label': 'Count',
                    'data': list(data['categories'].values()),
                    'backgroundColor': 'rgba(54, 162, 235, 0.8)'
                }],
                'title': 'Category Distribution'
            })
            charts_html.append(f'<img src="{chart_url}" alt="Category Chart" style="max-width: 100%;">')
        
        # Generate simple HTML
        simple_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CCGL Analytics - Simple Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        .chart {{ margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <h1>🏭 CCGL Analytics Dashboard</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    {''.join(f'<div class="chart">{chart}</div>' for chart in charts_html)}
</body>
</html>
        """
        
        # Save dashboard
        dashboard_file = self.output_dir / f"simple_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(simple_html)
        
        self.logger.info(f"Simple dashboard saved to: {dashboard_file}")
        return str(dashboard_file)