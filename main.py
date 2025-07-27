#!/usr/bin/env python3
"""
CCGL Analytics System - Main Analysis Program
Basic analysis mode with command-line interface
"""

import argparse
import sys
import os
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ccgl_analytics import CCGLAnalyzer, get_logger
from ccgl_analytics.utils.logger import setup_logging

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
    """
    required_sections = ['database', 'analysis', 'logging']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required configuration section: {section}")
            return False
    
    return True

def setup_environment():
    """Setup environment variables and paths."""
    # Create necessary directories
    directories = ['logs', 'reports', 'uploads', 'temp']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def perform_data_quality_analysis(analyzer: CCGLAnalyzer, data: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality analysis.
    
    Args:
        analyzer: CCGL analyzer instance
        data: Data to analyze
        
    Returns:
        Data quality analysis results
    """
    logger = get_logger(__name__)
    logger.info("Starting data quality analysis")
    
    results = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_rows': data.duplicated().sum(),
        'data_types': data.dtypes.to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum(),
        'quality_score': 0.0
    }
    
    # Calculate quality score
    missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
    duplicate_ratio = data.duplicated().sum() / len(data)
    
    results['quality_score'] = max(0, 1 - missing_ratio - duplicate_ratio)
    
    logger.info(f"Data quality analysis completed. Score: {results['quality_score']:.2f}")
    return results

def perform_clustering_analysis(analyzer: CCGLAnalyzer, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform clustering analysis.
    
    Args:
        analyzer: CCGL analyzer instance
        data: Data to analyze
        config: Analysis configuration
        
    Returns:
        Clustering analysis results
    """
    logger = get_logger(__name__)
    logger.info("Starting clustering analysis")
    
    # Select numeric columns for clustering
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.empty:
        logger.warning("No numeric columns found for clustering")
        return {'error': 'No numeric data available for clustering'}
    
    # Remove rows with missing values
    clean_data = numeric_data.dropna()
    
    if clean_data.empty:
        logger.warning("No complete rows found after removing missing values")
        return {'error': 'No complete data rows available for clustering'}
    
    clustering_config = config.get('analysis', {}).get('clustering', {})
    algorithm = clustering_config.get('default_algorithm', 'kmeans')
    
    results = {
        'algorithm': algorithm,
        'data_shape': clean_data.shape,
        'features_used': list(clean_data.columns),
        'clusters': [],
        'centroids': [],
        'inertia': None
    }
    
    # Simplified clustering implementation
    if algorithm == 'kmeans':
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clean_data)
            
            # Determine optimal number of clusters (simplified)
            n_clusters = clustering_config.get('n_clusters', 'auto')
            if n_clusters == 'auto':
                n_clusters = min(8, max(2, len(clean_data) // 10))
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            results.update({
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'centroids': kmeans.cluster_centers_.tolist(),
                'inertia': kmeans.inertia_
            })
            
            logger.info(f"KMeans clustering completed with {n_clusters} clusters")
            
        except ImportError:
            logger.error("scikit-learn not available for clustering")
            results['error'] = 'scikit-learn not available'
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            results['error'] = str(e)
    
    return results

def perform_anomaly_detection(analyzer: CCGLAnalyzer, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Perform anomaly detection.
    
    Args:
        analyzer: CCGL analyzer instance
        data: Data to analyze
        config: Analysis configuration
        
    Returns:
        Anomaly detection results
    """
    logger = get_logger(__name__)
    logger.info("Starting anomaly detection")
    
    # Select numeric columns
    numeric_data = data.select_dtypes(include=['number'])
    
    if numeric_data.empty:
        logger.warning("No numeric columns found for anomaly detection")
        return {'error': 'No numeric data available for anomaly detection'}
    
    clean_data = numeric_data.dropna()
    
    if clean_data.empty:
        logger.warning("No complete rows found for anomaly detection")
        return {'error': 'No complete data rows available for anomaly detection'}
    
    anomaly_config = config.get('analysis', {}).get('anomaly_detection', {})
    contamination = anomaly_config.get('contamination', 0.1)
    
    results = {
        'algorithm': 'isolation_forest',
        'data_shape': clean_data.shape,
        'contamination': contamination,
        'anomalies': [],
        'anomaly_scores': []
    }
    
    try:
        from sklearn.ensemble import IsolationForest
        
        # Perform anomaly detection
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = isolation_forest.fit_predict(clean_data)
        anomaly_scores = isolation_forest.score_samples(clean_data)
        
        # Count anomalies
        anomaly_count = (anomaly_labels == -1).sum()
        
        results.update({
            'total_anomalies': int(anomaly_count),
            'anomaly_percentage': float(anomaly_count / len(clean_data) * 100),
            'anomaly_labels': anomaly_labels.tolist(),
            'anomaly_scores': anomaly_scores.tolist()
        })
        
        logger.info(f"Anomaly detection completed. Found {anomaly_count} anomalies ({anomaly_count/len(clean_data)*100:.2f}%)")
        
    except ImportError:
        logger.error("scikit-learn not available for anomaly detection")
        results['error'] = 'scikit-learn not available'
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        results['error'] = str(e)
    
    return results

def generate_analysis_report(results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """Generate comprehensive analysis report.
    
    Args:
        results: Analysis results
        output_file: Optional output file path
        
    Returns:
        Report content
    """
    logger = get_logger(__name__)
    
    report_lines = [
        "# CCGL Analytics System - Analysis Report",
        f"Generated on: {pd.Timestamp.now()}",
        "",
        "## Data Quality Analysis",
    ]
    
    if 'data_quality' in results:
        dq = results['data_quality']
        report_lines.extend([
            f"- Total rows: {dq.get('total_rows', 'N/A')}",
            f"- Total columns: {dq.get('total_columns', 'N/A')}",
            f"- Quality score: {dq.get('quality_score', 0):.2f}",
            f"- Duplicate rows: {dq.get('duplicate_rows', 'N/A')}",
            "",
        ])
    
    if 'clustering' in results:
        clustering = results['clustering']
        report_lines.extend([
            "## Clustering Analysis",
            f"- Algorithm: {clustering.get('algorithm', 'N/A')}",
            f"- Number of clusters: {clustering.get('n_clusters', 'N/A')}",
            f"- Data shape: {clustering.get('data_shape', 'N/A')}",
            "",
        ])
    
    if 'anomaly_detection' in results:
        anomaly = results['anomaly_detection']
        report_lines.extend([
            "## Anomaly Detection",
            f"- Algorithm: {anomaly.get('algorithm', 'N/A')}",
            f"- Total anomalies: {anomaly.get('total_anomalies', 'N/A')}",
            f"- Anomaly percentage: {anomaly.get('anomaly_percentage', 0):.2f}%",
            "",
        ])
    
    report_content = "\n".join(report_lines)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"Report saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    return report_content

def main():
    """Main entry point for the CCGL Analytics System."""
    parser = argparse.ArgumentParser(
        description="CCGL Analytics System - Basic Analysis Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -c config.yml -q "SELECT * FROM sales_data"
  %(prog)s -c config.yml -f data.csv --analysis clustering,anomaly
  %(prog)s -c config.yml -f data.csv --report report.txt
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config.yml',
        help='Configuration file path (default: config.yml)'
    )
    
    parser.add_argument(
        '-q', '--query',
        help='SQL query to fetch data from database'
    )
    
    parser.add_argument(
        '-f', '--file',
        help='Input data file (CSV, Excel, JSON)'
    )
    
    parser.add_argument(
        '--analysis',
        default='quality,clustering,anomaly',
        help='Analysis types to perform (comma-separated): quality,clustering,anomaly,all'
    )
    
    parser.add_argument(
        '--report',
        help='Output report file path'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual analysis'
    )
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Load and validate configuration
    config = load_config(args.config)
    if not validate_config(config):
        sys.exit(1)
    
    # Setup logging
    logging_config = config.get('logging', {})
    setup_logging(
        level='DEBUG' if args.verbose else logging_config.get('level', 'INFO'),
        format_type=logging_config.get('format', 'text'),
        log_file=logging_config.get('file'),
        max_size=logging_config.get('max_size', '100MB'),
        backup_count=logging_config.get('backup_count', 5)
    )
    
    logger = get_logger(__name__)
    logger.info("Starting CCGL Analytics System")
    logger.info(f"Configuration loaded from: {args.config}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual analysis will be performed")
        return
    
    try:
        # Initialize analyzer
        analyzer = CCGLAnalyzer(args.config)
        
        # Load data
        data = None
        if args.query:
            logger.info(f"Fetching data with query: {args.query}")
            # This would use the data connection manager to execute the query
            logger.warning("Database query execution not implemented in this basic version")
        elif args.file:
            logger.info(f"Loading data from file: {args.file}")
            file_path = Path(args.file)
            
            if not file_path.exists():
                logger.error(f"File not found: {args.file}")
                sys.exit(1)
            
            # Load data based on file extension
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(args.file)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(args.file)
            elif file_path.suffix.lower() == '.json':
                data = pd.read_json(args.file)
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                sys.exit(1)
            
            logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
        else:
            logger.error("Either --query or --file must be specified")
            sys.exit(1)
        
        if data is None:
            logger.error("No data available for analysis")
            sys.exit(1)
        
        # Parse analysis types
        analysis_types = [t.strip() for t in args.analysis.split(',')]
        if 'all' in analysis_types:
            analysis_types = ['quality', 'clustering', 'anomaly']
        
        # Perform analysis
        results = {}
        
        if 'quality' in analysis_types:
            results['data_quality'] = perform_data_quality_analysis(analyzer, data)
        
        if 'clustering' in analysis_types:
            results['clustering'] = perform_clustering_analysis(analyzer, data, config)
        
        if 'anomaly' in analysis_types:
            results['anomaly_detection'] = perform_anomaly_detection(analyzer, data, config)
        
        # Generate and display report
        report = generate_analysis_report(results, args.report)
        print("\n" + report)
        
        logger.info("Analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()