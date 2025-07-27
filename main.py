#!/usr/bin/env python3
"""
CCGL Analytics - Traditional Main Entry Point
Basic data analysis without MCP architecture
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import yaml

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from ccgl_analytics.modules.data_connection import DataConnection, create_sample_data
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor
from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.utils.logger import get_logger, setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Create default config
        default_config = {
            'data_source': {
                'type': 'file',  # or 'mysql'
                'file_path': 'sample_data.csv',
                'database': {
                    'host': 'localhost',
                    'port': 3306,
                    'user': 'root',
                    'password': '',
                    'database': 'ccgl_warehouse'
                }
            },
            'preprocessing': {
                'steps': ['quality_check', 'handle_missing', 'detect_outliers', 'normalize_data'],
                'missing_strategy': 'auto',
                'outlier_method': 'iqr',
                'normalization_method': 'standard'
            },
            'analysis': {
                'include_clustering': True,
                'include_anomaly_detection': True,
                'include_dimensionality_reduction': True,
                'clustering': {
                    'max_clusters': 10
                },
                'anomaly_detection': {
                    'contamination': 0.1
                }
            },
            'output': {
                'save_results': True,
                'output_dir': 'results',
                'formats': ['csv', 'json']
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'ccgl_analytics.log'
            }
        }
        
        # Save default config
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"Created default configuration file: {config_path}")
        return default_config
    
    # Load existing config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_results(results: Dict[str, Any], output_config: Dict[str, Any]):
    """Save analysis results to files"""
    output_dir = Path(output_config.get('output_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    formats = output_config.get('formats', ['json'])
    
    # Custom JSON serializer to handle numpy types
    def json_serializer(obj):
        """Custom JSON serializer for numpy types and pandas objects"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'dtype'):
            if obj.dtype.kind in ['i', 'u']:  # integer types
                return int(obj)
            elif obj.dtype.kind == 'f':  # floating types
                return float(obj)
            elif obj.dtype.kind == 'b':  # boolean types
                return bool(obj)
            else:
                return str(obj)
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return str(obj)
    
    # Save summary results
    summary = {
        'analysis_summary': results.get('analysis_summary', {}),
        'dataset_info': results.get('dataset_info', {}),
        'preprocessing_summary': results.get('preprocessing_summary', {})
    }
    
    if 'json' in formats:
        import json
        with open(output_dir / 'analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=json_serializer)
    
    # Save detailed results for each analysis type
    if 'clustering' in results:
        clustering_results = results['clustering']
        for method, data in clustering_results.items():
            if 'clustered_data' in data and 'csv' in formats:
                data['clustered_data'].to_csv(output_dir / f'clustering_{method}_results.csv', index=False)
    
    if 'anomaly_detection' in results:
        anomaly_results = results['anomaly_detection']
        for method, data in anomaly_results.items():
            if 'anomaly_data' in data and 'csv' in formats:
                data['anomaly_data'].to_csv(output_dir / f'anomaly_{method}_results.csv', index=False)
    
    print(f"Results saved to: {output_dir}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='CCGL Analytics - Warehouse Data Analysis')
    parser.add_argument('-c', '--config', default='config.yml', help='Configuration file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data file')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else config.get('logging', {}).get('level', 'INFO')
    log_file = config.get('logging', {}).get('log_file')
    setup_logging(log_level, log_file)
    
    logger = get_logger("main")
    logger.info("Starting CCGL Analytics")
    
    try:
        # Create sample data if requested
        if args.create_sample:
            logger.info("Creating sample data")
            sample_data = create_sample_data()
            sample_data.to_csv('sample_data.csv', index=False)
            logger.info("Sample data created: sample_data.csv")
            return 0
        
        # Initialize data connection
        data_config = config.get('data_source', {})
        data_connection = DataConnection(data_config)
        
        # Test connection
        if not data_connection.test_connection():
            logger.error("Failed to establish data connection")
            return 1
        
        # Load data
        if data_config.get('type') == 'mysql':
            # Load from database
            table_name = data_config.get('table', 'warehouse_inventory')
            df = data_connection.load_data(table_name)
        else:
            # Load from file
            file_path = data_config.get('file_path', 'sample_data.csv')
            if not Path(file_path).exists():
                logger.info("Sample data file not found. Creating sample data...")
                sample_data = create_sample_data()
                sample_data.to_csv(file_path, index=False)
                logger.info(f"Sample data created: {file_path}")
            
            df = data_connection.load_data(file_path)
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Initialize preprocessor
        preprocessing_config = config.get('preprocessing', {})
        preprocessor = DataPreprocessor(preprocessing_config)
        
        # Preprocess data
        steps = preprocessing_config.get('steps', ['quality_check', 'handle_missing'])
        processed_df = preprocessor.preprocess(df, steps)
        
        logger.info("Data preprocessing completed")
        
        # Initialize analysis core
        analysis_config = config.get('analysis', {})
        analysis_core = AnalysisCore(analysis_config)
        
        # Perform comprehensive analysis
        analysis_results = analysis_core.comprehensive_analysis(
            processed_df,
            include_clustering=analysis_config.get('include_clustering', True),
            include_anomaly_detection=analysis_config.get('include_anomaly_detection', True),
            include_dimensionality_reduction=analysis_config.get('include_dimensionality_reduction', True)
        )
        
        # Generate summary
        analysis_summary = analysis_core.get_analysis_summary()
        preprocessing_summary = preprocessor.get_preprocessing_summary()
        
        # Combine all results
        final_results = {
            'analysis_summary': analysis_summary,
            'preprocessing_summary': preprocessing_summary,
            'dataset_info': analysis_results.get('dataset_info', {}),
            'clustering': analysis_results.get('clustering', {}),
            'anomaly_detection': analysis_results.get('anomaly_detection', {}),
            'dimensionality_reduction': analysis_results.get('dimensionality_reduction', {})
        }
        
        # Save results
        output_config = config.get('output', {})
        # Disable JSON save temporarily due to serialization issues
        if output_config.get('save_results', True):
            try:
                save_results(final_results, output_config)
            except Exception as e:
                logger.warning(f"Failed to save JSON results: {e}. Continuing with analysis display.")
        
        # Print summary
        print("\n=== CCGL Analytics Results Summary ===")
        print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Preprocessing steps: {len(preprocessing_summary.get('steps_applied', []))}")
        
        if 'clustering' in final_results:
            print("\nClustering Results:")
            for method, results in final_results['clustering'].items():
                if 'n_clusters' in results:
                    print(f"  {method.upper()}: {results['n_clusters']} clusters")
                    if 'silhouette_score' in results and results['silhouette_score']:
                        print(f"    Silhouette Score: {results['silhouette_score']:.3f}")
        
        if 'anomaly_detection' in final_results:
            print("\nAnomaly Detection Results:")
            for method, results in final_results['anomaly_detection'].items():
                if 'anomalies_detected' in results:
                    print(f"  {method.upper()}: {results['anomalies_detected']} anomalies "
                          f"({results.get('anomaly_percentage', 0):.2f}%)")
        
        if 'dimensionality_reduction' in final_results:
            print("\nDimensionality Reduction Results:")
            dim_results = final_results['dimensionality_reduction']
            if 'pca' in dim_results:
                pca_result = dim_results['pca']
                if 'explained_variance_ratio' in pca_result:
                    total_variance = sum(pca_result['explained_variance_ratio'])
                    print(f"  PCA: {total_variance:.3f} variance explained by 2 components")
        
        logger.info("Analysis completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())