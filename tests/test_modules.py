"""
Unit tests for CCGL Analytics core modules
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccgl_analytics.modules.data_connection import DataConnection, create_sample_data
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor, DataQualityAssessment
from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.utils.logger import get_logger


class TestDataConnection:
    """Test suite for data connection module"""
    
    def test_create_sample_data(self):
        """Test sample data creation"""
        df = create_sample_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert len(df.columns) == 10
        assert 'product_id' in df.columns
        assert 'total_value' in df.columns
    
    def test_file_data_connection(self, tmp_path):
        """Test file-based data connection"""
        # Create test data
        test_data = create_sample_data()
        test_file = tmp_path / "test_data.csv"
        test_data.to_csv(test_file, index=False)
        
        # Test connection
        config = {'type': 'file'}
        connection = DataConnection(config)
        
        # Test loading
        loaded_data = connection.load_data(str(test_file))
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(test_data)
        assert list(loaded_data.columns) == list(test_data.columns)
    
    def test_connection_test(self):
        """Test connection testing functionality"""
        config = {'type': 'file'}
        connection = DataConnection(config)
        
        assert connection.test_connection() is True


class TestDataPreprocessing:
    """Test suite for data preprocessing module"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for tests"""
        return create_sample_data()
    
    @pytest.fixture
    def preprocessor(self):
        """Fixture providing preprocessor instance"""
        return DataPreprocessor()
    
    @pytest.fixture
    def quality_assessor(self):
        """Fixture providing quality assessor instance"""
        return DataQualityAssessment()
    
    def test_quality_assessment(self, quality_assessor, sample_data):
        """Test data quality assessment"""
        report = quality_assessor.assess_quality(sample_data)
        
        assert isinstance(report, dict)
        assert 'overview' in report
        assert 'completeness' in report
        assert 'overall_score' in report
        assert 0 <= report['overall_score'] <= 1
    
    def test_missing_value_handling(self, preprocessor, sample_data):
        """Test missing value handling"""
        # Add some missing values
        test_data = sample_data.copy()
        test_data.loc[:10, 'quantity'] = np.nan
        
        processed = preprocessor.handle_missing_values(test_data)
        
        assert processed['quantity'].isnull().sum() == 0
    
    def test_outlier_detection(self, preprocessor, sample_data):
        """Test outlier detection and handling"""
        processed = preprocessor.detect_and_handle_outliers(sample_data)
        
        assert isinstance(processed, pd.DataFrame)
        assert len(processed) == len(sample_data)
    
    def test_normalization(self, preprocessor, sample_data):
        """Test data normalization"""
        processed = preprocessor.normalize_data(sample_data)
        
        numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
        
        # Check that numeric columns are normalized (mean ~0, std ~1)
        for col in numeric_cols:
            if col in processed.columns:
                assert abs(processed[col].mean()) < 0.1
                assert abs(processed[col].std() - 1.0) < 0.1
    
    def test_preprocessing_pipeline(self, preprocessor, sample_data):
        """Test complete preprocessing pipeline"""
        steps = ['quality_check', 'handle_missing', 'detect_outliers']
        processed = preprocessor.preprocess(sample_data, steps)
        
        assert isinstance(processed, pd.DataFrame)
        
        # Check preprocessing summary
        summary = preprocessor.get_preprocessing_summary()
        assert 'total_steps' in summary
        assert summary['total_steps'] == len(steps)


class TestAnalysisCore:
    """Test suite for analysis core module"""
    
    @pytest.fixture
    def analysis_core(self):
        """Fixture providing analysis core instance"""
        return AnalysisCore()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for analysis"""
        return create_sample_data()
    
    def test_comprehensive_analysis(self, analysis_core, sample_data):
        """Test comprehensive analysis functionality"""
        results = analysis_core.comprehensive_analysis(
            sample_data,
            include_clustering=True,
            include_anomaly_detection=True,
            include_dimensionality_reduction=True
        )
        
        assert isinstance(results, dict)
        assert 'dataset_info' in results
        assert 'clustering' in results
        assert 'anomaly_detection' in results
        assert 'dimensionality_reduction' in results
    
    def test_clustering_analysis(self, analysis_core, sample_data):
        """Test clustering analysis"""
        # Test KMeans clustering
        kmeans_result = analysis_core.clustering.perform_kmeans_clustering(sample_data)
        
        assert 'n_clusters' in kmeans_result
        assert 'silhouette_score' in kmeans_result
        assert 'clustered_data' in kmeans_result
        assert kmeans_result['n_clusters'] > 0
        
        # Test DBSCAN clustering
        dbscan_result = analysis_core.clustering.perform_dbscan_clustering(sample_data)
        
        assert 'n_clusters' in dbscan_result
        assert 'clustered_data' in dbscan_result
    
    def test_anomaly_detection(self, analysis_core, sample_data):
        """Test anomaly detection"""
        # Test isolation forest
        iso_result = analysis_core.anomaly_detection.detect_anomalies(
            sample_data, method='isolation_forest'
        )
        
        assert 'anomalies_detected' in iso_result
        assert 'anomaly_percentage' in iso_result
        assert 'full_data' in iso_result
        
        # Test statistical method
        stat_result = analysis_core.anomaly_detection.detect_anomalies(
            sample_data, method='statistical'
        )
        
        assert 'anomalies_detected' in stat_result
        assert 'anomaly_percentage' in stat_result
    
    def test_dimensionality_reduction(self, analysis_core, sample_data):
        """Test dimensionality reduction"""
        # Test PCA
        pca_result = analysis_core.dimensionality_reduction.perform_pca(sample_data)
        
        assert 'explained_variance_ratio' in pca_result
        assert 'transformed_data' in pca_result
        assert len(pca_result['explained_variance_ratio']) == 2  # Default n_components
        
        # Test t-SNE
        tsne_result = analysis_core.dimensionality_reduction.perform_tsne(sample_data)
        
        assert 'transformed_data' in tsne_result
        assert 'perplexity' in tsne_result
    
    def test_analysis_summary(self, analysis_core, sample_data):
        """Test analysis summary generation"""
        # Run analysis first
        analysis_core.comprehensive_analysis(sample_data)
        
        summary = analysis_core.get_analysis_summary()
        
        assert isinstance(summary, dict)
        assert 'dataset_overview' in summary
        assert 'analyses_performed' in summary


class TestLogger:
    """Test suite for logger utility"""
    
    def test_logger_creation(self):
        """Test logger creation and basic functionality"""
        logger = get_logger("test_logger")
        
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_logger_methods(self):
        """Test logger methods"""
        logger = get_logger("test_methods")
        
        # Test that methods don't raise exceptions
        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
        except Exception as e:
            pytest.fail(f"Logger methods raised exception: {e}")


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_workflow(self):
        """Test complete analysis workflow"""
        # Create test data
        df = create_sample_data()
        
        # Initialize components
        preprocessor = DataPreprocessor()
        analysis_core = AnalysisCore()
        
        # Run preprocessing
        processed_df = preprocessor.preprocess(df, ['quality_check', 'handle_missing'])
        
        # Run analysis
        results = analysis_core.comprehensive_analysis(processed_df)
        
        # Verify results
        assert isinstance(results, dict)
        assert 'dataset_info' in results
        
        # Check that analysis ran successfully
        summary = analysis_core.get_analysis_summary()
        assert len(summary['analyses_performed']) > 0
    
    def test_end_to_end_file_processing(self, tmp_path):
        """Test end-to-end file processing"""
        # Create test file
        test_data = create_sample_data()
        test_file = tmp_path / "test_data.csv"
        test_data.to_csv(test_file, index=False)
        
        # Initialize components
        config = {'type': 'file'}
        connection = DataConnection(config)
        preprocessor = DataPreprocessor()
        analysis_core = AnalysisCore()
        
        # Load data
        df = connection.load_data(str(test_file))
        
        # Process data
        processed_df = preprocessor.preprocess(df)
        
        # Analyze data
        results = analysis_core.comprehensive_analysis(processed_df)
        
        # Verify complete workflow
        assert isinstance(results, dict)
        assert len(results) > 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])