"""
Unit tests for analysis core module
"""

import pytest
import pandas as pd
import numpy as np
from ccgl_analytics.modules.analysis_core import AnalysisCore, ClusteringAnalyzer

class TestAnalysisCore:
    """Test AnalysisCore class."""
    
    def test_initialization(self):
        """Test analysis core initialization."""
        core = AnalysisCore()
        assert core is not None
        assert core.clustering_analyzer is not None
    
    def test_comprehensive_analysis(self, sample_data):
        """Test comprehensive analysis."""
        core = AnalysisCore()
        
        result = core.comprehensive_analysis(sample_data, ['clustering'])
        
        assert 'results' in result
        assert 'summary' in result

class TestClusteringAnalyzer:
    """Test ClusteringAnalyzer class."""
    
    def test_initialization(self):
        """Test clustering analyzer initialization."""
        analyzer = ClusteringAnalyzer()
        assert analyzer is not None
    
    def test_kmeans_clustering(self, sample_data):
        """Test K-means clustering."""
        analyzer = ClusteringAnalyzer()
        
        # Select only numeric data for clustering
        numeric_data = sample_data.select_dtypes(include=[np.number])
        
        result = analyzer.perform_clustering(numeric_data, algorithm='kmeans', n_clusters=3)
        
        assert 'algorithm' in result
        assert result['algorithm'] == 'kmeans'