"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
from ccgl_analytics.modules.data_preprocessing import DataPreprocessor, MissingValueHandler

class TestDataPreprocessor:
    """Test DataPreprocessor class."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
        assert preprocessor.missing_handler is not None
    
    def test_preprocess_data(self, sample_data):
        """Test data preprocessing."""
        preprocessor = DataPreprocessor()
        
        # Add some missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0:5, 'value'] = np.nan
        
        result = preprocessor.preprocess_data(data_with_missing)
        
        assert 'data' in result
        assert 'steps_completed' in result
        assert isinstance(result['data'], pd.DataFrame)

class TestMissingValueHandler:
    """Test MissingValueHandler class."""
    
    def test_analyze_missing_patterns(self, sample_data):
        """Test missing value pattern analysis."""
        handler = MissingValueHandler()
        
        # Add missing values
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0:10, 'value'] = np.nan
        
        analysis = handler.analyze_missing_patterns(data_with_missing)
        
        assert 'column_analysis' in analysis
        assert 'total_missing_values' in analysis
        assert analysis['total_missing_values'] > 0