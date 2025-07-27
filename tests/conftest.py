"""
Test configuration for CCGL Analytics
"""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(100),
        'value': np.random.normal(100, 15, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'date': pd.date_range('2023-01-01', periods=100)
    })

@pytest.fixture
def config_dict():
    """Sample configuration for testing."""
    return {
        'database': {
            'primary': {
                'type': 'sqlite',
                'database': ':memory:'
            }
        },
        'analysis': {
            'clustering': {
                'default_algorithm': 'kmeans'
            }
        }
    }