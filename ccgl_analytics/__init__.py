"""
CCGL Analytics - Enterprise Warehouse Management Data Analysis Platform
========================================================================

A comprehensive data analysis platform featuring:
- MCP (Model Context Protocol) architecture
- AI-powered insights with multiple LLM support
- Advanced machine learning capabilities
- Real-time web dashboard
- Enterprise-grade deployment

Author: CCGL Analytics Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "CCGL Analytics Team"
__email__ = "analytics@ccgl.com"

# Core module imports
from .modules.data_connection import DataConnection
from .modules.data_preprocessing import DataPreprocessor
from .modules.analysis_core import AnalysisCore
from .utils.logger import get_logger

__all__ = [
    "DataConnection",
    "DataPreprocessor", 
    "AnalysisCore",
    "get_logger",
]