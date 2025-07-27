"""
CCGL Analytics - 仓储管理系统核心分析模块

提供数据连接、预处理、机器学习分析等核心功能
"""

__version__ = "1.0.0"
__author__ = "CCGL Team"
__email__ = "team@ccgl.com"

from .modules.data_connection import DataConnection
from .modules.data_preprocessing import DataPreprocessor
from .modules.analysis_core import AnalysisCore
from .modules.web_dashboard import WebDashboard
from .modules.result_output import ResultOutput
from .utils.logger import setup_logger

__all__ = [
    'DataConnection',
    'DataPreprocessor', 
    'AnalysisCore',
    'WebDashboard',
    'ResultOutput',
    'setup_logger'
]