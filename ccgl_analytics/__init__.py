"""
CCGL Analytics - 仓储管理系统核心分析模块

提供数据连接、预处理、机器学习分析等核心功能
"""

__version__ = "1.0.0"
__author__ = "CCGL Team"
__email__ = "team@ccgl.com"

# 延迟导入，避免启动时的依赖问题
def get_data_connection(*args, **kwargs):
    from .modules.data_connection import DataConnection
    return DataConnection(*args, **kwargs)

def get_data_preprocessor(*args, **kwargs):
    from .modules.data_preprocessing import DataPreprocessor
    return DataPreprocessor(*args, **kwargs)

def get_analysis_core(*args, **kwargs):
    from .modules.analysis_core import AnalysisCore
    return AnalysisCore(*args, **kwargs)

def get_web_dashboard(*args, **kwargs):
    from .modules.web_dashboard import WebDashboard
    return WebDashboard(*args, **kwargs)

def get_result_output(*args, **kwargs):
    from .modules.result_output import ResultOutput
    return ResultOutput(*args, **kwargs)

# 直接导入工具模块（无外部依赖）
from .utils.logger import setup_logger

__all__ = [
    'get_data_connection',
    'get_data_preprocessor', 
    'get_analysis_core',
    'get_web_dashboard',
    'get_result_output',
    'setup_logger'
]