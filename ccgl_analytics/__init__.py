# CCGL Analytics Package
"""
CCGL仓储管理系统数据分析工程

企业级仓储管理数据分析平台，支持：
- 智能数据分析和机器学习
- MCP架构设计
- AI增强分析功能
- 实时Web仪表板
"""

__version__ = "1.0.0"
__author__ = "CCGL Team"
__email__ = "team@ccgl.com"

from .modules.data_connection import DataConnectionManager
from .modules.data_preprocessing import DataPreprocessor
from .modules.analysis_core import AnalysisCore
from .modules.llm_config_manager import LLMConfigManager
from .modules.web_dashboard import WebDashboard

__all__ = [
    "DataConnectionManager",
    "DataPreprocessor", 
    "AnalysisCore",
    "LLMConfigManager",
    "WebDashboard",
]