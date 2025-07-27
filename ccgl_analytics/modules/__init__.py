"""
CCGL Analytics Modules
Core modules for data analysis, processing, and integration
"""

from .data_connection import DataConnectionManager
from .data_preprocessing import DataPreprocessor
from .analysis_core import AnalysisCore
from .llm_config_manager import LLMConfigManager

# Optional imports for modules with external dependencies
try:
    from .web_dashboard import WebDashboard
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False

from .mcp_alchemy_connector import MCPAlchemyConnector
from .mcp_config_manager import MCPConfigManager
from .quickchart_mcp_client import QuickChartMCPClient
from .integrated_mcp_system import IntegratedMCPSystem
from .mcp_client_orchestrator import MCPClientOrchestrator
from .relationship_extraction import RelationshipExtraction
from .result_output import ResultOutput

__all__ = [
    'DataConnectionManager',
    'DataPreprocessor',
    'AnalysisCore',
    'LLMConfigManager',
    'MCPAlchemyConnector',
    'MCPConfigManager',
    'QuickChartMCPClient',
    'IntegratedMCPSystem',
    'MCPClientOrchestrator',
    'RelationshipExtraction',
    'ResultOutput',
    'WEB_DASHBOARD_AVAILABLE'
]

# Only add WebDashboard to __all__ if it's available
if WEB_DASHBOARD_AVAILABLE:
    __all__.append('WebDashboard')