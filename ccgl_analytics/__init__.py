"""
CCGL Analytics System
Centralized Control and Group Learning - Advanced Analytics Platform

This package provides comprehensive data analytics capabilities including:
- Data quality assessment and preprocessing
- Machine learning analysis (clustering, anomaly detection, dimensionality reduction)
- LLM integration for natural language querying
- MCP architecture for distributed processing
- Web dashboard for interactive visualization
"""

__version__ = "1.0.0"
__author__ = "CCGL Analytics Team"
__email__ = "support@ccgl-analytics.com"
__license__ = "MIT"

# Core imports
from .modules.data_connection import DataConnectionManager
from .modules.data_preprocessing import DataPreprocessor
from .modules.analysis_core import AnalysisCore
from .modules.llm_config_manager import LLMConfigManager

# MCP imports
from .modules.mcp_alchemy_connector import MCPAlchemyConnector
from .modules.mcp_config_manager import MCPConfigManager
from .modules.integrated_mcp_system import IntegratedMCPSystem

# Utilities
from .utils.logger import get_logger

# Optional imports
try:
    from .modules.web_dashboard import WebDashboard
    WEB_DASHBOARD_AVAILABLE = True
except ImportError:
    WEB_DASHBOARD_AVAILABLE = False

# Main analyzer class
class CCGLAnalyzer:
    """Main analyzer class for CCGL Analytics System."""
    
    def __init__(self, config_file=None):
        """Initialize the CCGL Analyzer.
        
        Args:
            config_file (str): Path to configuration file
        """
        self.logger = get_logger(__name__)
        self.config_file = config_file
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        self.data_connection = DataConnectionManager(self.config_file)
        self.preprocessor = DataPreprocessor()
        self.analysis_core = AnalysisCore()
        self.llm_manager = LLMConfigManager()
        
        # Initialize web dashboard if available
        if WEB_DASHBOARD_AVAILABLE:
            self.web_dashboard = WebDashboard()
        else:
            self.web_dashboard = None
            self.logger.warning("Web dashboard not available (missing dependencies)")
        
    def analyze_data(self, query=None, data=None, analysis_type=None):
        """Perform comprehensive data analysis.
        
        Args:
            query (str): SQL query to fetch data
            data (DataFrame): Data to analyze
            analysis_type (list): Types of analysis to perform
            
        Returns:
            AnalysisResults: Comprehensive analysis results
        """
        self.logger.info("Starting data analysis")
        
        # Implementation will be in the respective modules
        results = {
            'data_quality': None,
            'clustering': None,
            'anomaly_detection': None,
            'insights': None
        }
        
        return results
    
    def generate_report(self, results):
        """Generate analysis report.
        
        Args:
            results: Analysis results
            
        Returns:
            dict: Generated report
        """
        if self.web_dashboard:
            return self.web_dashboard.generate_report(results)
        else:
            # Simple text report if web dashboard not available
            return {
                'title': 'CCGL Analytics Report',
                'summary': 'Analysis completed successfully',
                'results': results
            }

# Package-level convenience functions
def get_analyzer(config_file=None):
    """Get a configured CCGL Analyzer instance."""
    return CCGLAnalyzer(config_file)

def quick_analysis(data, analysis_type=['clustering', 'quality']):
    """Perform quick analysis on data."""
    analyzer = CCGLAnalyzer()
    return analyzer.analyze_data(data=data, analysis_type=analysis_type)

# Export all public classes and functions
__all__ = [
    'CCGLAnalyzer',
    'DataConnectionManager',
    'DataPreprocessor', 
    'AnalysisCore',
    'LLMConfigManager',
    'MCPAlchemyConnector',
    'MCPConfigManager',
    'IntegratedMCPSystem',
    'get_analyzer',
    'quick_analysis',
    'get_logger',
    'WEB_DASHBOARD_AVAILABLE'
]

# Add WebDashboard to exports if available
if WEB_DASHBOARD_AVAILABLE:
    __all__.append('WebDashboard')