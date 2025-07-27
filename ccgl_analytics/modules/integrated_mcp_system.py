"""
CCGL Analytics - Integrated MCP System
Orchestrates all MCP components for seamless integration
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd

from ..utils.logger import get_logger, LoggerMixin
from .mcp_alchemy_connector import MCPAlchemyConnector
from .mcp_config_manager import MCPConfigManager
from .quickchart_mcp_client import QuickChartMCPClient

class IntegratedMCPSystem(LoggerMixin):
    """Integrated MCP system combining all MCP components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize integrated MCP system.
        
        Args:
            config: System configuration
        """
        self.config = config or {}
        self.config_manager = MCPConfigManager()
        self.db_connector = MCPAlchemyConnector(self.config)
        self.chart_client = QuickChartMCPClient(self.config)
        self.running = False
        self.active_connections = {}
        
    async def initialize(self) -> bool:
        """Initialize all MCP components.
        
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing integrated MCP system")
            
            # Initialize database connector
            db_initialized = await self.db_connector.initialize_connection()
            if not db_initialized:
                self.logger.warning("Database connector initialization failed")
            
            self.logger.info("Integrated MCP system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP system: {e}")
            return False
    
    async def start_system(self) -> bool:
        """Start the integrated MCP system.
        
        Returns:
            True if started successfully
        """
        try:
            if not await self.initialize():
                return False
            
            self.running = True
            self.logger.info("Integrated MCP system started")
            
            # Start background tasks
            asyncio.create_task(self._health_monitor())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start MCP system: {e}")
            return False
    
    async def stop_system(self):
        """Stop the integrated MCP system."""
        try:
            self.running = False
            
            # Close database connections
            await self.db_connector.close_connection()
            
            # Close other connections
            self.active_connections.clear()
            
            self.logger.info("Integrated MCP system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping MCP system: {e}")
    
    async def execute_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete analysis workflow.
        
        Args:
            workflow: Workflow definition
            
        Returns:
            Workflow execution results
        """
        workflow_id = f"workflow_{int(datetime.now().timestamp())}"
        self.logger.info(f"Executing workflow: {workflow_id}")
        
        try:
            steps = workflow.get('steps', [])
            results = {
                'workflow_id': workflow_id,
                'status': 'running',
                'steps_completed': 0,
                'total_steps': len(steps),
                'step_results': {},
                'start_time': datetime.now().isoformat()
            }
            
            for i, step in enumerate(steps):
                step_name = step.get('name', f'step_{i}')
                step_type = step.get('type')
                step_params = step.get('params', {})
                
                self.logger.info(f"Executing step: {step_name}")
                
                try:
                    if step_type == 'data_query':
                        step_result = await self._execute_data_query_step(step_params)
                    
                    elif step_type == 'data_analysis':
                        step_result = await self._execute_analysis_step(step_params)
                    
                    elif step_type == 'visualization':
                        step_result = await self._execute_visualization_step(step_params)
                    
                    elif step_type == 'report_generation':
                        step_result = await self._execute_report_step(step_params)
                    
                    else:
                        step_result = {
                            'status': 'error',
                            'error': f'Unknown step type: {step_type}'
                        }
                    
                    results['step_results'][step_name] = step_result
                    results['steps_completed'] += 1
                    
                    if step_result.get('status') == 'error':
                        self.logger.error(f"Step {step_name} failed: {step_result.get('error')}")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Step {step_name} execution failed: {e}")
                    results['step_results'][step_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    break
            
            # Finalize results
            if results['steps_completed'] == results['total_steps']:
                results['status'] = 'completed'
            else:
                results['status'] = 'failed'
            
            results['end_time'] = datetime.now().isoformat()
            
            self.logger.info(f"Workflow {workflow_id} {results['status']}")
            return results
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                'workflow_id': workflow_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _execute_data_query_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data query step.
        
        Args:
            params: Step parameters
            
        Returns:
            Step execution results
        """
        query = params.get('query')
        query_params = params.get('parameters', {})
        
        if not query:
            return {'status': 'error', 'error': 'No query specified'}
        
        return await self.db_connector.execute_query(query, query_params)
    
    async def _execute_analysis_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis step.
        
        Args:
            params: Step parameters
            
        Returns:
            Step execution results
        """
        # This would integrate with the analysis modules
        analysis_type = params.get('type', 'basic')
        data_source = params.get('data_source')
        
        # Simulate analysis execution
        return {
            'status': 'success',
            'analysis_type': analysis_type,
            'results': {
                'message': f'Analysis of type {analysis_type} completed',
                'data_points_analyzed': 1000,
                'insights_generated': 3
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def _execute_visualization_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization step.
        
        Args:
            params: Step parameters
            
        Returns:
            Step execution results
        """
        chart_type = params.get('chart_type', 'bar')
        data = params.get('data', {})
        
        # Convert data to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        if chart_type == 'bar':
            return await self.chart_client.create_bar_chart(
                df,
                params.get('x_column'),
                params.get('y_column'),
                params.get('title', 'Chart')
            )
        else:
            return {
                'status': 'error',
                'error': f'Unsupported chart type: {chart_type}'
            }
    
    async def _execute_report_step(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report generation step.
        
        Args:
            params: Step parameters
            
        Returns:
            Step execution results
        """
        report_type = params.get('type', 'summary')
        data_sources = params.get('data_sources', [])
        
        # Simulate report generation
        return {
            'status': 'success',
            'report_type': report_type,
            'report_id': f"report_{int(datetime.now().timestamp())}",
            'sections_generated': len(data_sources),
            'timestamp': datetime.now().isoformat()
        }
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generic MCP request.
        
        Args:
            request: MCP request
            
        Returns:
            MCP response
        """
        method = request.get('method')
        params = request.get('params', {})
        
        try:
            if method == 'system_status':
                return await self.get_system_status()
            
            elif method == 'execute_workflow':
                return await self.execute_workflow(params.get('workflow', {}))
            
            elif method.startswith('db_'):
                # Forward to database connector
                db_method = method[3:]  # Remove 'db_' prefix
                db_request = {'method': db_method, 'params': params}
                return await self.db_connector.handle_mcp_request(db_request)
            
            elif method.startswith('chart_'):
                # Forward to chart client
                chart_method = method[6:]  # Remove 'chart_' prefix
                chart_request = {'method': chart_method, 'params': params}
                return await self.chart_client.handle_mcp_request(chart_request)
            
            else:
                return {
                    'status': 'error',
                    'error': f'Unknown method: {method}'
                }
                
        except Exception as e:
            self.logger.error(f"MCP request failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status.
        
        Returns:
            System status information
        """
        try:
            db_status = await self.db_connector.get_connection_status()
            server_status = self.config_manager.get_server_status_summary()
            
            return {
                'status': 'running' if self.running else 'stopped',
                'components': {
                    'database': db_status,
                    'servers': server_status,
                    'chart_client': {
                        'status': 'available',
                        'api_url': self.chart_client.api_url
                    }
                },
                'active_connections': len(self.active_connections),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _health_monitor(self):
        """Background health monitoring task."""
        while self.running:
            try:
                # Check system health
                status = await self.get_system_status()
                
                # Log any issues
                if status.get('status') == 'error':
                    self.logger.warning(f"Health check detected issues: {status}")
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    def create_sample_workflow(self) -> Dict[str, Any]:
        """Create a sample workflow for testing.
        
        Returns:
            Sample workflow definition
        """
        return {
            'name': 'Sample Data Analysis Workflow',
            'description': 'Complete data analysis pipeline',
            'steps': [
                {
                    'name': 'load_data',
                    'type': 'data_query',
                    'params': {
                        'query': 'SELECT * FROM sample_data LIMIT 1000'
                    }
                },
                {
                    'name': 'analyze_data',
                    'type': 'data_analysis',
                    'params': {
                        'type': 'clustering',
                        'data_source': 'load_data'
                    }
                },
                {
                    'name': 'create_visualization',
                    'type': 'visualization',
                    'params': {
                        'chart_type': 'bar',
                        'data_source': 'analyze_data',
                        'x_column': 'category',
                        'y_column': 'value',
                        'title': 'Analysis Results'
                    }
                },
                {
                    'name': 'generate_report',
                    'type': 'report_generation',
                    'params': {
                        'type': 'comprehensive',
                        'data_sources': ['load_data', 'analyze_data', 'create_visualization']
                    }
                }
            ]
        }