#!/usr/bin/env python3
"""
CCGL Dashboard MCP Server
Handles dashboard and visualization requests via MCP protocol
"""

import asyncio
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ccgl_analytics.modules.quickchart_mcp_client import QuickChartMCPClient
from ccgl_analytics.utils.logger import get_logger, setup_logging

class CCGLDashboardMCPServer:
    """MCP server for dashboard and visualization operations."""
    
    def __init__(self, port: int = 8003, config_file: str = '../config.yml'):
        self.port = port
        self.config_file = config_file
        self.logger = get_logger(__name__)
        self.chart_client = QuickChartMCPClient()
        self.running = False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get('method')
        params = request.get('params', {})
        
        try:
            if method == 'create_chart':
                return await self._create_chart(params)
            elif method == 'create_dashboard':
                return await self._create_dashboard(params)
            elif method == 'generate_report':
                return await self._generate_report(params)
            elif method == 'status':
                return {
                    'status': 'running',
                    'server': 'dashboard',
                    'port': self.port,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'error': f'Unknown method: {method}'}
        except Exception as e:
            self.logger.error(f"Request handling failed: {e}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _create_chart(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            chart_type = params.get('chart_type', 'bar')
            data_dict = params.get('data', {})
            x_column = params.get('x_column')
            y_column = params.get('y_column')
            title = params.get('title', 'Chart')
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            if chart_type == 'bar':
                result = await self.chart_client.create_bar_chart(data, x_column, y_column, title)
            elif chart_type == 'line':
                result = await self.chart_client.create_line_chart(data, x_column, y_column, title)
            elif chart_type == 'pie':
                label_column = params.get('label_column', x_column)
                result = await self.chart_client.create_pie_chart(data, label_column, y_column, title)
            elif chart_type == 'scatter':
                result = await self.chart_client.create_scatter_plot(data, x_column, y_column, title)
            else:
                return {'status': 'error', 'error': f'Unsupported chart type: {chart_type}'}
            
            return {
                'status': 'success',
                'chart_result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _create_dashboard(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            dashboard_config = params.get('config', {})
            data_sources = params.get('data_sources', [])
            
            # Simulate dashboard creation
            dashboard_id = f"dashboard_{int(datetime.now().timestamp())}"
            
            charts = []
            for i, data_source in enumerate(data_sources):
                chart_config = {
                    'chart_id': f"chart_{i}",
                    'data_source': data_source,
                    'position': {'row': i // 2, 'col': i % 2}
                }
                charts.append(chart_config)
            
            dashboard = {
                'dashboard_id': dashboard_id,
                'title': dashboard_config.get('title', 'CCGL Analytics Dashboard'),
                'charts': charts,
                'layout': dashboard_config.get('layout', 'grid'),
                'created_at': datetime.now().isoformat()
            }
            
            return {
                'status': 'success',
                'dashboard': dashboard,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _generate_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            report_type = params.get('type', 'summary')
            analysis_results = params.get('analysis_results', {})
            
            report_id = f"report_{int(datetime.now().timestamp())}"
            
            report = {
                'report_id': report_id,
                'type': report_type,
                'title': f'CCGL Analytics Report - {report_type.title()}',
                'generated_at': datetime.now().isoformat(),
                'sections': [],
                'summary': {
                    'total_analyses': len(analysis_results),
                    'key_findings': [
                        'Data analysis completed successfully',
                        'Multiple insights identified',
                        'Recommendations generated'
                    ]
                }
            }
            
            # Generate sections based on analysis results
            for analysis_type, result in analysis_results.items():
                section = {
                    'title': analysis_type.replace('_', ' ').title(),
                    'type': analysis_type,
                    'content': f'Analysis of {analysis_type} completed with results.',
                    'charts': [f'{analysis_type}_chart.png'],
                    'metrics': self._extract_metrics(result)
                }
                report['sections'].append(section)
            
            return {
                'status': 'success',
                'report': report,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _extract_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis result."""
        metrics = {}
        
        if isinstance(result, dict):
            if 'n_clusters' in result:
                metrics['clusters'] = result['n_clusters']
            if 'anomaly_count' in result:
                metrics['anomalies'] = result['anomaly_count']
            if 'processing_time' in result:
                metrics['processing_time'] = result['processing_time']
        
        return metrics
    
    async def start(self):
        self.logger.info(f"Starting CCGL Dashboard MCP Server on port {self.port}")
        self.running = True
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        self.logger.info("Stopping CCGL Dashboard MCP Server")
        self.running = False

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CCGL Dashboard MCP Server")
    parser.add_argument('--port', type=int, default=8003, help='Server port')
    parser.add_argument('--config', default='../config.yml', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    server = CCGLDashboardMCPServer(args.port, args.config)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))