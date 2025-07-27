"""
CCGL Analytics - QuickChart MCP Client
Client for QuickChart service integration via MCP
"""

import asyncio
import json
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from ..utils.logger import get_logger, LoggerMixin

class QuickChartMCPClient(LoggerMixin):
    """MCP client for QuickChart integration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize QuickChart MCP client.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.chart_config = self.config.get('web', {}).get('charts', {})
        self.api_url = self.chart_config.get('api_url', 'https://quickchart.io')
        self.default_width = self.chart_config.get('width', 800)
        self.default_height = self.chart_config.get('height', 600)
    
    async def create_chart(self, chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create chart using QuickChart API.
        
        Args:
            chart_config: Chart configuration
            
        Returns:
            Chart creation response
        """
        try:
            # Simulate QuickChart API call
            chart_url = f"{self.api_url}/chart"
            
            # In a real implementation, this would make an HTTP request
            # For now, we'll simulate the response
            
            chart_id = f"chart_{int(datetime.now().timestamp())}"
            
            response = {
                'status': 'success',
                'chart_id': chart_id,
                'chart_url': f"{chart_url}?chart={chart_id}",
                'config': chart_config,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Chart created successfully: {chart_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to create chart: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def create_bar_chart(self, data: pd.DataFrame, x_column: str, y_column: str,
                              title: str = "Bar Chart") -> Dict[str, Any]:
        """Create bar chart from DataFrame.
        
        Args:
            data: Source DataFrame
            x_column: X-axis column
            y_column: Y-axis column
            title: Chart title
            
        Returns:
            Chart response
        """
        chart_config = {
            'type': 'bar',
            'data': {
                'labels': data[x_column].tolist(),
                'datasets': [{
                    'label': y_column,
                    'data': data[y_column].tolist(),
                    'backgroundColor': 'rgba(54, 162, 235, 0.8)',
                    'borderColor': 'rgba(54, 162, 235, 1)',
                    'borderWidth': 1
                }]
            },
            'options': {
                'title': {
                    'display': True,
                    'text': title
                },
                'responsive': True,
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': x_column
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': y_column
                        }
                    }
                }
            }
        }
        
        return await self.create_chart(chart_config)
    
    async def create_line_chart(self, data: pd.DataFrame, x_column: str, y_column: str,
                               title: str = "Line Chart") -> Dict[str, Any]:
        """Create line chart from DataFrame.
        
        Args:
            data: Source DataFrame
            x_column: X-axis column
            y_column: Y-axis column
            title: Chart title
            
        Returns:
            Chart response
        """
        chart_config = {
            'type': 'line',
            'data': {
                'labels': data[x_column].tolist(),
                'datasets': [{
                    'label': y_column,
                    'data': data[y_column].tolist(),
                    'borderColor': 'rgba(75, 192, 192, 1)',
                    'backgroundColor': 'rgba(75, 192, 192, 0.2)',
                    'fill': False,
                    'tension': 0.1
                }]
            },
            'options': {
                'title': {
                    'display': True,
                    'text': title
                },
                'responsive': True,
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': x_column
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': y_column
                        }
                    }
                }
            }
        }
        
        return await self.create_chart(chart_config)
    
    async def create_pie_chart(self, data: pd.DataFrame, label_column: str, value_column: str,
                              title: str = "Pie Chart") -> Dict[str, Any]:
        """Create pie chart from DataFrame.
        
        Args:
            data: Source DataFrame
            label_column: Labels column
            value_column: Values column
            title: Chart title
            
        Returns:
            Chart response
        """
        chart_config = {
            'type': 'pie',
            'data': {
                'labels': data[label_column].tolist(),
                'datasets': [{
                    'data': data[value_column].tolist(),
                    'backgroundColor': [
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)',
                        'rgba(255, 159, 64, 0.8)'
                    ]
                }]
            },
            'options': {
                'title': {
                    'display': True,
                    'text': title
                },
                'responsive': True
            }
        }
        
        return await self.create_chart(chart_config)
    
    async def create_scatter_plot(self, data: pd.DataFrame, x_column: str, y_column: str,
                                 title: str = "Scatter Plot") -> Dict[str, Any]:
        """Create scatter plot from DataFrame.
        
        Args:
            data: Source DataFrame
            x_column: X-axis column
            y_column: Y-axis column
            title: Chart title
            
        Returns:
            Chart response
        """
        scatter_data = []
        for _, row in data.iterrows():
            scatter_data.append({
                'x': row[x_column],
                'y': row[y_column]
            })
        
        chart_config = {
            'type': 'scatter',
            'data': {
                'datasets': [{
                    'label': f'{y_column} vs {x_column}',
                    'data': scatter_data,
                    'backgroundColor': 'rgba(255, 99, 132, 0.8)',
                    'borderColor': 'rgba(255, 99, 132, 1)'
                }]
            },
            'options': {
                'title': {
                    'display': True,
                    'text': title
                },
                'responsive': True,
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': x_column
                        }
                    },
                    'y': {
                        'title': {
                            'display': True,
                            'text': y_column
                        }
                    }
                }
            }
        }
        
        return await self.create_chart(chart_config)
    
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP chart request.
        
        Args:
            request: MCP request
            
        Returns:
            MCP response
        """
        method = request.get('method')
        params = request.get('params', {})
        
        try:
            if method == 'create_chart':
                return await self.create_chart(params.get('config', {}))
            
            elif method == 'create_bar_chart':
                # Convert data from dict back to DataFrame
                data_dict = params.get('data', {})
                data = pd.DataFrame(data_dict)
                
                return await self.create_bar_chart(
                    data,
                    params.get('x_column'),
                    params.get('y_column'),
                    params.get('title', 'Bar Chart')
                )
            
            elif method == 'create_line_chart':
                data_dict = params.get('data', {})
                data = pd.DataFrame(data_dict)
                
                return await self.create_line_chart(
                    data,
                    params.get('x_column'),
                    params.get('y_column'),
                    params.get('title', 'Line Chart')
                )
            
            elif method == 'create_pie_chart':
                data_dict = params.get('data', {})
                data = pd.DataFrame(data_dict)
                
                return await self.create_pie_chart(
                    data,
                    params.get('label_column'),
                    params.get('value_column'),
                    params.get('title', 'Pie Chart')
                )
            
            elif method == 'create_scatter_plot':
                data_dict = params.get('data', {})
                data = pd.DataFrame(data_dict)
                
                return await self.create_scatter_plot(
                    data,
                    params.get('x_column'),
                    params.get('y_column'),
                    params.get('title', 'Scatter Plot')
                )
            
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