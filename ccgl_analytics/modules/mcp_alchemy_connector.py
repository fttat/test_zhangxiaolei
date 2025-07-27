"""
CCGL Analytics - MCP Alchemy Connector
Database connector using MCP protocol and SQLAlchemy
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd

from ..utils.logger import get_logger, LoggerMixin

class MCPAlchemyConnector(LoggerMixin):
    """MCP-compatible database connector using SQLAlchemy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MCP Alchemy connector.
        
        Args:
            config: Connection configuration
        """
        self.config = config
        self.connection_pool = None
        self.mcp_config = config.get('mcp', {})
        self.db_config = config.get('database', {})
        
    async def initialize_connection(self) -> bool:
        """Initialize database connection.
        
        Returns:
            True if successful
        """
        try:
            # Import here to handle optional dependencies
            from .data_connection import DatabaseConnectionPool
            
            self.connection_pool = DatabaseConnectionPool(self.config)
            is_connected = self.connection_pool.test_connection()
            
            if is_connected:
                self.logger.info("MCP Alchemy connector initialized successfully")
            else:
                self.logger.error("Failed to establish database connection")
                
            return is_connected
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP Alchemy connector: {e}")
            return False
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute SQL query via MCP protocol.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        start_time = datetime.now()
        
        try:
            if not self.connection_pool:
                await self.initialize_connection()
            
            # Execute query
            with self.connection_pool.get_connection() as conn:
                result_df = pd.read_sql(query, conn, params=params)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            response = {
                'status': 'success',
                'data': result_df.to_dict('records'),
                'metadata': {
                    'rows_returned': len(result_df),
                    'columns': list(result_df.columns),
                    'execution_time': execution_time,
                    'timestamp': end_time.isoformat()
                }
            }
            
            self.logger.info(f"Query executed successfully: {len(result_df)} rows returned")
            return response
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get table schema information.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table schema information
        """
        try:
            if not self.connection_pool:
                await self.initialize_connection()
            
            # Query to get table schema (MySQL specific)
            schema_query = """
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE,
                COLUMN_DEFAULT,
                COLUMN_KEY,
                EXTRA
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = %s
            ORDER BY ORDINAL_POSITION
            """
            
            result = await self.execute_query(schema_query, {'table_name': table_name})
            
            if result['status'] == 'success':
                return {
                    'status': 'success',
                    'table_name': table_name,
                    'schema': result['data'],
                    'column_count': len(result['data'])
                }
            else:
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to get table schema: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status.
        
        Returns:
            Connection status information
        """
        try:
            if not self.connection_pool:
                return {
                    'status': 'disconnected',
                    'message': 'Connection pool not initialized'
                }
            
            pool_status = self.connection_pool.get_pool_status()
            is_healthy = self.connection_pool.test_connection()
            
            return {
                'status': 'connected' if is_healthy else 'error',
                'pool_info': pool_status,
                'health_check': is_healthy,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def close_connection(self):
        """Close database connection."""
        try:
            if self.connection_pool and hasattr(self.connection_pool, 'engine'):
                self.connection_pool.engine.dispose()
            self.logger.info("MCP Alchemy connector closed")
        except Exception as e:
            self.logger.error(f"Error closing connection: {e}")
    
    # MCP Protocol Methods
    async def handle_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol request.
        
        Args:
            request: MCP request
            
        Returns:
            MCP response
        """
        method = request.get('method')
        params = request.get('params', {})
        
        if method == 'query':
            return await self.execute_query(
                params.get('sql'),
                params.get('parameters')
            )
        
        elif method == 'schema':
            return await self.get_table_schema(params.get('table_name'))
        
        elif method == 'status':
            return await self.get_connection_status()
        
        else:
            return {
                'status': 'error',
                'error': f'Unknown method: {method}'
            }
    
    def create_mcp_response(self, request_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Create MCP protocol response.
        
        Args:
            request_id: Request identifier
            result: Result data
            
        Returns:
            MCP response
        """
        return {
            'jsonrpc': '2.0',
            'id': request_id,
            'result': result
        }