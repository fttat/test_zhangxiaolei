#!/usr/bin/env python3
"""
CCGL Data Preprocessing MCP Server
Handles data preprocessing requests via MCP protocol
"""

import asyncio
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ccgl_analytics.modules.data_preprocessing import DataPreprocessor
from ccgl_analytics.utils.logger import get_logger, setup_logging

class CCGLPreprocessingMCPServer:
    """MCP server for data preprocessing operations."""
    
    def __init__(self, port: int = 8001, config_file: str = '../config.yml'):
        """Initialize preprocessing MCP server.
        
        Args:
            port: Server port
            config_file: Configuration file path
        """
        self.port = port
        self.config_file = config_file
        self.logger = get_logger(__name__)
        self.preprocessor = DataPreprocessor()
        self.running = False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP preprocessing request.
        
        Args:
            request: MCP request
            
        Returns:
            MCP response
        """
        method = request.get('method')
        params = request.get('params', {})
        
        try:
            if method == 'preprocess_data':
                return await self._preprocess_data(params)
            
            elif method == 'analyze_missing_values':
                return await self._analyze_missing_values(params)
            
            elif method == 'detect_outliers':
                return await self._detect_outliers(params)
            
            elif method == 'scale_features':
                return await self._scale_features(params)
            
            elif method == 'status':
                return {
                    'status': 'running',
                    'server': 'preprocessing',
                    'port': self.port,
                    'timestamp': datetime.now().isoformat()
                }
            
            else:
                return {
                    'status': 'error',
                    'error': f'Unknown method: {method}'
                }
                
        except Exception as e:
            self.logger.error(f"Request handling failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _preprocess_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data preprocessing request.
        
        Args:
            params: Request parameters
            
        Returns:
            Preprocessing results
        """
        try:
            # Extract parameters
            data_dict = params.get('data', {})
            target_column = params.get('target_column')
            steps = params.get('steps', ['missing_values', 'outliers', 'scaling'])
            
            # Convert data to DataFrame
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            # Perform preprocessing
            results = self.preprocessor.preprocess_data(
                data, 
                target_column=target_column, 
                steps=steps
            )
            
            # Convert processed data back to dict for JSON serialization
            processed_data = results['data'].to_dict('records')
            
            return {
                'status': 'success',
                'original_shape': results['original_shape'],
                'final_shape': results['final_shape'],
                'steps_completed': results['steps_completed'],
                'processed_data': processed_data,
                'metadata': results['preprocessing_metadata'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_missing_values(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze missing values in data.
        
        Args:
            params: Request parameters
            
        Returns:
            Missing value analysis results
        """
        try:
            data_dict = params.get('data', {})
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            analysis = self.preprocessor.missing_handler.analyze_missing_patterns(data)
            
            return {
                'status': 'success',
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _detect_outliers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect outliers in data.
        
        Args:
            params: Request parameters
            
        Returns:
            Outlier detection results
        """
        try:
            data_dict = params.get('data', {})
            method = params.get('method', 'iqr')
            columns = params.get('columns')
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            self.preprocessor.outlier_detector.method = method
            results = self.preprocessor.outlier_detector.detect_outliers(data, columns)
            
            return {
                'status': 'success',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _scale_features(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Scale features in data.
        
        Args:
            params: Request parameters
            
        Returns:
            Feature scaling results
        """
        try:
            data_dict = params.get('data', {})
            method = params.get('method', 'standard')
            columns = params.get('columns')
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            self.preprocessor.scaler.method = method
            scaled_data = self.preprocessor.scaler.fit_transform(data, columns)
            
            return {
                'status': 'success',
                'scaled_data': scaled_data.to_dict('records'),
                'method': method,
                'columns_scaled': columns or list(data.select_dtypes(include=['number']).columns),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def start(self):
        """Start the MCP server."""
        self.logger.info(f"Starting CCGL Preprocessing MCP Server on port {self.port}")
        self.running = True
        
        # Simulate server running
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        """Stop the MCP server."""
        self.logger.info("Stopping CCGL Preprocessing MCP Server")
        self.running = False

async def main():
    """Main function for preprocessing MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CCGL Preprocessing MCP Server")
    parser.add_argument('--port', type=int, default=8001, help='Server port')
    parser.add_argument('--config', default='../config.yml', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    # Create and start server
    server = CCGLPreprocessingMCPServer(args.port, args.config)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))