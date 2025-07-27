#!/usr/bin/env python3
"""
CCGL Machine Learning MCP Server
Handles ML analysis requests via MCP protocol
"""

import asyncio
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.utils.logger import get_logger, setup_logging

class CCGLMLMCPServer:
    """MCP server for machine learning operations."""
    
    def __init__(self, port: int = 8002, config_file: str = '../config.yml'):
        self.port = port
        self.config_file = config_file
        self.logger = get_logger(__name__)
        self.analysis_core = AnalysisCore()
        self.running = False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get('method')
        params = request.get('params', {})
        
        try:
            if method == 'perform_clustering':
                return await self._perform_clustering(params)
            elif method == 'detect_anomalies':
                return await self._detect_anomalies(params)
            elif method == 'reduce_dimensions':
                return await self._reduce_dimensions(params)
            elif method == 'comprehensive_analysis':
                return await self._comprehensive_analysis(params)
            elif method == 'status':
                return {
                    'status': 'running',
                    'server': 'ml_analysis',
                    'port': self.port,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'error': f'Unknown method: {method}'}
        except Exception as e:
            self.logger.error(f"Request handling failed: {e}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _perform_clustering(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data_dict = params.get('data', {})
            algorithm = params.get('algorithm', 'kmeans')
            n_clusters = params.get('n_clusters', 'auto')
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            results = self.analysis_core.clustering_analyzer.perform_clustering(
                data, algorithm=algorithm, n_clusters=n_clusters
            )
            
            return {
                'status': 'success',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _detect_anomalies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data_dict = params.get('data', {})
            algorithm = params.get('algorithm', 'isolation_forest')
            contamination = params.get('contamination', 0.1)
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            results = self.analysis_core.anomaly_detector.detect_anomalies(
                data, algorithm=algorithm, contamination=contamination
            )
            
            return {
                'status': 'success',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _reduce_dimensions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data_dict = params.get('data', {})
            algorithm = params.get('algorithm', 'pca')
            n_components = params.get('n_components', 2)
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            results = self.analysis_core.dimensionality_reducer.reduce_dimensions(
                data, algorithm=algorithm, n_components=n_components
            )
            
            return {
                'status': 'success',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _comprehensive_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data_dict = params.get('data', {})
            analysis_types = params.get('analysis_types', ['clustering', 'anomaly_detection'])
            
            import pandas as pd
            data = pd.DataFrame(data_dict)
            
            results = self.analysis_core.comprehensive_analysis(data, analysis_types)
            
            return {
                'status': 'success',
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def start(self):
        self.logger.info(f"Starting CCGL ML Analysis MCP Server on port {self.port}")
        self.running = True
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        self.logger.info("Stopping CCGL ML Analysis MCP Server")
        self.running = False

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CCGL ML Analysis MCP Server")
    parser.add_argument('--port', type=int, default=8002, help='Server port')
    parser.add_argument('--config', default='../config.yml', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    server = CCGLMLMCPServer(args.port, args.config)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))