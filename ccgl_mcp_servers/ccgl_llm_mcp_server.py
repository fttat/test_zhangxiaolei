#!/usr/bin/env python3
"""
CCGL LLM Integration MCP Server
Handles LLM integration requests via MCP protocol
"""

import asyncio
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from ccgl_analytics.modules.llm_config_manager import LLMConfigManager
from ccgl_analytics.utils.logger import get_logger, setup_logging

class CCGLLLMMCPServer:
    """MCP server for LLM integration operations."""
    
    def __init__(self, port: int = 8004, config_file: str = '../config.yml'):
        self.port = port
        self.config_file = config_file
        self.logger = get_logger(__name__)
        
        # Load config for LLM manager
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        except:
            config = {}
        
        self.llm_manager = LLMConfigManager(config)
        self.running = False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        method = request.get('method')
        params = request.get('params', {})
        
        try:
            if method == 'generate_response':
                return await self._generate_response(params)
            elif method == 'analyze_data':
                return await self._analyze_data(params)
            elif method == 'generate_insights':
                return await self._generate_insights(params)
            elif method == 'list_providers':
                return await self._list_providers(params)
            elif method == 'status':
                return {
                    'status': 'running',
                    'server': 'llm_integration',
                    'port': self.port,
                    'available_providers': self.llm_manager.get_available_providers(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'error': f'Unknown method: {method}'}
        except Exception as e:
            self.logger.error(f"Request handling failed: {e}")
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _generate_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = params.get('prompt', '')
            provider = params.get('provider')
            context = params.get('context', {})
            
            if not prompt:
                return {'status': 'error', 'error': 'No prompt provided'}
            
            response = await self.llm_manager.generate_response(
                prompt, provider=provider, context=context
            )
            
            return {
                'status': 'success',
                'response': {
                    'content': response.content,
                    'provider': response.provider,
                    'model': response.model,
                    'response_time': response.response_time,
                    'usage': response.usage
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _analyze_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            data_summary = params.get('data_summary', {})
            analysis_results = params.get('analysis_results', {})
            question = params.get('question', 'What insights can you provide about this data?')
            provider = params.get('provider')
            
            # Create optimized prompt for data analysis
            prompt = self.llm_manager.create_data_analysis_prompt(
                data_summary, analysis_results, question
            )
            
            response = await self.llm_manager.generate_response(
                prompt, provider=provider
            )
            
            return {
                'status': 'success',
                'analysis': {
                    'question': question,
                    'insights': response.content,
                    'provider': response.provider,
                    'response_time': response.response_time
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _generate_insights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            analysis_results = params.get('analysis_results', {})
            business_context = params.get('business_context', '')
            providers = params.get('providers', None)
            
            if not analysis_results:
                return {'status': 'error', 'error': 'No analysis results provided'}
            
            # Generate insights from multiple providers if specified
            if providers:
                insights = {}
                
                for provider in providers:
                    try:
                        prompt = f"Analyze these results and provide business insights: {json.dumps(analysis_results)}"
                        if business_context:
                            prompt = self.llm_manager.create_business_context_prompt(prompt, business_context)
                        
                        response = await self.llm_manager.generate_response(prompt, provider=provider)
                        
                        insights[provider] = {
                            'content': response.content,
                            'response_time': response.response_time
                        }
                    except Exception as e:
                        insights[provider] = {'error': str(e)}
                
                return {
                    'status': 'success',
                    'multi_provider_insights': insights,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                # Single provider insights
                prompt = f"Provide comprehensive business insights based on these analysis results: {json.dumps(analysis_results)}"
                if business_context:
                    prompt = self.llm_manager.create_business_context_prompt(prompt, business_context)
                
                response = await self.llm_manager.generate_response(prompt)
                
                return {
                    'status': 'success',
                    'insights': {
                        'content': response.content,
                        'provider': response.provider,
                        'response_time': response.response_time
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def _list_providers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            available_providers = self.llm_manager.get_available_providers()
            configuration_status = self.llm_manager.get_configuration_status()
            
            return {
                'status': 'success',
                'available_providers': available_providers,
                'configuration_status': configuration_status,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    async def start(self):
        self.logger.info(f"Starting CCGL LLM Integration MCP Server on port {self.port}")
        self.running = True
        
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()
    
    async def stop(self):
        self.logger.info("Stopping CCGL LLM Integration MCP Server")
        self.running = False

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CCGL LLM Integration MCP Server")
    parser.add_argument('--port', type=int, default=8004, help='Server port')
    parser.add_argument('--config', default='../config.yml', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    
    server = CCGLLLMMCPServer(args.port, args.config)
    
    try:
        await server.start()
    except KeyboardInterrupt:
        await server.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))