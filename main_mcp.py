#!/usr/bin/env python3
"""
CCGL Analytics System - MCP Architecture Main Program
Distributed analysis mode using Model Context Protocol (MCP)
"""

import argparse
import asyncio
import sys
import os
import yaml
import json
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import subprocess
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ccgl_analytics.utils.logger import get_logger, setup_logging

class MCPServerManager:
    """Manager for MCP server cluster."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MCP server manager.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.servers = {}
        self.running = False
        
    async def start_server(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """Start a single MCP server.
        
        Args:
            server_name: Name of the server
            server_config: Server configuration
            
        Returns:
            True if started successfully
        """
        try:
            port = server_config.get('port', 8000)
            script_path = f"ccgl_mcp_servers/ccgl_{server_name}_mcp_server.py"
            
            if not Path(script_path).exists():
                self.logger.error(f"MCP server script not found: {script_path}")
                return False
            
            # Start server process
            process = subprocess.Popen([
                sys.executable, script_path,
                '--port', str(port),
                '--config', 'config.yml'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.servers[server_name] = {
                'process': process,
                'port': port,
                'config': server_config,
                'status': 'starting'
            }
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                self.servers[server_name]['status'] = 'running'
                self.logger.info(f"MCP server '{server_name}' started on port {port}")
                return True
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"MCP server '{server_name}' failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start MCP server '{server_name}': {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop a single MCP server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            True if stopped successfully
        """
        if server_name not in self.servers:
            return True
        
        try:
            server_info = self.servers[server_name]
            process = server_info['process']
            
            # Gracefully terminate
            process.terminate()
            
            # Wait for termination
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Force killing MCP server '{server_name}'")
                process.kill()
                process.wait()
            
            server_info['status'] = 'stopped'
            self.logger.info(f"MCP server '{server_name}' stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop MCP server '{server_name}': {e}")
            return False
    
    async def start_all_servers(self) -> bool:
        """Start all configured MCP servers.
        
        Returns:
            True if all servers started successfully
        """
        mcp_config = self.config.get('mcp', {})
        servers_config = mcp_config.get('servers', {})
        
        if not servers_config:
            self.logger.warning("No MCP servers configured")
            return False
        
        self.logger.info("Starting MCP server cluster...")
        
        start_tasks = []
        for server_name, server_config in servers_config.items():
            if server_config.get('enabled', True):
                task = self.start_server(server_name, server_config)
                start_tasks.append(task)
        
        results = await asyncio.gather(*start_tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        total_count = len(start_tasks)
        
        if success_count == total_count:
            self.logger.info(f"All {total_count} MCP servers started successfully")
            self.running = True
            return True
        else:
            self.logger.error(f"Only {success_count}/{total_count} MCP servers started successfully")
            return False
    
    async def stop_all_servers(self) -> bool:
        """Stop all running MCP servers.
        
        Returns:
            True if all servers stopped successfully
        """
        if not self.servers:
            return True
        
        self.logger.info("Stopping MCP server cluster...")
        
        stop_tasks = []
        for server_name in self.servers.keys():
            task = self.stop_server(server_name)
            stop_tasks.append(task)
        
        results = await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        total_count = len(stop_tasks)
        
        self.running = False
        
        if success_count == total_count:
            self.logger.info(f"All {total_count} MCP servers stopped successfully")
            return True
        else:
            self.logger.warning(f"Only {success_count}/{total_count} MCP servers stopped successfully")
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers.
        
        Returns:
            Server status information
        """
        status = {
            'cluster_running': self.running,
            'total_servers': len(self.servers),
            'servers': {}
        }
        
        for server_name, server_info in self.servers.items():
            process = server_info['process']
            if process.poll() is None:
                status['servers'][server_name] = {
                    'status': 'running',
                    'port': server_info['port'],
                    'pid': process.pid
                }
            else:
                status['servers'][server_name] = {
                    'status': 'stopped',
                    'port': server_info['port'],
                    'exit_code': process.poll()
                }
        
        return status

class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MCP client.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.connections = {}
    
    async def connect_to_servers(self) -> bool:
        """Connect to all available MCP servers.
        
        Returns:
            True if connected successfully
        """
        mcp_config = self.config.get('mcp', {})
        servers_config = mcp_config.get('servers', {})
        
        self.logger.info("Connecting to MCP servers...")
        
        connection_tasks = []
        for server_name, server_config in servers_config.items():
            if server_config.get('enabled', True):
                task = self._connect_to_server(server_name, server_config)
                connection_tasks.append(task)
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        total_count = len(connection_tasks)
        
        if success_count > 0:
            self.logger.info(f"Connected to {success_count}/{total_count} MCP servers")
            return True
        else:
            self.logger.error("Failed to connect to any MCP servers")
            return False
    
    async def _connect_to_server(self, server_name: str, server_config: Dict[str, Any]) -> bool:
        """Connect to a single MCP server.
        
        Args:
            server_name: Name of the server
            server_config: Server configuration
            
        Returns:
            True if connected successfully
        """
        try:
            port = server_config.get('port', 8000)
            host = self.config.get('mcp', {}).get('server', {}).get('host', 'localhost')
            
            # Simulate connection (in real implementation, this would use actual MCP protocol)
            await asyncio.sleep(0.1)  # Simulate connection delay
            
            self.connections[server_name] = {
                'host': host,
                'port': port,
                'status': 'connected'
            }
            
            self.logger.info(f"Connected to MCP server '{server_name}' at {host}:{port}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MCP server '{server_name}': {e}")
            return False
    
    async def request_analysis(self, server_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send analysis request to specific MCP server.
        
        Args:
            server_name: Target server name
            request: Analysis request
            
        Returns:
            Analysis response
        """
        if server_name not in self.connections:
            raise ValueError(f"Not connected to server '{server_name}'")
        
        self.logger.info(f"Sending analysis request to '{server_name}'")
        
        # Simulate analysis request (in real implementation, this would use actual MCP protocol)
        await asyncio.sleep(1)  # Simulate processing time
        
        response = {
            'server': server_name,
            'request_id': f"req_{int(time.time())}",
            'status': 'completed',
            'results': {
                'message': f"Analysis completed by {server_name}",
                'request': request
            }
        }
        
        self.logger.info(f"Received response from '{server_name}'")
        return response
    
    async def orchestrate_distributed_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate distributed analysis across multiple MCP servers.
        
        Args:
            analysis_request: Comprehensive analysis request
            
        Returns:
            Combined analysis results
        """
        self.logger.info("Starting distributed analysis orchestration")
        
        # Define analysis distribution strategy
        analysis_plan = {
            'preprocessing': {
                'server': 'preprocessing',
                'tasks': ['data_quality', 'cleaning', 'transformation']
            },
            'ml_analysis': {
                'server': 'ml_analysis',
                'tasks': ['clustering', 'anomaly_detection', 'dimensionality_reduction']
            },
            'dashboard': {
                'server': 'dashboard',
                'tasks': ['visualization', 'report_generation']
            },
            'llm_integration': {
                'server': 'llm_integration',
                'tasks': ['natural_language_insights', 'report_enhancement']
            }
        }
        
        # Execute distributed analysis
        analysis_tasks = []
        for stage_name, stage_config in analysis_plan.items():
            server_name = stage_config['server']
            if server_name in self.connections:
                request = {
                    'stage': stage_name,
                    'tasks': stage_config['tasks'],
                    'data': analysis_request.get('data'),
                    'parameters': analysis_request.get('parameters', {})
                }
                task = self.request_analysis(server_name, request)
                analysis_tasks.append((stage_name, task))
        
        # Wait for all analysis stages to complete
        results = {}
        for stage_name, task in analysis_tasks:
            try:
                result = await task
                results[stage_name] = result
            except Exception as e:
                self.logger.error(f"Analysis stage '{stage_name}' failed: {e}")
                results[stage_name] = {'error': str(e)}
        
        # Combine results
        combined_results = {
            'orchestration_id': f"orch_{int(time.time())}",
            'total_stages': len(analysis_plan),
            'completed_stages': len([r for r in results.values() if 'error' not in r]),
            'results': results,
            'summary': self._generate_analysis_summary(results)
        }
        
        self.logger.info("Distributed analysis orchestration completed")
        return combined_results
    
    def _generate_analysis_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of distributed analysis results.
        
        Args:
            results: Analysis results from all stages
            
        Returns:
            Analysis summary
        """
        summary = {
            'successful_stages': [],
            'failed_stages': [],
            'key_insights': [],
            'recommendations': []
        }
        
        for stage_name, stage_result in results.items():
            if 'error' in stage_result:
                summary['failed_stages'].append(stage_name)
            else:
                summary['successful_stages'].append(stage_name)
        
        # Generate insights based on successful stages
        if 'preprocessing' in summary['successful_stages']:
            summary['key_insights'].append("Data preprocessing completed successfully")
        
        if 'ml_analysis' in summary['successful_stages']:
            summary['key_insights'].append("Machine learning analysis provided valuable patterns")
        
        if 'llm_integration' in summary['successful_stages']:
            summary['key_insights'].append("AI-enhanced insights generated")
        
        return summary

async def monitor_servers(server_manager: MCPServerManager, interval: int = 30):
    """Monitor MCP servers and restart if necessary.
    
    Args:
        server_manager: Server manager instance
        interval: Monitoring interval in seconds
    """
    logger = get_logger(__name__)
    
    while server_manager.running:
        try:
            status = server_manager.get_server_status()
            
            # Check for failed servers
            failed_servers = []
            for server_name, server_info in status['servers'].items():
                if server_info['status'] != 'running':
                    failed_servers.append(server_name)
            
            if failed_servers:
                logger.warning(f"Detected failed servers: {failed_servers}")
                # In a production system, you might implement auto-restart here
            
            await asyncio.sleep(interval)
            
        except Exception as e:
            logger.error(f"Server monitoring error: {e}")
            await asyncio.sleep(interval)

def setup_signal_handlers(server_manager: MCPServerManager):
    """Setup signal handlers for graceful shutdown.
    
    Args:
        server_manager: Server manager instance
    """
    def signal_handler(signum, frame):
        logger = get_logger(__name__)
        logger.info(f"Received signal {signum}, shutting down...")
        
        # Create new event loop for cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server_manager.stop_all_servers())
        loop.close()
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main_async(args, config):
    """Main async function for MCP mode."""
    logger = get_logger(__name__)
    
    # Initialize MCP server manager
    server_manager = MCPServerManager(config)
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(server_manager)
    
    try:
        if args.start_mcp_servers:
            # Start MCP servers
            success = await server_manager.start_all_servers()
            if not success:
                logger.error("Failed to start MCP server cluster")
                return 1
            
            # Start monitoring in background
            monitor_task = asyncio.create_task(monitor_servers(server_manager))
            
            if args.interactive:
                # Interactive mode
                logger.info("Entering interactive mode. Type 'help' for commands.")
                client = MCPClient(config)
                await client.connect_to_servers()
                
                while True:
                    try:
                        command = input("\nCCGL-MCP> ").strip()
                        
                        if command in ['exit', 'quit']:
                            break
                        elif command == 'status':
                            status = server_manager.get_server_status()
                            print(json.dumps(status, indent=2))
                        elif command == 'help':
                            print("Available commands:")
                            print("  status - Show server status")
                            print("  analyze - Run distributed analysis")
                            print("  exit - Exit interactive mode")
                        elif command == 'analyze':
                            print("Starting distributed analysis...")
                            request = {
                                'data': 'sample_data',
                                'parameters': {'type': 'comprehensive'}
                            }
                            results = await client.orchestrate_distributed_analysis(request)
                            print(json.dumps(results, indent=2))
                        else:
                            print(f"Unknown command: {command}")
                            
                    except KeyboardInterrupt:
                        break
                    except EOFError:
                        break
            else:
                # Non-interactive mode - keep servers running
                logger.info("MCP servers running. Press Ctrl+C to stop.")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    pass
            
            # Cleanup
            monitor_task.cancel()
            await server_manager.stop_all_servers()
            
        elif args.run_analysis:
            # Run distributed analysis
            client = MCPClient(config)
            await client.connect_to_servers()
            
            analysis_request = {
                'data': args.data_source,
                'parameters': {
                    'analysis_type': args.analysis_type,
                    'output_format': args.output_format
                }
            }
            
            results = await client.orchestrate_distributed_analysis(analysis_request)
            
            if args.output_file:
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to: {args.output_file}")
            else:
                print(json.dumps(results, indent=2))
        
        return 0
        
    except Exception as e:
        logger.error(f"MCP operation failed: {e}", exc_info=True)
        await server_manager.stop_all_servers()
        return 1

def main():
    """Main entry point for MCP mode."""
    parser = argparse.ArgumentParser(
        description="CCGL Analytics System - MCP Architecture Mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --start-mcp-servers --interactive
  %(prog)s --start-mcp-servers --daemon
  %(prog)s --run-analysis --data-source "sales_data" --output results.json
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        default='config.yml',
        help='Configuration file path (default: config.yml)'
    )
    
    parser.add_argument(
        '--start-mcp-servers',
        action='store_true',
        help='Start MCP server cluster'
    )
    
    parser.add_argument(
        '--run-analysis',
        action='store_true',
        help='Run distributed analysis'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--data-source',
        help='Data source for analysis'
    )
    
    parser.add_argument(
        '--analysis-type',
        default='comprehensive',
        help='Type of analysis to perform'
    )
    
    parser.add_argument(
        '--output-file', '-o',
        help='Output file for results'
    )
    
    parser.add_argument(
        '--output-format',
        default='json',
        choices=['json', 'yaml', 'csv'],
        help='Output format'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)
    
    # Setup logging
    logging_config = config.get('logging', {})
    setup_logging(
        level='DEBUG' if args.verbose else logging_config.get('level', 'INFO'),
        format_type=logging_config.get('format', 'text'),
        log_file=logging_config.get('file'),
    )
    
    logger = get_logger(__name__)
    logger.info("Starting CCGL Analytics System - MCP Mode")
    
    # Run main async function
    try:
        return asyncio.run(main_async(args, config))
    except KeyboardInterrupt:
        logger.info("MCP system interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())