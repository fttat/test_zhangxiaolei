#!/usr/bin/env python3
"""
CCGL MCP Server Orchestrator
Manages and coordinates multiple MCP servers
"""

import asyncio
import json
import signal
import sys
import subprocess
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ccgl_analytics.utils.logger import get_logger, setup_logging

class MCPOrchestrator:
    """Orchestrates multiple MCP servers."""
    
    def __init__(self, config_file: str = '../config.yml'):
        """Initialize MCP orchestrator.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.logger = get_logger(__name__)
        self.servers = {}
        self.running = False
        
        # Server definitions
        self.server_definitions = {
            'preprocessing': {
                'script': 'ccgl_preprocessing_mcp_server.py',
                'port': 8001,
                'description': 'Data preprocessing MCP server'
            },
            'ml_analysis': {
                'script': 'ccgl_ml_mcp_server.py',
                'port': 8002,
                'description': 'Machine learning analysis MCP server'
            },
            'dashboard': {
                'script': 'ccgl_dashboard_mcp_server.py',
                'port': 8003,
                'description': 'Dashboard MCP server'
            },
            'llm_integration': {
                'script': 'ccgl_llm_mcp_server.py',
                'port': 8004,
                'description': 'LLM integration MCP server'
            }
        }
    
    async def start_all_servers(self) -> bool:
        """Start all MCP servers.
        
        Returns:
            True if all servers started successfully
        """
        self.logger.info("Starting all MCP servers")
        
        success_count = 0
        
        for server_name, server_def in self.server_definitions.items():
            try:
                if await self.start_server(server_name):
                    success_count += 1
                    # Wait a bit between server starts
                    await asyncio.sleep(1)
                else:
                    self.logger.error(f"Failed to start server: {server_name}")
            except Exception as e:
                self.logger.error(f"Error starting server {server_name}: {e}")
        
        total_servers = len(self.server_definitions)
        if success_count == total_servers:
            self.logger.info(f"All {total_servers} MCP servers started successfully")
            self.running = True
            return True
        else:
            self.logger.error(f"Only {success_count}/{total_servers} servers started")
            return False
    
    async def start_server(self, server_name: str) -> bool:
        """Start a single MCP server.
        
        Args:
            server_name: Name of the server to start
            
        Returns:
            True if server started successfully
        """
        if server_name not in self.server_definitions:
            self.logger.error(f"Unknown server: {server_name}")
            return False
        
        server_def = self.server_definitions[server_name]
        script_path = Path(__file__).parent / server_def['script']
        
        if not script_path.exists():
            self.logger.error(f"Server script not found: {script_path}")
            return False
        
        try:
            # Start server process
            process = subprocess.Popen([
                sys.executable, str(script_path),
                '--port', str(server_def['port']),
                '--config', self.config_file
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait a moment for server to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                self.servers[server_name] = {
                    'process': process,
                    'definition': server_def,
                    'status': 'running',
                    'start_time': datetime.now().isoformat()
                }
                
                self.logger.info(f"Started MCP server '{server_name}' on port {server_def['port']}")
                return True
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"Server '{server_name}' failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start server '{server_name}': {e}")
            return False
    
    async def stop_all_servers(self):
        """Stop all running MCP servers."""
        self.logger.info("Stopping all MCP servers")
        
        for server_name in list(self.servers.keys()):
            await self.stop_server(server_name)
        
        self.running = False
        self.logger.info("All MCP servers stopped")
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop a single MCP server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            True if server stopped successfully
        """
        if server_name not in self.servers:
            self.logger.warning(f"Server '{server_name}' not running")
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
                self.logger.warning(f"Force killing server '{server_name}'")
                process.kill()
                process.wait()
            
            del self.servers[server_name]
            self.logger.info(f"Stopped MCP server '{server_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop server '{server_name}': {e}")
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers.
        
        Returns:
            Server status information
        """
        status = {
            'orchestrator_running': self.running,
            'total_servers_defined': len(self.server_definitions),
            'servers_running': len(self.servers),
            'servers': {}
        }
        
        # Status of running servers
        for server_name, server_info in self.servers.items():
            process = server_info['process']
            status['servers'][server_name] = {
                'status': 'running' if process.poll() is None else 'stopped',
                'port': server_info['definition']['port'],
                'pid': process.pid,
                'start_time': server_info['start_time']
            }
        
        # Status of defined but not running servers
        for server_name, server_def in self.server_definitions.items():
            if server_name not in self.servers:
                status['servers'][server_name] = {
                    'status': 'not_running',
                    'port': server_def['port'],
                    'description': server_def['description']
                }
        
        return status
    
    async def monitor_servers(self):
        """Monitor server health and restart if needed."""
        while self.running:
            try:
                # Check each running server
                failed_servers = []
                
                for server_name, server_info in self.servers.items():
                    process = server_info['process']
                    if process.poll() is not None:
                        # Server has stopped
                        failed_servers.append(server_name)
                
                # Restart failed servers
                for server_name in failed_servers:
                    self.logger.warning(f"Server '{server_name}' has failed, attempting restart")
                    await self.stop_server(server_name)
                    await asyncio.sleep(2)
                    await self.start_server(server_name)
                
                # Wait before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Server monitoring error: {e}")
                await asyncio.sleep(30)
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down servers...")
            
            # Create new event loop for cleanup
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_all_servers())
            loop.close()
            
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main function for MCP orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CCGL MCP Server Orchestrator")
    parser.add_argument('--config', default='../config.yml', help='Configuration file path')
    parser.add_argument('--start', action='store_true', help='Start all servers')
    parser.add_argument('--stop', action='store_true', help='Stop all servers')
    parser.add_argument('--status', action='store_true', help='Show server status')
    parser.add_argument('--monitor', action='store_true', help='Start with monitoring')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='DEBUG' if args.verbose else 'INFO')
    logger = get_logger(__name__)
    
    orchestrator = MCPOrchestrator(args.config)
    
    try:
        if args.start:
            logger.info("Starting MCP server orchestrator")
            orchestrator.setup_signal_handlers()
            
            if await orchestrator.start_all_servers():
                logger.info("All servers started successfully")
                
                if args.monitor:
                    logger.info("Starting monitoring mode")
                    await orchestrator.monitor_servers()
                else:
                    # Keep running until interrupted
                    try:
                        while orchestrator.running:
                            await asyncio.sleep(1)
                    except KeyboardInterrupt:
                        pass
            
            await orchestrator.stop_all_servers()
            
        elif args.stop:
            logger.info("Stopping all servers")
            await orchestrator.stop_all_servers()
            
        elif args.status:
            status = orchestrator.get_server_status()
            print(json.dumps(status, indent=2))
            
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Orchestrator error: {e}", exc_info=True)
        await orchestrator.stop_all_servers()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))