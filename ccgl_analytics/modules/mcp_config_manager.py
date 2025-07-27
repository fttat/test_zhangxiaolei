"""
CCGL Analytics - MCP Configuration Manager
Manage MCP server configurations and connections
"""

import json
import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from ..utils.logger import get_logger, LoggerMixin

class MCPConfigManager(LoggerMixin):
    """Manager for MCP server configurations."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize MCP config manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or 'config.yml'
        self.config = self._load_config()
        self.mcp_config = self.config.get('mcp', {})
        self.servers_config = self.mcp_config.get('servers', {})
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.endswith('.json'):
                        return json.load(f)
                    else:
                        return yaml.safe_load(f) or {}
            else:
                self.logger.warning(f"Configuration file not found: {self.config_file}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def get_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server configuration or None
        """
        return self.servers_config.get(server_name)
    
    def get_all_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get all server configurations.
        
        Returns:
            All server configurations
        """
        return self.servers_config.copy()
    
    def get_enabled_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get only enabled server configurations.
        
        Returns:
            Enabled server configurations
        """
        return {
            name: config for name, config in self.servers_config.items()
            if config.get('enabled', True)
        }
    
    def add_server_config(self, server_name: str, config: Dict[str, Any]) -> bool:
        """Add or update server configuration.
        
        Args:
            server_name: Name of the server
            config: Server configuration
            
        Returns:
            True if successful
        """
        try:
            self.servers_config[server_name] = config
            self._save_config()
            self.logger.info(f"Added/updated server configuration: {server_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add server config: {e}")
            return False
    
    def remove_server_config(self, server_name: str) -> bool:
        """Remove server configuration.
        
        Args:
            server_name: Name of the server
            
        Returns:
            True if successful
        """
        try:
            if server_name in self.servers_config:
                del self.servers_config[server_name]
                self._save_config()
                self.logger.info(f"Removed server configuration: {server_name}")
                return True
            else:
                self.logger.warning(f"Server configuration not found: {server_name}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to remove server config: {e}")
            return False
    
    def enable_server(self, server_name: str) -> bool:
        """Enable a server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            True if successful
        """
        if server_name in self.servers_config:
            self.servers_config[server_name]['enabled'] = True
            self._save_config()
            return True
        return False
    
    def disable_server(self, server_name: str) -> bool:
        """Disable a server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            True if successful
        """
        if server_name in self.servers_config:
            self.servers_config[server_name]['enabled'] = False
            self._save_config()
            return True
        return False
    
    def get_server_endpoints(self) -> Dict[str, str]:
        """Get all server endpoints.
        
        Returns:
            Dictionary of server endpoints
        """
        endpoints = {}
        host = self.mcp_config.get('server', {}).get('host', 'localhost')
        
        for server_name, config in self.get_enabled_servers().items():
            port = config.get('port', 8000)
            protocol = 'https' if self.mcp_config.get('server', {}).get('enable_ssl', False) else 'http'
            endpoints[server_name] = f"{protocol}://{host}:{port}"
        
        return endpoints
    
    def validate_server_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate server configuration.
        
        Args:
            config: Server configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Required fields
        if 'port' not in config:
            errors.append("Missing required field: port")
        
        # Port validation
        port = config.get('port')
        if port and (not isinstance(port, int) or port < 1 or port > 65535):
            errors.append("Port must be an integer between 1 and 65535")
        
        # Check for port conflicts
        used_ports = set()
        for name, server_config in self.servers_config.items():
            if server_config.get('enabled', True):
                used_ports.add(server_config.get('port'))
        
        if port in used_ports:
            errors.append(f"Port {port} is already in use by another server")
        
        return errors
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default MCP configuration.
        
        Returns:
            Default configuration
        """
        return {
            'mcp': {
                'server': {
                    'host': 'localhost',
                    'port': 8000,
                    'enable_ssl': False,
                    'ssl_cert_path': '',
                    'ssl_key_path': ''
                },
                'client': {
                    'timeout': 30,
                    'max_connections': 100,
                    'heartbeat_interval': 30,
                    'retry_attempts': 3
                },
                'servers': {
                    'preprocessing': {
                        'port': 8001,
                        'enabled': True,
                        'description': 'Data preprocessing MCP server'
                    },
                    'ml_analysis': {
                        'port': 8002,
                        'enabled': True,
                        'description': 'Machine learning analysis MCP server'
                    },
                    'dashboard': {
                        'port': 8003,
                        'enabled': True,
                        'description': 'Dashboard MCP server'
                    },
                    'llm_integration': {
                        'port': 8004,
                        'enabled': True,
                        'description': 'LLM integration MCP server'
                    }
                }
            }
        }
    
    def export_config(self, output_file: str) -> bool:
        """Export configuration to file.
        
        Args:
            output_file: Output file path
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration exported to: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, input_file: str) -> bool:
        """Import configuration from file.
        
        Args:
            input_file: Input file path
            
        Returns:
            True if successful
        """
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.endswith('.json'):
                    imported_config = json.load(f)
                else:
                    imported_config = yaml.safe_load(f)
            
            # Merge configurations
            if 'mcp' in imported_config:
                self.config['mcp'] = imported_config['mcp']
                self.mcp_config = self.config['mcp']
                self.servers_config = self.mcp_config.get('servers', {})
                
                self._save_config()
                self.logger.info(f"Configuration imported from: {input_file}")
                return True
            else:
                self.logger.error("Invalid configuration file: missing 'mcp' section")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    def get_server_status_summary(self) -> Dict[str, Any]:
        """Get summary of server configurations.
        
        Returns:
            Server status summary
        """
        all_servers = self.get_all_servers()
        enabled_servers = self.get_enabled_servers()
        
        return {
            'total_servers': len(all_servers),
            'enabled_servers': len(enabled_servers),
            'disabled_servers': len(all_servers) - len(enabled_servers),
            'server_list': list(all_servers.keys()),
            'enabled_list': list(enabled_servers.keys()),
            'endpoints': self.get_server_endpoints(),
            'last_updated': datetime.now().isoformat()
        }
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def generate_server_script(self, server_name: str, template_type: str = 'basic') -> str:
        """Generate server script template.
        
        Args:
            server_name: Name of the server
            template_type: Type of template ('basic', 'advanced')
            
        Returns:
            Generated script content
        """
        config = self.get_server_config(server_name)
        if not config:
            return "# Error: Server configuration not found"
        
        port = config.get('port', 8000)
        description = config.get('description', f'{server_name} MCP server')
        
        if template_type == 'basic':
            return f'''#!/usr/bin/env python3
"""
{description}
Generated automatically by CCGL Analytics
"""

import asyncio
import json
from typing import Dict, Any
from datetime import datetime

class {server_name.title()}MCPServer:
    """MCP server for {server_name}."""
    
    def __init__(self, port: int = {port}):
        """Initialize server."""
        self.port = port
        self.running = False
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        method = request.get('method')
        params = request.get('params', {{}})
        
        if method == 'status':
            return {{
                'status': 'running',
                'server': '{server_name}',
                'timestamp': datetime.now().isoformat()
            }}
        else:
            return {{
                'error': f'Unknown method: {{method}}'
            }}
    
    async def start(self):
        """Start the server."""
        self.running = True
        print(f"{description} started on port {port}")
        
        # Server implementation would go here
        while self.running:
            await asyncio.sleep(1)
    
    def stop(self):
        """Stop the server."""
        self.running = False

if __name__ == "__main__":
    server = {server_name.title()}MCPServer()
    asyncio.run(server.start())
'''
        else:
            return "# Advanced template not implemented yet"