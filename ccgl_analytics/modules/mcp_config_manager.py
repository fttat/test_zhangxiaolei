"""
MCP Configuration Manager for CCGL Analytics
Handles MCP server and client configuration management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio

from ..utils.logger import get_logger


class MCPConfigManager:
    """
    MCP Configuration Manager
    Handles loading, validation, and management of MCP configurations
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize MCP configuration manager
        
        Args:
            config_dir: Directory containing MCP configuration files
        """
        self.config_dir = Path(config_dir)
        self.logger = get_logger("mcp_config")
        
        # Configuration storage
        self.mcp_config = {}
        self.server_configs = {}
        self.llm_config = {}
        
        # Load configurations
        self._load_configurations()
    
    def _load_configurations(self):
        """Load all MCP configuration files"""
        try:
            # Load main MCP configuration
            mcp_config_file = self.config_dir / "mcp_config.json"
            if mcp_config_file.exists():
                with open(mcp_config_file, 'r') as f:
                    self.mcp_config = json.load(f)
                self.logger.info("Loaded main MCP configuration")
            
            # Load server configurations
            server_config_file = self.config_dir / "mcp_servers.json"
            if server_config_file.exists():
                with open(server_config_file, 'r') as f:
                    self.server_configs = json.load(f)
                self.logger.info("Loaded MCP server configurations")
            
            # Load LLM configuration
            llm_config_file = self.config_dir / "llm_config.json"
            if llm_config_file.exists():
                with open(llm_config_file, 'r') as f:
                    self.llm_config = json.load(f)
                self.logger.info("Loaded LLM configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to load MCP configurations: {str(e)}")
            raise
    
    def get_server_config(self, server_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific MCP server
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            Dictionary containing server configuration
        """
        if server_name not in self.mcp_config.get('servers', {}):
            raise ValueError(f"Server '{server_name}' not found in configuration")
        
        base_config = self.mcp_config['servers'][server_name]
        server_details = self.server_configs.get(f"{server_name}_server", {})
        
        # Merge configurations
        config = {**base_config, **server_details}
        
        # Add global settings
        config['settings'] = self.mcp_config.get('settings', {})
        
        return config
    
    def get_all_server_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all MCP servers
        
        Returns:
            Dictionary mapping server names to their configurations
        """
        configs = {}
        
        for server_name in self.mcp_config.get('servers', {}):
            configs[server_name] = self.get_server_config(server_name)
        
        return configs
    
    def get_llm_config(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get LLM configuration
        
        Args:
            provider: Specific LLM provider (optional)
            
        Returns:
            Dictionary containing LLM configuration
        """
        if provider:
            if provider not in self.llm_config.get('llm_providers', {}):
                raise ValueError(f"LLM provider '{provider}' not found in configuration")
            return self.llm_config['llm_providers'][provider]
        
        return self.llm_config
    
    def get_server_endpoints(self) -> Dict[str, str]:
        """
        Get endpoints for all MCP servers
        
        Returns:
            Dictionary mapping server names to their endpoints
        """
        endpoints = {}
        
        for server_name, config in self.server_configs.items():
            if 'port' in config:
                port = config['port']
                endpoints[server_name] = f"http://localhost:{port}"
        
        return endpoints
    
    def validate_configuration(self) -> List[str]:
        """
        Validate MCP configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required main configuration fields
        required_fields = ['servers', 'clients', 'settings']
        for field in required_fields:
            if field not in self.mcp_config:
                errors.append(f"Missing required field '{field}' in main MCP configuration")
        
        # Check server configurations
        for server_name, server_config in self.mcp_config.get('servers', {}).items():
            required_server_fields = ['name', 'description', 'command', 'args']
            for field in required_server_fields:
                if field not in server_config:
                    errors.append(f"Missing required field '{field}' for server '{server_name}'")
        
        # Check LLM configuration
        if 'llm_providers' not in self.llm_config:
            errors.append("Missing 'llm_providers' in LLM configuration")
        
        # Check environment variables for LLM providers
        for provider_name, provider_config in self.llm_config.get('llm_providers', {}).items():
            api_key_env = provider_config.get('api_key_env')
            if api_key_env and not os.getenv(api_key_env):
                self.logger.warning(f"Environment variable '{api_key_env}' not set for {provider_name}")
        
        return errors
    
    def create_server_startup_command(self, server_name: str) -> List[str]:
        """
        Create startup command for a specific MCP server
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            List containing command and arguments
        """
        config = self.get_server_config(server_name)
        
        command = [config['command']] + config['args']
        
        # Add environment variables
        env_vars = config.get('env', {})
        for key, value in env_vars.items():
            # Expand environment variables
            if value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]
                value = os.getenv(env_var, '')
            
            os.environ[key] = value
        
        return command
    
    def get_available_tools(self, server_name: str) -> List[str]:
        """
        Get available tools for a specific MCP server
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            List of available tool names
        """
        server_config = self.server_configs.get(f"{server_name}_server", {})
        return server_config.get('tools', [])
    
    def get_available_resources(self, server_name: str) -> List[str]:
        """
        Get available resources for a specific MCP server
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            List of available resource names
        """
        server_config = self.server_configs.get(f"{server_name}_server", {})
        return server_config.get('resources', [])
    
    def update_server_config(self, server_name: str, updates: Dict[str, Any]):
        """
        Update configuration for a specific server
        
        Args:
            server_name: Name of the server to update
            updates: Dictionary of updates to apply
        """
        if server_name in self.mcp_config.get('servers', {}):
            self.mcp_config['servers'][server_name].update(updates)
            self.logger.info(f"Updated configuration for server '{server_name}'")
        else:
            raise ValueError(f"Server '{server_name}' not found in configuration")
    
    def save_configuration(self, config_type: str = "mcp"):
        """
        Save configuration to file
        
        Args:
            config_type: Type of configuration to save ('mcp', 'servers', 'llm')
        """
        try:
            if config_type == "mcp":
                config_file = self.config_dir / "mcp_config.json"
                with open(config_file, 'w') as f:
                    json.dump(self.mcp_config, f, indent=2)
                    
            elif config_type == "servers":
                config_file = self.config_dir / "mcp_servers.json"
                with open(config_file, 'w') as f:
                    json.dump(self.server_configs, f, indent=2)
                    
            elif config_type == "llm":
                config_file = self.config_dir / "llm_config.json"
                with open(config_file, 'w') as f:
                    json.dump(self.llm_config, f, indent=2)
            
            self.logger.info(f"Saved {config_type} configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to save {config_type} configuration: {str(e)}")
            raise
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of all configurations
        
        Returns:
            Dictionary containing configuration summary
        """
        return {
            'servers': list(self.mcp_config.get('servers', {}).keys()),
            'server_count': len(self.mcp_config.get('servers', {})),
            'llm_providers': list(self.llm_config.get('llm_providers', {}).keys()),
            'validation_errors': self.validate_configuration(),
            'endpoints': self.get_server_endpoints()
        }


def create_default_mcp_config(config_dir: str = "config") -> MCPConfigManager:
    """
    Create default MCP configuration files and return manager
    
    Args:
        config_dir: Directory to create configuration files in
        
    Returns:
        MCPConfigManager instance with default configuration
    """
    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger("mcp_config_init")
    
    # Create default configurations if they don't exist
    default_configs = {
        "mcp_config.json": {
            "name": "ccgl-mcp-system",
            "version": "1.0.0",
            "description": "CCGL Analytics MCP Server Configuration",
            "servers": {
                "preprocessing": {
                    "name": "ccgl-preprocessing-server",
                    "description": "Data preprocessing and quality management MCP server",
                    "command": "python",
                    "args": ["ccgl_mcp_servers/ccgl_preprocessing_mcp_server.py"],
                    "env": {"PYTHONPATH": "."}
                }
            },
            "clients": {
                "orchestrator": {
                    "name": "ccgl-orchestrator",
                    "description": "Main orchestrator client for coordinating MCP servers"
                }
            },
            "settings": {
                "log_level": "INFO",
                "max_connections": 10,
                "timeout": 30,
                "retry_attempts": 3
            }
        },
        "mcp_servers.json": {
            "preprocessing_server": {
                "port": 3001,
                "tools": ["data_quality_assessment", "handle_missing_values"],
                "resources": ["quality_reports", "preprocessing_history"]
            }
        },
        "llm_config.json": {
            "llm_providers": {
                "openai": {
                    "name": "OpenAI",
                    "models": {
                        "gpt-4": {"max_tokens": 4096, "temperature": 0.7}
                    },
                    "api_key_env": "OPENAI_API_KEY"
                }
            },
            "default_provider": "openai",
            "default_model": "gpt-4"
        }
    }
    
    for filename, config in default_configs.items():
        config_file = config_path / filename
        if not config_file.exists():
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Created default configuration: {filename}")
    
    return MCPConfigManager(config_dir)