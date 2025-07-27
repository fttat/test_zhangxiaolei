#!/usr/bin/env python3
"""
Initialize MCP Configuration
Setup script for MCP server configurations
"""

import json
import yaml
import os
from pathlib import Path

def create_mcp_configs():
    """Create MCP configuration files."""
    
    # Create config directory
    config_dir = Path('config')
    config_dir.mkdir(exist_ok=True)
    
    # Development configuration
    dev_config = {
        "environment": "development",
        "mcp": {
            "server": {
                "host": "localhost",
                "enable_ssl": False,
                "log_level": "DEBUG"
            },
            "servers": {
                "preprocessing": {"port": 8001, "enabled": True},
                "ml_analysis": {"port": 8002, "enabled": True},
                "dashboard": {"port": 8003, "enabled": True},
                "llm_integration": {"port": 8004, "enabled": True}
            }
        }
    }
    
    with open(config_dir / 'mcp_config_development.json', 'w') as f:
        json.dump(dev_config, f, indent=2)
    
    # Testing configuration
    test_config = dev_config.copy()
    test_config["environment"] = "testing"
    test_config["mcp"]["servers"] = {
        "preprocessing": {"port": 9001, "enabled": True},
        "ml_analysis": {"port": 9002, "enabled": False},
        "dashboard": {"port": 9003, "enabled": False},
        "llm_integration": {"port": 9004, "enabled": False}
    }
    
    with open(config_dir / 'mcp_config_testing.json', 'w') as f:
        json.dump(test_config, f, indent=2)
    
    # Production configuration
    prod_config = {
        "environment": "production",
        "mcp": {
            "server": {
                "host": "0.0.0.0",
                "enable_ssl": True,
                "log_level": "INFO"
            },
            "servers": {
                "preprocessing": {"port": 8001, "enabled": True},
                "ml_analysis": {"port": 8002, "enabled": True},
                "dashboard": {"port": 8003, "enabled": True},
                "llm_integration": {"port": 8004, "enabled": True}
            }
        }
    }
    
    with open(config_dir / 'mcp_config_production.json', 'w') as f:
        json.dump(prod_config, f, indent=2)
    
    print("✅ MCP configuration files created successfully!")

if __name__ == "__main__":
    create_mcp_configs()