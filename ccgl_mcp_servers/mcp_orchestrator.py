"""
MCP Orchestrator for CCGL Analytics
Coordinates multiple MCP servers and provides unified interface
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ccgl_analytics.modules.mcp_config_manager import MCPConfigManager
from ccgl_analytics.utils.logger import get_logger


class MCPOrchestrator:
    """
    MCP Orchestrator that coordinates multiple MCP servers
    Provides a unified interface for accessing all MCP functionality
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize MCP orchestrator
        
        Args:
            config_dir: Directory containing MCP configuration files
        """
        self.logger = get_logger("mcp_orchestrator")
        self.config_manager = MCPConfigManager(config_dir)
        
        # Server management
        self.active_servers = {}
        self.server_processes = {}
        
        # Tool and resource registry
        self.available_tools = {}
        self.available_resources = {}
        
        self.logger.info("Initialized MCP Orchestrator")
    
    async def start_all_servers(self):
        """Start all configured MCP servers"""
        self.logger.info("Starting all MCP servers")
        
        server_configs = self.config_manager.get_all_server_configs()
        
        for server_name, config in server_configs.items():
            try:
                await self.start_server(server_name)
            except Exception as e:
                self.logger.error(f"Failed to start server {server_name}: {str(e)}")
    
    async def start_server(self, server_name: str):
        """
        Start a specific MCP server
        
        Args:
            server_name: Name of the server to start
        """
        if server_name in self.active_servers:
            self.logger.warning(f"Server {server_name} is already running")
            return
        
        try:
            config = self.config_manager.get_server_config(server_name)
            
            # For demonstration, we'll create mock server instances
            # In a real implementation, this would start actual MCP server processes
            if server_name == "preprocessing":
                from ccgl_mcp_servers.ccgl_preprocessing_mcp_server import CCGLPreprocessingMCPServer
                server_instance = CCGLPreprocessingMCPServer()
                self.active_servers[server_name] = server_instance
                
                # Register tools and resources
                tools = await server_instance.get_tools()
                resources = await server_instance.get_resources()
                
                for tool in tools:
                    self.available_tools[tool["name"]] = {
                        "server": server_name,
                        "definition": tool
                    }
                
                for resource in resources:
                    self.available_resources[resource["uri"]] = {
                        "server": server_name,
                        "definition": resource
                    }
                
                self.logger.info(f"Started {server_name} server with {len(tools)} tools and {len(resources)} resources")
            
            else:
                # Mock implementation for other servers
                self.active_servers[server_name] = {
                    "status": "running",
                    "config": config
                }
                self.logger.info(f"Started mock server: {server_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to start server {server_name}: {str(e)}")
            raise
    
    async def stop_server(self, server_name: str):
        """
        Stop a specific MCP server
        
        Args:
            server_name: Name of the server to stop
        """
        if server_name not in self.active_servers:
            self.logger.warning(f"Server {server_name} is not running")
            return
        
        try:
            # Remove tools and resources registered by this server
            tools_to_remove = [name for name, info in self.available_tools.items() 
                             if info["server"] == server_name]
            resources_to_remove = [uri for uri, info in self.available_resources.items() 
                                 if info["server"] == server_name]
            
            for tool_name in tools_to_remove:
                del self.available_tools[tool_name]
            
            for resource_uri in resources_to_remove:
                del self.available_resources[resource_uri]
            
            # Stop the server
            del self.active_servers[server_name]
            
            self.logger.info(f"Stopped server: {server_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to stop server {server_name}: {str(e)}")
            raise
    
    async def stop_all_servers(self):
        """Stop all running MCP servers"""
        self.logger.info("Stopping all MCP servers")
        
        server_names = list(self.active_servers.keys())
        for server_name in server_names:
            try:
                await self.stop_server(server_name)
            except Exception as e:
                self.logger.error(f"Failed to stop server {server_name}: {str(e)}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the appropriate MCP server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool {tool_name} not found")
        
        tool_info = self.available_tools[tool_name]
        server_name = tool_info["server"]
        
        if server_name not in self.active_servers:
            raise RuntimeError(f"Server {server_name} is not running")
        
        try:
            server = self.active_servers[server_name]
            
            # Call the tool on the appropriate server
            if hasattr(server, 'call_tool'):
                result = await server.call_tool(tool_name, arguments)
            else:
                # Mock implementation for other servers
                result = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "server": server_name,
                    "status": "mock_success"
                }
            
            self.logger.info(f"Tool {tool_name} executed successfully on {server_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute tool {tool_name}: {str(e)}")
            raise
    
    async def get_resource(self, resource_uri: str) -> Dict[str, Any]:
        """
        Get a resource from the appropriate MCP server
        
        Args:
            resource_uri: URI of the resource to get
            
        Returns:
            Resource content
        """
        if resource_uri not in self.available_resources:
            raise ValueError(f"Resource {resource_uri} not found")
        
        resource_info = self.available_resources[resource_uri]
        server_name = resource_info["server"]
        
        if server_name not in self.active_servers:
            raise RuntimeError(f"Server {server_name} is not running")
        
        try:
            server = self.active_servers[server_name]
            
            # Get the resource from the appropriate server
            if hasattr(server, 'get_resource'):
                result = await server.get_resource(resource_uri)
            else:
                # Mock implementation for other servers
                result = {
                    "uri": resource_uri,
                    "server": server_name,
                    "content": "mock_content",
                    "mimeType": "application/json"
                }
            
            self.logger.info(f"Resource {resource_uri} retrieved successfully from {server_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get resource {resource_uri}: {str(e)}")
            raise
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of all available tools across all servers
        
        Returns:
            List of tool definitions
        """
        tools = []
        for tool_name, tool_info in self.available_tools.items():
            tool_def = tool_info["definition"].copy()
            tool_def["server"] = tool_info["server"]
            tools.append(tool_def)
        return tools
    
    def get_available_resources(self) -> List[Dict[str, Any]]:
        """
        Get list of all available resources across all servers
        
        Returns:
            List of resource definitions
        """
        resources = []
        for resource_uri, resource_info in self.available_resources.items():
            resource_def = resource_info["definition"].copy()
            resource_def["server"] = resource_info["server"]
            resources.append(resource_def)
        return resources
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get status of all MCP servers
        
        Returns:
            Dictionary containing server status information
        """
        return {
            "active_servers": list(self.active_servers.keys()),
            "server_count": len(self.active_servers),
            "tool_count": len(self.available_tools),
            "resource_count": len(self.available_resources),
            "configuration_summary": self.config_manager.get_configuration_summary()
        }
    
    async def run_data_analysis_workflow(self, dataset_id: str = "sample_data") -> Dict[str, Any]:
        """
        Run a complete data analysis workflow using available MCP tools
        
        Args:
            dataset_id: Identifier for the dataset to analyze
            
        Returns:
            Workflow execution results
        """
        self.logger.info(f"Starting data analysis workflow for dataset: {dataset_id}")
        
        workflow_results = {
            "dataset_id": dataset_id,
            "steps_completed": [],
            "results": {}
        }
        
        try:
            # Step 1: Data Quality Assessment
            if "data_quality_assessment" in self.available_tools:
                quality_result = await self.call_tool("data_quality_assessment", {
                    "dataset_id": dataset_id
                })
                workflow_results["steps_completed"].append("data_quality_assessment")
                workflow_results["results"]["quality_assessment"] = quality_result
            
            # Step 2: Handle Missing Values
            if "handle_missing_values" in self.available_tools:
                missing_result = await self.call_tool("handle_missing_values", {
                    "dataset_id": dataset_id,
                    "strategy": "auto"
                })
                workflow_results["steps_completed"].append("handle_missing_values")
                workflow_results["results"]["missing_values"] = missing_result
            
            # Step 3: Detect Outliers
            if "detect_outliers" in self.available_tools:
                outlier_result = await self.call_tool("detect_outliers", {
                    "dataset_id": dataset_id,
                    "method": "iqr"
                })
                workflow_results["steps_completed"].append("detect_outliers")
                workflow_results["results"]["outlier_detection"] = outlier_result
            
            # Step 4: Normalize Data
            if "normalize_data" in self.available_tools:
                normalize_result = await self.call_tool("normalize_data", {
                    "dataset_id": dataset_id,
                    "method": "standard"
                })
                workflow_results["steps_completed"].append("normalize_data")
                workflow_results["results"]["normalization"] = normalize_result
            
            workflow_results["status"] = "success"
            workflow_results["message"] = f"Completed {len(workflow_results['steps_completed'])} workflow steps"
            
            self.logger.info(f"Data analysis workflow completed successfully")
            
        except Exception as e:
            workflow_results["status"] = "error"
            workflow_results["error"] = str(e)
            self.logger.error(f"Data analysis workflow failed: {str(e)}")
        
        return workflow_results


async def main():
    """Main entry point for the MCP orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CCGL MCP Orchestrator')
    parser.add_argument('--config-dir', default='config', help='Configuration directory')
    parser.add_argument('--demo', action='store_true', help='Run demo workflow')
    args = parser.parse_args()
    
    orchestrator = MCPOrchestrator(config_dir=args.config_dir)
    
    try:
        # Start servers
        await orchestrator.start_all_servers()
        
        # Print server status
        status = orchestrator.get_server_status()
        print(f"MCP Orchestrator Status:")
        print(f"  Active servers: {status['active_servers']}")
        print(f"  Available tools: {status['tool_count']}")
        print(f"  Available resources: {status['resource_count']}")
        
        if args.demo:
            # Run demo workflow
            print("\nRunning demo data analysis workflow...")
            result = await orchestrator.run_data_analysis_workflow()
            print(f"Workflow result: {result['status']}")
            print(f"Steps completed: {result['steps_completed']}")
        
        # Keep orchestrator running
        print("\nMCP Orchestrator is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping MCP Orchestrator...")
        await orchestrator.stop_all_servers()
        print("MCP Orchestrator stopped.")
    except Exception as e:
        print(f"MCP Orchestrator error: {str(e)}")
        await orchestrator.stop_all_servers()


if __name__ == "__main__":
    asyncio.run(main())