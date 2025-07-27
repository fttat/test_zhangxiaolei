#!/usr/bin/env python3
"""
CCGL Analytics - MCP Architecture Main Entry Point
Enterprise data analysis with Model Context Protocol architecture
"""

import sys
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from ccgl_mcp_servers.mcp_orchestrator import MCPOrchestrator
from ccgl_analytics.modules.mcp_config_manager import MCPConfigManager, create_default_mcp_config
from ccgl_analytics.utils.logger import get_logger, setup_logging


async def main():
    """Main execution function for MCP architecture"""
    parser = argparse.ArgumentParser(description='CCGL Analytics - MCP Architecture')
    parser.add_argument('-c', '--config', default='config.yml', help='Configuration file path')
    parser.add_argument('--config-dir', default='config', help='MCP configuration directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--full-mcp', action='store_true', help='Use full MCP architecture')
    parser.add_argument('--start-mcp-servers', action='store_true', help='Start MCP servers')
    parser.add_argument('--demo-workflow', action='store_true', help='Run demo workflow')
    parser.add_argument('--list-tools', action='store_true', help='List available MCP tools')
    parser.add_argument('--list-resources', action='store_true', help='List available MCP resources')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    
    logger = get_logger("main_mcp")
    logger.info("Starting CCGL Analytics with MCP Architecture")
    
    try:
        # Initialize MCP configuration
        config_manager = create_default_mcp_config(args.config_dir)
        
        # Validate configuration
        validation_errors = config_manager.validate_configuration()
        if validation_errors:
            logger.warning(f"Configuration validation issues: {validation_errors}")
        
        # Initialize MCP orchestrator
        orchestrator = MCPOrchestrator(config_dir=args.config_dir)
        
        if args.start_mcp_servers:
            logger.info("Starting MCP servers...")
            await orchestrator.start_all_servers()
            
            # Display server status
            status = orchestrator.get_server_status()
            print("\n=== MCP Server Status ===")
            print(f"Active servers: {status['active_servers']}")
            print(f"Available tools: {status['tool_count']}")
            print(f"Available resources: {status['resource_count']}")
            print()
            
            if args.list_tools:
                print("=== Available Tools ===")
                tools = orchestrator.get_available_tools()
                for tool in tools:
                    print(f"  {tool['name']} ({tool['server']}): {tool['description']}")
                print()
            
            if args.list_resources:
                print("=== Available Resources ===")
                resources = orchestrator.get_available_resources()
                for resource in resources:
                    print(f"  {resource['name']} ({resource['server']}): {resource['description']}")
                print()
            
            if args.demo_workflow:
                print("=== Running Demo Workflow ===")
                workflow_result = await orchestrator.run_data_analysis_workflow("demo_dataset")
                
                if workflow_result["status"] == "success":
                    print(f"✅ Workflow completed successfully!")
                    print(f"   Steps completed: {', '.join(workflow_result['steps_completed'])}")
                    
                    # Display results summary
                    if "results" in workflow_result:
                        results = workflow_result["results"]
                        
                        if "quality_assessment" in results:
                            qa = results["quality_assessment"]["quality_report"]
                            print(f"   Data Quality Score: {qa['overall_score']:.2f}")
                        
                        if "missing_values" in results:
                            mv = results["missing_values"]
                            print(f"   Missing Values: {mv['rows_processed']} rows processed")
                        
                        if "outlier_detection" in results:
                            od = results["outlier_detection"]
                            print(f"   Outlier Detection: {od['rows_processed']} rows processed")
                        
                        if "normalization" in results:
                            norm = results["normalization"]
                            print(f"   Normalization: {norm['rows_processed']} rows processed")
                else:
                    print(f"❌ Workflow failed: {workflow_result.get('error', 'Unknown error')}")
                print()
            
            # Interactive mode
            if args.full_mcp:
                print("=== Interactive MCP Mode ===")
                print("Available commands:")
                print("  tools - List available tools")
                print("  resources - List available resources") 
                print("  status - Show server status")
                print("  workflow [dataset_id] - Run analysis workflow")
                print("  tool <tool_name> <args_json> - Call specific tool")
                print("  resource <uri> - Get specific resource")
                print("  quit - Exit")
                print()
                
                while True:
                    try:
                        command = input("MCP> ").strip()
                        
                        if command == "quit":
                            break
                        elif command == "tools":
                            tools = orchestrator.get_available_tools()
                            for tool in tools:
                                print(f"  {tool['name']}: {tool['description']}")
                        elif command == "resources":
                            resources = orchestrator.get_available_resources()
                            for resource in resources:
                                print(f"  {resource['uri']}: {resource['description']}")
                        elif command == "status":
                            status = orchestrator.get_server_status()
                            print(f"Active servers: {status['active_servers']}")
                            print(f"Tools: {status['tool_count']}, Resources: {status['resource_count']}")
                        elif command.startswith("workflow"):
                            parts = command.split()
                            dataset_id = parts[1] if len(parts) > 1 else "interactive_dataset"
                            result = await orchestrator.run_data_analysis_workflow(dataset_id)
                            print(f"Workflow result: {result['status']}")
                            if result["status"] == "success":
                                print(f"Steps: {', '.join(result['steps_completed'])}")
                        elif command.startswith("tool"):
                            parts = command.split(maxsplit=2)
                            if len(parts) >= 3:
                                tool_name = parts[1]
                                try:
                                    import json
                                    args = json.loads(parts[2])
                                    result = await orchestrator.call_tool(tool_name, args)
                                    print(f"Tool result: {result}")
                                except Exception as e:
                                    print(f"Error: {e}")
                            else:
                                print("Usage: tool <tool_name> <args_json>")
                        elif command.startswith("resource"):
                            parts = command.split(maxsplit=1)
                            if len(parts) >= 2:
                                uri = parts[1]
                                try:
                                    result = await orchestrator.get_resource(uri)
                                    print(f"Resource content: {result}")
                                except Exception as e:
                                    print(f"Error: {e}")
                            else:
                                print("Usage: resource <uri>")
                        else:
                            print("Unknown command. Type 'quit' to exit.")
                    
                    except EOFError:
                        break
                    except Exception as e:
                        print(f"Error: {e}")
            
            # Stop servers when done
            logger.info("Stopping MCP servers...")
            await orchestrator.stop_all_servers()
        
        else:
            # Non-MCP mode - just show configuration
            print("=== MCP Configuration Summary ===")
            summary = config_manager.get_configuration_summary()
            print(f"Servers configured: {summary['servers']}")
            print(f"LLM providers: {summary['llm_providers']}")
            print(f"Endpoints: {summary['endpoints']}")
            
            if summary['validation_errors']:
                print(f"Validation errors: {summary['validation_errors']}")
        
        logger.info("MCP operation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"MCP operation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)