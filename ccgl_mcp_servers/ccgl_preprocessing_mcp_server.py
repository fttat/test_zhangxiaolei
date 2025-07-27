"""
CCGL Data Preprocessing MCP Server
Provides data preprocessing and quality management tools via MCP
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ccgl_analytics.modules.data_preprocessing import DataPreprocessor, DataQualityAssessment
from ccgl_analytics.modules.data_connection import DataConnection
from ccgl_analytics.utils.logger import get_logger


class CCGLPreprocessingMCPServer:
    """
    MCP Server for data preprocessing functionality
    Exposes preprocessing tools and resources via Model Context Protocol
    """
    
    def __init__(self, port: int = 3001):
        """
        Initialize the preprocessing MCP server
        
        Args:
            port: Port to run the server on
        """
        self.port = port
        self.logger = get_logger("preprocessing_mcp_server")
        
        # Initialize preprocessing components
        self.preprocessor = DataPreprocessor()
        self.quality_assessor = DataQualityAssessment()
        
        # Server state
        self.active_datasets = {}
        self.processing_history = []
        
        self.logger.info(f"Initialized CCGL Preprocessing MCP Server on port {port}")
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """
        Return list of available tools
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "data_quality_assessment",
                "description": "Assess data quality including completeness, consistency, and accuracy",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Identifier for the dataset to assess"
                        },
                        "data": {
                            "type": "object",
                            "description": "Optional raw data if dataset_id not provided"
                        }
                    },
                    "required": ["dataset_id"]
                }
            },
            {
                "name": "handle_missing_values",
                "description": "Handle missing values in dataset using various strategies",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Identifier for the dataset to process"
                        },
                        "strategy": {
                            "type": "string",
                            "enum": ["auto", "mean", "median", "mode", "knn", "drop"],
                            "description": "Strategy for handling missing values",
                            "default": "auto"
                        }
                    },
                    "required": ["dataset_id"]
                }
            },
            {
                "name": "detect_outliers",
                "description": "Detect and handle outliers in the dataset",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Identifier for the dataset to process"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["iqr", "zscore", "isolation_forest"],
                            "description": "Method for outlier detection",
                            "default": "iqr"
                        }
                    },
                    "required": ["dataset_id"]
                }
            },
            {
                "name": "normalize_data",
                "description": "Normalize numeric data using various scaling methods",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Identifier for the dataset to normalize"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["standard", "minmax", "robust"],
                            "description": "Normalization method",
                            "default": "standard"
                        }
                    },
                    "required": ["dataset_id"]
                }
            },
            {
                "name": "encode_categorical",
                "description": "Encode categorical variables for machine learning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Identifier for the dataset to encode"
                        }
                    },
                    "required": ["dataset_id"]
                }
            },
            {
                "name": "full_preprocessing_pipeline",
                "description": "Run complete preprocessing pipeline on dataset",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset_id": {
                            "type": "string",
                            "description": "Identifier for the dataset to process"
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["quality_check", "handle_missing", "detect_outliers", "normalize_data", "encode_categorical"]
                            },
                            "description": "List of preprocessing steps to apply",
                            "default": ["quality_check", "handle_missing", "detect_outliers", "normalize_data"]
                        }
                    },
                    "required": ["dataset_id"]
                }
            }
        ]
    
    async def get_resources(self) -> List[Dict[str, Any]]:
        """
        Return list of available resources
        
        Returns:
            List of resource definitions
        """
        return [
            {
                "uri": "preprocessing://quality_reports",
                "name": "Data Quality Reports",
                "description": "Access historical data quality assessment reports",
                "mimeType": "application/json"
            },
            {
                "uri": "preprocessing://preprocessing_history", 
                "name": "Preprocessing History",
                "description": "Access history of preprocessing operations",
                "mimeType": "application/json"
            },
            {
                "uri": "preprocessing://active_datasets",
                "name": "Active Datasets",
                "description": "List of currently loaded datasets",
                "mimeType": "application/json"
            }
        ]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given arguments
        
        Args:
            name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        try:
            self.logger.info(f"Executing tool: {name} with arguments: {arguments}")
            
            if name == "data_quality_assessment":
                return await self._assess_data_quality(arguments)
            elif name == "handle_missing_values":
                return await self._handle_missing_values(arguments)
            elif name == "detect_outliers":
                return await self._detect_outliers(arguments)
            elif name == "normalize_data":
                return await self._normalize_data(arguments)
            elif name == "encode_categorical":
                return await self._encode_categorical(arguments)
            elif name == "full_preprocessing_pipeline":
                return await self._run_preprocessing_pipeline(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return {
                "error": str(e),
                "tool": name,
                "arguments": arguments
            }
    
    async def get_resource(self, uri: str) -> Dict[str, Any]:
        """
        Get a resource by URI
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content
        """
        try:
            if uri == "preprocessing://quality_reports":
                return {
                    "content": [entry for entry in self.processing_history if entry.get("type") == "quality_assessment"],
                    "mimeType": "application/json"
                }
            elif uri == "preprocessing://preprocessing_history":
                return {
                    "content": self.processing_history,
                    "mimeType": "application/json"
                }
            elif uri == "preprocessing://active_datasets":
                return {
                    "content": list(self.active_datasets.keys()),
                    "mimeType": "application/json"
                }
            else:
                raise ValueError(f"Unknown resource URI: {uri}")
                
        except Exception as e:
            self.logger.error(f"Resource access failed: {str(e)}")
            return {
                "error": str(e),
                "uri": uri
            }
    
    async def _assess_data_quality(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality for a dataset"""
        dataset_id = arguments["dataset_id"]
        
        if dataset_id not in self.active_datasets:
            # Load sample data for demonstration
            from ccgl_analytics.modules.data_connection import create_sample_data
            df = create_sample_data()
            self.active_datasets[dataset_id] = df
        
        df = self.active_datasets[dataset_id]
        quality_report = self.quality_assessor.assess_quality(df)
        
        # Store in history
        self.processing_history.append({
            "type": "quality_assessment",
            "dataset_id": dataset_id,
            "timestamp": str(asyncio.get_event_loop().time()),
            "result": quality_report
        })
        
        return {
            "dataset_id": dataset_id,
            "quality_report": quality_report,
            "status": "success"
        }
    
    async def _handle_missing_values(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing values in dataset"""
        dataset_id = arguments["dataset_id"]
        strategy = arguments.get("strategy", "auto")
        
        if dataset_id not in self.active_datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        df = self.active_datasets[dataset_id]
        processed_df = self.preprocessor.handle_missing_values(df, strategy)
        
        # Update dataset
        self.active_datasets[dataset_id] = processed_df
        
        # Store in history
        self.processing_history.append({
            "type": "missing_values",
            "dataset_id": dataset_id,
            "strategy": strategy,
            "timestamp": str(asyncio.get_event_loop().time()),
            "status": "success"
        })
        
        return {
            "dataset_id": dataset_id,
            "strategy": strategy,
            "rows_processed": len(processed_df),
            "status": "success"
        }
    
    async def _detect_outliers(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Detect and handle outliers"""
        dataset_id = arguments["dataset_id"]
        method = arguments.get("method", "iqr")
        
        if dataset_id not in self.active_datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        df = self.active_datasets[dataset_id]
        processed_df = self.preprocessor.detect_and_handle_outliers(df, method)
        
        # Update dataset
        self.active_datasets[dataset_id] = processed_df
        
        # Store in history
        self.processing_history.append({
            "type": "outlier_detection",
            "dataset_id": dataset_id,
            "method": method,
            "timestamp": str(asyncio.get_event_loop().time()),
            "status": "success"
        })
        
        return {
            "dataset_id": dataset_id,
            "method": method,
            "rows_processed": len(processed_df),
            "status": "success"
        }
    
    async def _normalize_data(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data"""
        dataset_id = arguments["dataset_id"]
        method = arguments.get("method", "standard")
        
        if dataset_id not in self.active_datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        df = self.active_datasets[dataset_id]
        processed_df = self.preprocessor.normalize_data(df, method)
        
        # Update dataset
        self.active_datasets[dataset_id] = processed_df
        
        # Store in history
        self.processing_history.append({
            "type": "normalization",
            "dataset_id": dataset_id,
            "method": method,
            "timestamp": str(asyncio.get_event_loop().time()),
            "status": "success"
        })
        
        return {
            "dataset_id": dataset_id,
            "method": method,
            "rows_processed": len(processed_df),
            "status": "success"
        }
    
    async def _encode_categorical(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Encode categorical variables"""
        dataset_id = arguments["dataset_id"]
        
        if dataset_id not in self.active_datasets:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        df = self.active_datasets[dataset_id]
        processed_df = self.preprocessor.encode_categorical_variables(df)
        
        # Update dataset
        self.active_datasets[dataset_id] = processed_df
        
        # Store in history
        self.processing_history.append({
            "type": "categorical_encoding",
            "dataset_id": dataset_id,
            "timestamp": str(asyncio.get_event_loop().time()),
            "status": "success"
        })
        
        return {
            "dataset_id": dataset_id,
            "rows_processed": len(processed_df),
            "columns_processed": len(processed_df.columns),
            "status": "success"
        }
    
    async def _run_preprocessing_pipeline(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Run full preprocessing pipeline"""
        dataset_id = arguments["dataset_id"]
        steps = arguments.get("steps", ["quality_check", "handle_missing", "detect_outliers", "normalize_data"])
        
        if dataset_id not in self.active_datasets:
            # Load sample data for demonstration
            from ccgl_analytics.modules.data_connection import create_sample_data
            df = create_sample_data()
            self.active_datasets[dataset_id] = df
        
        df = self.active_datasets[dataset_id]
        processed_df = self.preprocessor.preprocess(df, steps)
        
        # Update dataset
        self.active_datasets[dataset_id] = processed_df
        
        # Get preprocessing summary
        summary = self.preprocessor.get_preprocessing_summary()
        
        # Store in history
        self.processing_history.append({
            "type": "full_pipeline",
            "dataset_id": dataset_id,
            "steps": steps,
            "summary": summary,
            "timestamp": str(asyncio.get_event_loop().time()),
            "status": "success"
        })
        
        return {
            "dataset_id": dataset_id,
            "steps_completed": steps,
            "preprocessing_summary": summary,
            "rows_processed": len(processed_df),
            "status": "success"
        }
    
    async def start_server(self):
        """Start the MCP server"""
        self.logger.info(f"Starting CCGL Preprocessing MCP Server on port {self.port}")
        
        # For now, just start a simple message loop
        # In a real MCP implementation, this would start the actual MCP server
        while True:
            await asyncio.sleep(1)
            # Server heartbeat
            pass


async def main():
    """Main entry point for the MCP server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CCGL Preprocessing MCP Server')
    parser.add_argument('--port', type=int, default=3001, help='Port to run server on')
    args = parser.parse_args()
    
    server = CCGLPreprocessingMCPServer(port=args.port)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        server.logger.info("Server stopped by user")
    except Exception as e:
        server.logger.error(f"Server error: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())