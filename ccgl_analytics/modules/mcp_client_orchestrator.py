"""
CCGL Analytics - MCP Client Orchestrator
Orchestrates multiple MCP clients and manages distributed analysis
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import uuid

from ..utils.logger import get_logger, LoggerMixin

class MCPClientOrchestrator(LoggerMixin):
    """Orchestrator for managing multiple MCP clients."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MCP client orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.clients = {}
        self.active_tasks = {}
        self.task_results = {}
        self.running = False
        
    async def register_client(self, client_name: str, client_instance: Any) -> bool:
        """Register an MCP client.
        
        Args:
            client_name: Name of the client
            client_instance: Client instance
            
        Returns:
            True if registered successfully
        """
        try:
            self.clients[client_name] = client_instance
            self.logger.info(f"Registered MCP client: {client_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register client {client_name}: {e}")
            return False
    
    async def unregister_client(self, client_name: str) -> bool:
        """Unregister an MCP client.
        
        Args:
            client_name: Name of the client
            
        Returns:
            True if unregistered successfully
        """
        try:
            if client_name in self.clients:
                del self.clients[client_name]
                self.logger.info(f"Unregistered MCP client: {client_name}")
                return True
            else:
                self.logger.warning(f"Client not found: {client_name}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to unregister client {client_name}: {e}")
            return False
    
    async def execute_distributed_task(self, task_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a distributed task across multiple clients.
        
        Args:
            task_definition: Task definition
            
        Returns:
            Task execution results
        """
        task_id = str(uuid.uuid4())
        self.logger.info(f"Executing distributed task: {task_id}")
        
        try:
            task_name = task_definition.get('name', f'task_{task_id}')
            subtasks = task_definition.get('subtasks', [])
            execution_mode = task_definition.get('execution_mode', 'parallel')
            
            task_result = {
                'task_id': task_id,
                'task_name': task_name,
                'status': 'running',
                'execution_mode': execution_mode,
                'subtasks_completed': 0,
                'total_subtasks': len(subtasks),
                'subtask_results': {},
                'start_time': datetime.now().isoformat()
            }
            
            self.active_tasks[task_id] = task_result
            
            if execution_mode == 'parallel':
                # Execute subtasks in parallel
                task_result = await self._execute_parallel_subtasks(task_id, subtasks, task_result)
            else:
                # Execute subtasks sequentially
                task_result = await self._execute_sequential_subtasks(task_id, subtasks, task_result)
            
            # Finalize task
            task_result['end_time'] = datetime.now().isoformat()
            task_result['status'] = 'completed' if task_result['subtasks_completed'] == task_result['total_subtasks'] else 'failed'
            
            self.task_results[task_id] = task_result
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            self.logger.info(f"Distributed task {task_id} {task_result['status']}")
            return task_result
            
        except Exception as e:
            self.logger.error(f"Distributed task {task_id} failed: {e}")
            error_result = {
                'task_id': task_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.task_results[task_id] = error_result
            return error_result
    
    async def _execute_parallel_subtasks(self, task_id: str, subtasks: List[Dict[str, Any]], 
                                       task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtasks in parallel.
        
        Args:
            task_id: Task identifier
            subtasks: List of subtasks
            task_result: Task result dictionary
            
        Returns:
            Updated task result
        """
        # Create tasks for parallel execution
        async_tasks = []
        
        for i, subtask in enumerate(subtasks):
            subtask_name = subtask.get('name', f'subtask_{i}')
            task = asyncio.create_task(
                self._execute_single_subtask(subtask),
                name=subtask_name
            )
            async_tasks.append((subtask_name, task))
        
        # Wait for all tasks to complete
        for subtask_name, task in async_tasks:
            try:
                subtask_result = await task
                task_result['subtask_results'][subtask_name] = subtask_result
                
                if subtask_result.get('status') == 'success':
                    task_result['subtasks_completed'] += 1
                    
            except Exception as e:
                self.logger.error(f"Subtask {subtask_name} failed: {e}")
                task_result['subtask_results'][subtask_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return task_result
    
    async def _execute_sequential_subtasks(self, task_id: str, subtasks: List[Dict[str, Any]], 
                                         task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtasks sequentially.
        
        Args:
            task_id: Task identifier
            subtasks: List of subtasks
            task_result: Task result dictionary
            
        Returns:
            Updated task result
        """
        for i, subtask in enumerate(subtasks):
            subtask_name = subtask.get('name', f'subtask_{i}')
            
            try:
                subtask_result = await self._execute_single_subtask(subtask)
                task_result['subtask_results'][subtask_name] = subtask_result
                
                if subtask_result.get('status') == 'success':
                    task_result['subtasks_completed'] += 1
                else:
                    # Stop execution on first failure in sequential mode
                    self.logger.error(f"Sequential subtask {subtask_name} failed, stopping execution")
                    break
                    
            except Exception as e:
                self.logger.error(f"Subtask {subtask_name} failed: {e}")
                task_result['subtask_results'][subtask_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                break
        
        return task_result
    
    async def _execute_single_subtask(self, subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single subtask.
        
        Args:
            subtask: Subtask definition
            
        Returns:
            Subtask execution result
        """
        client_name = subtask.get('client')
        method = subtask.get('method')
        params = subtask.get('params', {})
        
        if not client_name or client_name not in self.clients:
            return {
                'status': 'error',
                'error': f'Client not found: {client_name}'
            }
        
        if not method:
            return {
                'status': 'error',
                'error': 'No method specified'
            }
        
        try:
            client = self.clients[client_name]
            
            # Create MCP request
            request = {
                'method': method,
                'params': params
            }
            
            # Execute request
            if hasattr(client, 'handle_mcp_request'):
                result = await client.handle_mcp_request(request)
            else:
                result = {
                    'status': 'error',
                    'error': f'Client {client_name} does not support MCP requests'
                }
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status or None if not found
        """
        # Check active tasks first
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        
        # Check completed tasks
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        return None
    
    async def get_all_tasks_status(self) -> Dict[str, Any]:
        """Get status of all tasks.
        
        Returns:
            Status of all tasks
        """
        return {
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.task_results),
            'active_task_ids': list(self.active_tasks.keys()),
            'completed_task_ids': list(self.task_results.keys()),
            'timestamp': datetime.now().isoformat()
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancelled successfully
        """
        try:
            if task_id in self.active_tasks:
                # Mark task as cancelled
                self.active_tasks[task_id]['status'] = 'cancelled'
                self.active_tasks[task_id]['end_time'] = datetime.now().isoformat()
                
                # Move to results
                self.task_results[task_id] = self.active_tasks[task_id]
                del self.active_tasks[task_id]
                
                self.logger.info(f"Task {task_id} cancelled")
                return True
            else:
                self.logger.warning(f"Task {task_id} not found or not active")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def cleanup_old_results(self, max_age_hours: int = 24):
        """Clean up old task results.
        
        Args:
            max_age_hours: Maximum age of results to keep
        """
        try:
            current_time = datetime.now()
            cutoff_time = current_time.timestamp() - (max_age_hours * 3600)
            
            tasks_to_remove = []
            
            for task_id, result in self.task_results.items():
                end_time_str = result.get('end_time')
                if end_time_str:
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                    if end_time.timestamp() < cutoff_time:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.task_results[task_id]
            
            if tasks_to_remove:
                self.logger.info(f"Cleaned up {len(tasks_to_remove)} old task results")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old results: {e}")
    
    def create_sample_distributed_task(self) -> Dict[str, Any]:
        """Create a sample distributed task for testing.
        
        Returns:
            Sample task definition
        """
        return {
            'name': 'Sample Distributed Analysis',
            'description': 'Distributed data analysis across multiple MCP clients',
            'execution_mode': 'parallel',
            'subtasks': [
                {
                    'name': 'data_preprocessing',
                    'client': 'preprocessing_client',
                    'method': 'preprocess_data',
                    'params': {
                        'data_source': 'sample_data',
                        'steps': ['missing_values', 'outliers', 'scaling']
                    }
                },
                {
                    'name': 'clustering_analysis',
                    'client': 'ml_client',
                    'method': 'perform_clustering',
                    'params': {
                        'algorithm': 'kmeans',
                        'n_clusters': 3
                    }
                },
                {
                    'name': 'anomaly_detection',
                    'client': 'ml_client',
                    'method': 'detect_anomalies',
                    'params': {
                        'algorithm': 'isolation_forest',
                        'contamination': 0.1
                    }
                },
                {
                    'name': 'visualization',
                    'client': 'dashboard_client',
                    'method': 'create_visualizations',
                    'params': {
                        'chart_types': ['scatter', 'bar', 'heatmap']
                    }
                }
            ]
        }