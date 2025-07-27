"""
MCP编排器

管理和协调所有MCP服务器的运行
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime


class MCPOrchestrator:
    """MCP服务器编排器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化MCP编排器"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 服务器状态跟踪
        self.servers = {}
        self.server_status = {}
        
        # 编排策略
        self.load_balancing = config.get('load_balancing', 'round_robin')
        self.health_check_interval = config.get('health_check_interval', 30)
        
        # 任务队列
        self.task_queue = asyncio.Queue()
        self.result_cache = {}
        
        self.logger.info("MCP编排器初始化完成")
    
    async def start(self):
        """启动编排器"""
        try:
            self.logger.info("启动MCP编排器")
            
            # 启动健康检查任务
            asyncio.create_task(self._health_check_loop())
            
            # 启动任务处理器
            asyncio.create_task(self._task_processor())
            
            self.logger.info("MCP编排器启动成功")
            
        except Exception as e:
            self.logger.error(f"MCP编排器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止编排器"""
        try:
            self.logger.info("停止MCP编排器")
            
            # 停止所有注册的服务器
            for server_name, server in self.servers.items():
                if hasattr(server, 'stop'):
                    await server.stop()
                    self.logger.info(f"服务器 {server_name} 已停止")
            
            self.logger.info("MCP编排器已停止")
            
        except Exception as e:
            self.logger.error(f"MCP编排器停止失败: {e}")
    
    def register_server(self, name: str, server: Any):
        """注册MCP服务器"""
        self.servers[name] = server
        self.server_status[name] = {
            'status': 'registered',
            'last_check': datetime.now(),
            'response_time': 0,
            'error_count': 0
        }
        self.logger.info(f"服务器 {name} 已注册")
    
    def unregister_server(self, name: str):
        """注销MCP服务器"""
        if name in self.servers:
            del self.servers[name]
            del self.server_status[name]
            self.logger.info(f"服务器 {name} 已注销")
    
    async def execute_task(self, task_type: str, 
                          data: Dict[str, Any],
                          server_preference: Optional[str] = None) -> Dict[str, Any]:
        """执行任务"""
        try:
            # 选择合适的服务器
            server_name = await self._select_server(task_type, server_preference)
            
            if not server_name:
                return {'error': f'没有可用的服务器执行任务: {task_type}'}
            
            # 执行任务
            result = await self._execute_on_server(server_name, task_type, data)
            
            # 缓存结果
            cache_key = self._generate_cache_key(task_type, data)
            self.result_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now(),
                'server': server_name
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            return {'error': str(e)}
    
    async def _select_server(self, task_type: str, 
                           preference: Optional[str] = None) -> Optional[str]:
        """选择服务器"""
        # 如果指定了偏好服务器且可用，优先使用
        if preference and preference in self.servers:
            if self.server_status[preference]['status'] == 'healthy':
                return preference
        
        # 根据任务类型选择合适的服务器
        suitable_servers = []
        
        for server_name, server in self.servers.items():
            if self._can_handle_task(server_name, task_type):
                if self.server_status[server_name]['status'] == 'healthy':
                    suitable_servers.append(server_name)
        
        if not suitable_servers:
            return None
        
        # 负载均衡选择
        if self.load_balancing == 'round_robin':
            return self._round_robin_select(suitable_servers)
        elif self.load_balancing == 'least_loaded':
            return self._least_loaded_select(suitable_servers)
        else:
            return suitable_servers[0]
    
    def _can_handle_task(self, server_name: str, task_type: str) -> bool:
        """检查服务器是否可以处理指定任务"""
        task_mapping = {
            'preprocessing': ['preprocessing'],
            'ml_analysis': ['clustering', 'anomaly', 'association', 'dimensionality_reduction'],
            'dashboard': ['visualization', 'reporting'],
            'llm_integration': ['nlp', 'text_analysis', 'report_generation']
        }
        
        for server_type, supported_tasks in task_mapping.items():
            if server_name.startswith(server_type) or server_type in server_name:
                return task_type in supported_tasks or task_type.startswith(server_type)
        
        return False
    
    def _round_robin_select(self, servers: List[str]) -> str:
        """轮询选择服务器"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        server = servers[self._round_robin_index % len(servers)]
        self._round_robin_index += 1
        
        return server
    
    def _least_loaded_select(self, servers: List[str]) -> str:
        """选择负载最低的服务器"""
        # 简单实现：基于错误计数选择
        min_errors = float('inf')
        selected_server = servers[0]
        
        for server in servers:
            error_count = self.server_status[server]['error_count']
            if error_count < min_errors:
                min_errors = error_count
                selected_server = server
        
        return selected_server
    
    async def _execute_on_server(self, server_name: str, 
                                task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """在指定服务器上执行任务"""
        server = self.servers[server_name]
        
        start_time = datetime.now()
        
        try:
            # 假设所有服务器都有execute_task方法
            if hasattr(server, 'execute_task'):
                result = await server.execute_task(task_type, data)
            else:
                # 模拟任务执行
                result = await self._simulate_task_execution(server_name, task_type, data)
            
            # 更新服务器状态
            response_time = (datetime.now() - start_time).total_seconds()
            self.server_status[server_name]['response_time'] = response_time
            self.server_status[server_name]['last_check'] = datetime.now()
            
            return result
            
        except Exception as e:
            # 记录错误
            self.server_status[server_name]['error_count'] += 1
            self.logger.error(f"服务器 {server_name} 执行任务失败: {e}")
            raise
    
    async def _simulate_task_execution(self, server_name: str, 
                                     task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟任务执行（用于演示）"""
        # 模拟处理时间
        await asyncio.sleep(0.1)
        
        return {
            'status': 'completed',
            'server': server_name,
            'task_type': task_type,
            'timestamp': datetime.now().isoformat(),
            'result': f'Task {task_type} completed on {server_name}'
        }
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"健康检查失败: {e}")
                await asyncio.sleep(5)  # 错误时短暂等待
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        for server_name, server in self.servers.items():
            try:
                # 执行健康检查
                health_status = await self._check_server_health(server_name, server)
                
                # 更新服务器状态
                self.server_status[server_name].update({
                    'status': health_status['status'],
                    'last_check': datetime.now(),
                    'response_time': health_status.get('response_time', 0)
                })
                
            except Exception as e:
                self.server_status[server_name]['status'] = 'unhealthy'
                self.server_status[server_name]['error_count'] += 1
                self.logger.warning(f"服务器 {server_name} 健康检查失败: {e}")
    
    async def _check_server_health(self, server_name: str, server: Any) -> Dict[str, Any]:
        """检查单个服务器健康状态"""
        start_time = datetime.now()
        
        try:
            # 尝试调用服务器的健康检查方法
            if hasattr(server, 'health_check'):
                health_result = await server.health_check()
            else:
                # 默认健康检查
                health_result = {'status': 'healthy'}
            
            response_time = (datetime.now() - start_time).total_seconds()
            health_result['response_time'] = response_time
            
            return health_result
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _task_processor(self):
        """任务处理器"""
        while True:
            try:
                # 从队列获取任务
                task = await self.task_queue.get()
                
                # 执行任务
                result = await self.execute_task(
                    task['type'],
                    task['data'],
                    task.get('server_preference')
                )
                
                # 设置结果
                if 'result_future' in task:
                    task['result_future'].set_result(result)
                
                self.task_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"任务处理失败: {e}")
    
    def _generate_cache_key(self, task_type: str, data: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        
        cache_data = {
            'task_type': task_type,
            'data': json.dumps(data, sort_keys=True)
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    async def get_status(self) -> Dict[str, Any]:
        """获取编排器状态"""
        return {
            'orchestrator_status': 'running',
            'registered_servers': len(self.servers),
            'server_status': self.server_status.copy(),
            'task_queue_size': self.task_queue.qsize(),
            'cache_size': len(self.result_cache),
            'load_balancing': self.load_balancing,
            'health_check_interval': self.health_check_interval
        }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        healthy_servers = sum(1 for status in self.server_status.values() 
                             if status['status'] == 'healthy')
        
        total_errors = sum(status['error_count'] for status in self.server_status.values())
        
        avg_response_time = 0
        if self.server_status:
            avg_response_time = sum(status['response_time'] for status in self.server_status.values()) / len(self.server_status)
        
        return {
            'healthy_servers': healthy_servers,
            'total_servers': len(self.servers),
            'total_errors': total_errors,
            'average_response_time': avg_response_time,
            'cache_hit_ratio': self._calculate_cache_hit_ratio(),
            'system_health': 'healthy' if healthy_servers > 0 else 'degraded'
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """计算缓存命中率"""
        # 简单实现，实际应该跟踪缓存命中和未命中次数
        return 0.85  # 模拟85%的缓存命中率