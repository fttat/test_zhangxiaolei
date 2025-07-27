"""
机器学习MCP服务器

专门处理机器学习分析任务的MCP服务器
"""

import asyncio
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd

# 导入核心分析模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ccgl_analytics.modules.analysis_core import AnalysisCore
from ccgl_analytics.modules.data_connection import DataConnection


class MachineLearningMCPServer:
    """机器学习MCP服务器"""
    
    def __init__(self, port: int, config: Dict[str, Any]):
        """初始化机器学习服务器"""
        self.port = port
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化核心组件
        self.analysis_core = AnalysisCore(config.get('machine_learning', {}))
        self.data_connection = DataConnection(config.get('database', {}))
        
        # 服务器状态
        self.is_running = False
        self.start_time = None
        self.processed_requests = 0
        self.model_cache = {}
        
        self.logger.info(f"机器学习MCP服务器初始化完成，端口: {port}")
    
    async def start(self):
        """启动服务器"""
        try:
            self.logger.info(f"启动机器学习MCP服务器，端口: {self.port}")
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # 模拟启动Web服务器
            await self._start_mock_server()
            
        except Exception as e:
            self.logger.error(f"服务器启动失败: {e}")
            raise
    
    async def stop(self):
        """停止服务器"""
        try:
            self.logger.info("停止机器学习MCP服务器")
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"服务器停止失败: {e}")
    
    async def _start_mock_server(self):
        """启动模拟服务器"""
        self.logger.info(f"机器学习服务器运行在端口 {self.port}")
        
        # 模拟服务器持续运行
        while self.is_running:
            await asyncio.sleep(1)
    
    async def execute_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行机器学习任务"""
        try:
            self.processed_requests += 1
            self.logger.info(f"执行机器学习任务: {task_type}")
            
            if task_type == 'clustering':
                return await self._handle_clustering_task(data)
            elif task_type == 'anomaly':
                return await self._handle_anomaly_task(data)
            elif task_type == 'association':
                return await self._handle_association_task(data)
            elif task_type == 'dimensionality_reduction':
                return await self._handle_dimensionality_reduction_task(data)
            elif task_type == 'prediction':
                return await self._handle_prediction_task(data)
            else:
                return {'error': f'不支持的任务类型: {task_type}'}
                
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            return {'error': str(e)}
    
    async def _handle_clustering_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理聚类分析任务"""
        try:
            # 获取数据
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            algorithm = data.get('algorithm', 'kmeans')
            
            # 执行聚类分析
            clustering_result = await self.analysis_core.cluster_analysis(df, algorithm)
            
            # 缓存模型
            if 'model' in clustering_result:
                model_id = f"clustering_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.model_cache[model_id] = clustering_result['model']
                clustering_result['model_id'] = model_id
                del clustering_result['model']  # 不返回模型对象
            
            return {
                'status': 'completed',
                'task_type': 'clustering',
                'algorithm': algorithm,
                'result': clustering_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"聚类分析失败: {e}")
            return {'error': str(e)}
    
    async def _handle_anomaly_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理异常检测任务"""
        try:
            # 获取数据
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            algorithms = data.get('algorithms', ['isolation_forest'])
            
            # 执行异常检测
            anomaly_result = await self.analysis_core.anomaly_detection(df, algorithms)
            
            # 处理模型缓存
            for algorithm, result in anomaly_result.get('individual_results', {}).items():
                if 'model' in result:
                    model_id = f"anomaly_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.model_cache[model_id] = result['model']
                    result['model_id'] = model_id
                    del result['model']
            
            return {
                'status': 'completed',
                'task_type': 'anomaly_detection',
                'algorithms': algorithms,
                'result': anomaly_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            return {'error': str(e)}
    
    async def _handle_association_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理关联规则挖掘任务"""
        try:
            # 获取数据
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            min_support = data.get('min_support', 0.01)
            min_confidence = data.get('min_confidence', 0.5)
            
            # 执行关联规则挖掘
            association_result = await self.analysis_core.association_rules(
                df, min_support, min_confidence
            )
            
            # 转换DataFrame为可序列化格式
            if 'frequent_itemsets' in association_result and not association_result['frequent_itemsets'].empty:
                association_result['frequent_itemsets'] = association_result['frequent_itemsets'].to_dict('records')
            
            if 'rules' in association_result and not association_result['rules'].empty:
                association_result['rules'] = association_result['rules'].to_dict('records')
            
            return {
                'status': 'completed',
                'task_type': 'association_rules',
                'parameters': {
                    'min_support': min_support,
                    'min_confidence': min_confidence
                },
                'result': association_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"关联规则挖掘失败: {e}")
            return {'error': str(e)}
    
    async def _handle_dimensionality_reduction_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理降维任务"""
        try:
            # 获取数据
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            methods = data.get('methods', ['pca'])
            
            # 执行降维分析
            dimred_result = await self.analysis_core.dimensionality_reduction(df, methods)
            
            # 处理模型缓存和数据序列化
            for method, result in dimred_result.items():
                if 'model' in result:
                    model_id = f"dimred_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.model_cache[model_id] = result['model']
                    result['model_id'] = model_id
                    del result['model']
                
                # 转换numpy数组为列表
                if 'transformed_data' in result:
                    result['transformed_data'] = result['transformed_data'].tolist()
                if 'components' in result:
                    result['components'] = result['components'].tolist()
                if 'explained_variance_ratio' in result:
                    result['explained_variance_ratio'] = result['explained_variance_ratio'].tolist()
                if 'cumulative_variance_ratio' in result:
                    result['cumulative_variance_ratio'] = result['cumulative_variance_ratio'].tolist()
            
            return {
                'status': 'completed',
                'task_type': 'dimensionality_reduction',
                'methods': methods,
                'result': dimred_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"降维分析失败: {e}")
            return {'error': str(e)}
    
    async def _handle_prediction_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理预测任务"""
        try:
            model_id = data.get('model_id')
            input_data = data.get('input_data')
            
            if not model_id or model_id not in self.model_cache:
                return {'error': '模型不存在或已过期'}
            
            if not input_data:
                return {'error': '缺少输入数据'}
            
            model = self.model_cache[model_id]
            
            # 转换输入数据
            if isinstance(input_data, list):
                input_df = pd.DataFrame(input_data)
            else:
                input_df = pd.DataFrame([input_data])
            
            # 执行预测
            if hasattr(model, 'predict'):
                predictions = model.predict(input_df)
                
                # 如果是聚类模型，还可以计算概率
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(input_df)
                    except:
                        pass
                
                result = {
                    'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                    'probabilities': probabilities.tolist() if probabilities is not None else None,
                    'n_samples': len(input_df)
                }
            else:
                return {'error': '模型不支持预测功能'}
            
            return {
                'status': 'completed',
                'task_type': 'prediction',
                'model_id': model_id,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"预测任务失败: {e}")
            return {'error': str(e)}
    
    async def _get_dataframe_from_request(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """从请求中获取DataFrame"""
        if 'dataframe' in data:
            return pd.DataFrame(data['dataframe'])
        elif 'query' in data:
            return await self.data_connection.execute_query(data['query'])
        else:
            return await self.data_connection.get_data()
    
    async def get_models(self) -> Dict[str, Any]:
        """获取已缓存的模型列表"""
        models_info = {}
        
        for model_id, model in self.model_cache.items():
            model_info = {
                'model_id': model_id,
                'model_type': type(model).__name__,
                'created_time': model_id.split('_')[-1] if '_' in model_id else 'unknown'
            }
            
            # 添加模型特定信息
            if hasattr(model, 'n_clusters'):
                model_info['n_clusters'] = model.n_clusters
            if hasattr(model, 'contamination'):
                model_info['contamination'] = model.contamination
            if hasattr(model, 'n_components'):
                model_info['n_components'] = model.n_components
            
            models_info[model_id] = model_info
        
        return {
            'total_models': len(self.model_cache),
            'models': models_info
        }
    
    async def clear_model_cache(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """清理模型缓存"""
        try:
            if model_id:
                if model_id in self.model_cache:
                    del self.model_cache[model_id]
                    return {'status': 'success', 'message': f'模型 {model_id} 已删除'}
                else:
                    return {'status': 'error', 'message': f'模型 {model_id} 不存在'}
            else:
                cleared_count = len(self.model_cache)
                self.model_cache.clear()
                return {'status': 'success', 'message': f'已清理 {cleared_count} 个模型'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试数据连接
            db_healthy = await self.data_connection.test_connection()
            
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                'status': 'healthy' if self.is_running and db_healthy else 'unhealthy',
                'server_type': 'machine_learning',
                'port': self.port,
                'uptime': uptime,
                'processed_requests': self.processed_requests,
                'cached_models': len(self.model_cache),
                'database_connection': 'healthy' if db_healthy else 'unhealthy',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """获取服务器状态"""
        return {
            'server_type': 'machine_learning',
            'port': self.port,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'processed_requests': self.processed_requests,
            'cached_models': len(self.model_cache),
            'supported_tasks': [
                'clustering',
                'anomaly',
                'association',
                'dimensionality_reduction',
                'prediction'
            ]
        }