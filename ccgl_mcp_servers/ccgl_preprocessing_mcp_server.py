"""
数据预处理MCP服务器

专门处理数据清洗、预处理和质量评估的MCP服务器
"""

import asyncio
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import pandas as pd

# 导入核心预处理模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from ccgl_analytics.modules.data_preprocessing import DataPreprocessor
from ccgl_analytics.modules.data_connection import DataConnection


class PreprocessingMCPServer:
    """数据预处理MCP服务器"""
    
    def __init__(self, port: int, config: Dict[str, Any]):
        """初始化预处理服务器"""
        self.port = port
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 初始化核心组件
        self.data_preprocessor = DataPreprocessor(config.get('data_processing', {}))
        self.data_connection = DataConnection(config.get('database', {}))
        
        # 服务器状态
        self.is_running = False
        self.start_time = None
        self.processed_requests = 0
        
        self.logger.info(f"数据预处理MCP服务器初始化完成，端口: {port}")
    
    async def start(self):
        """启动服务器"""
        try:
            self.logger.info(f"启动数据预处理MCP服务器，端口: {self.port}")
            
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
            self.logger.info("停止数据预处理MCP服务器")
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"服务器停止失败: {e}")
    
    async def _start_mock_server(self):
        """启动模拟服务器"""
        self.logger.info(f"数据预处理服务器运行在端口 {self.port}")
        
        # 模拟服务器持续运行
        while self.is_running:
            await asyncio.sleep(1)
    
    async def execute_task(self, task_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行预处理任务"""
        try:
            self.processed_requests += 1
            self.logger.info(f"执行预处理任务: {task_type}")
            
            if task_type == 'preprocess':
                return await self._handle_preprocess_task(data)
            elif task_type == 'data_quality':
                return await self._handle_data_quality_task(data)
            elif task_type == 'missing_values':
                return await self._handle_missing_values_task(data)
            elif task_type == 'outlier_detection':
                return await self._handle_outlier_detection_task(data)
            elif task_type == 'normalization':
                return await self._handle_normalization_task(data)
            else:
                return {'error': f'不支持的任务类型: {task_type}'}
                
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            return {'error': str(e)}
    
    async def _handle_preprocess_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理完整预处理任务"""
        try:
            # 获取数据
            if 'dataframe' in data:
                # 直接使用提供的数据
                df = pd.DataFrame(data['dataframe'])
            else:
                # 从数据库获取数据
                df = await self.data_connection.get_data()
                if df is None:
                    return {'error': '无法获取数据'}
            
            original_shape = df.shape
            
            # 执行预处理
            processed_df = await self.data_preprocessor.process(df)
            
            # 生成预处理摘要
            summary = self.data_preprocessor.get_preprocessing_summary(df, processed_df)
            
            return {
                'status': 'completed',
                'original_shape': original_shape,
                'processed_shape': processed_df.shape,
                'summary': summary,
                'data_sample': processed_df.head(10).to_dict('records'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"预处理任务失败: {e}")
            return {'error': str(e)}
    
    async def _handle_data_quality_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据质量评估任务"""
        try:
            # 获取数据
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            # 数据质量评估
            quality_report = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'data_types': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'column_stats': {}
            }
            
            # 数值列统计
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                quality_report['column_stats'][col] = {
                    'type': 'numeric',
                    'missing_count': df[col].isnull().sum(),
                    'unique_count': df[col].nunique(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
            
            # 分类列统计
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                quality_report['column_stats'][col] = {
                    'type': 'categorical',
                    'missing_count': df[col].isnull().sum(),
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head(5).to_dict()
                }
            
            return {
                'status': 'completed',
                'quality_report': quality_report,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"数据质量评估失败: {e}")
            return {'error': str(e)}
    
    async def _handle_missing_values_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理缺失值任务"""
        try:
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            strategy = data.get('strategy', 'auto')
            
            # 分析缺失值模式
            missing_analysis = {
                'total_missing': df.isnull().sum().sum(),
                'missing_by_column': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
            }
            
            # 处理缺失值
            if missing_analysis['total_missing'] > 0:
                processed_df = self.data_preprocessor._handle_missing_values(df)
                
                result = {
                    'status': 'completed',
                    'strategy_used': strategy,
                    'missing_analysis': missing_analysis,
                    'processed_shape': processed_df.shape,
                    'remaining_missing': processed_df.isnull().sum().sum(),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                result = {
                    'status': 'no_action_needed',
                    'message': '数据中没有缺失值',
                    'missing_analysis': missing_analysis,
                    'timestamp': datetime.now().isoformat()
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"缺失值处理失败: {e}")
            return {'error': str(e)}
    
    async def _handle_outlier_detection_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理异常值检测任务"""
        try:
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            methods = data.get('methods', ['iqr'])
            
            # 选择数值列
            numeric_data = df.select_dtypes(include=['number'])
            if numeric_data.empty:
                return {
                    'status': 'no_numeric_data',
                    'message': '数据中没有数值列可用于异常检测'
                }
            
            outlier_results = {}
            
            for method in methods:
                if method == 'iqr':
                    outliers = self.data_preprocessor._detect_outliers_iqr(numeric_data)
                elif method == 'zscore':
                    outliers = self.data_preprocessor._detect_outliers_zscore(numeric_data)
                elif method == 'isolation_forest':
                    outliers = self.data_preprocessor._detect_outliers_isolation_forest(numeric_data)
                else:
                    continue
                
                outlier_results[method] = {
                    'outlier_count': outliers.sum(),
                    'outlier_percentage': (outliers.sum() / len(df)) * 100,
                    'outlier_indices': outliers[outliers].index.tolist()[:20]  # 限制返回的索引数量
                }
            
            return {
                'status': 'completed',
                'methods_used': methods,
                'outlier_results': outlier_results,
                'total_records': len(df),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"异常值检测失败: {e}")
            return {'error': str(e)}
    
    async def _handle_normalization_task(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据标准化任务"""
        try:
            df = await self._get_dataframe_from_request(data)
            if df is None:
                return {'error': '无法获取数据'}
            
            method = data.get('method', 'standard')
            
            # 选择数值列
            numeric_data = df.select_dtypes(include=['number'])
            if numeric_data.empty:
                return {
                    'status': 'no_numeric_data',
                    'message': '数据中没有数值列可用于标准化'
                }
            
            # 执行标准化
            normalized_df = df.copy()
            normalized_df = self.data_preprocessor._normalize_data(normalized_df)
            
            # 计算标准化统计信息
            normalization_stats = {
                'method': method,
                'columns_normalized': numeric_data.columns.tolist(),
                'original_stats': {
                    'mean': numeric_data.mean().to_dict(),
                    'std': numeric_data.std().to_dict(),
                    'min': numeric_data.min().to_dict(),
                    'max': numeric_data.max().to_dict()
                },
                'normalized_stats': {
                    'mean': normalized_df[numeric_data.columns].mean().to_dict(),
                    'std': normalized_df[numeric_data.columns].std().to_dict(),
                    'min': normalized_df[numeric_data.columns].min().to_dict(),
                    'max': normalized_df[numeric_data.columns].max().to_dict()
                }
            }
            
            return {
                'status': 'completed',
                'normalization_stats': normalization_stats,
                'sample_data': normalized_df.head(5).to_dict('records'),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"数据标准化失败: {e}")
            return {'error': str(e)}
    
    async def _get_dataframe_from_request(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """从请求中获取DataFrame"""
        if 'dataframe' in data:
            return pd.DataFrame(data['dataframe'])
        elif 'query' in data:
            return await self.data_connection.execute_query(data['query'])
        else:
            return await self.data_connection.get_data()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试数据连接
            db_healthy = await self.data_connection.test_connection()
            
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            return {
                'status': 'healthy' if self.is_running and db_healthy else 'unhealthy',
                'server_type': 'preprocessing',
                'port': self.port,
                'uptime': uptime,
                'processed_requests': self.processed_requests,
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
            'server_type': 'preprocessing',
            'port': self.port,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'processed_requests': self.processed_requests,
            'supported_tasks': [
                'preprocess',
                'data_quality',
                'missing_values',
                'outlier_detection',
                'normalization'
            ]
        }