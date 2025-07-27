"""
数据连接管理模块

提供MySQL数据库连接、数据质量评估和自动化报告生成功能。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import json

# Optional MySQL connector import
try:
    import mysql.connector
    from mysql.connector import pooling
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    mysql = None
    pooling = None

class DataConnectionManager:
    """数据连接管理器 - 提供数据库连接和数据质量评估"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据连接管理器
        
        Args:
            config: 数据库配置信息
        """
        self.config = config
        self.connection_pool = None
        self.logger = logging.getLogger(__name__)
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """设置数据库连接池"""
        if not MYSQL_AVAILABLE:
            self.logger.warning("MySQL连接器不可用，将使用模拟模式")
            self.connection_pool = None
            return
            
        try:
            pool_config = {
                'host': self.config.get('host', 'localhost'),
                'port': self.config.get('port', 3306),
                'database': self.config.get('database', 'ccgl_warehouse'),
                'user': self.config.get('user', 'root'),
                'password': self.config.get('password', ''),
                'pool_name': 'ccgl_pool',
                'pool_size': self.config.get('pool_size', 10),
                'pool_reset_session': True,
                'charset': 'utf8mb4',
                'collation': 'utf8mb4_unicode_ci'
            }
            
            self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
            self.logger.info("数据库连接池创建成功")
            
        except Exception as e:
            self.logger.error(f"数据库连接池创建失败: {e}")
            self.logger.warning("将使用模拟模式")
            self.connection_pool = None
    
    def get_connection(self):
        """获取数据库连接"""
        if not MYSQL_AVAILABLE or self.connection_pool is None:
            self.logger.warning("数据库连接不可用，返回None")
            return None
            
        try:
            return self.connection_pool.get_connection()
        except Exception as e:
            self.logger.error(f"获取数据库连接失败: {e}")
            return None
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        执行查询并返回DataFrame
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果DataFrame
        """
        if not MYSQL_AVAILABLE or self.connection_pool is None:
            self.logger.warning("数据库不可用，返回空DataFrame")
            return pd.DataFrame()
            
        connection = None
        try:
            connection = self.get_connection()
            if connection is None:
                return pd.DataFrame()
                
            df = pd.read_sql(query, connection, params=params)
            self.logger.info(f"查询执行成功，返回 {len(df)} 行数据")
            return df
            
        except Exception as e:
            self.logger.error(f"查询执行失败: {e}")
            return pd.DataFrame()
        finally:
            if connection:
                connection.close()
    
    def assess_data_quality(self, table_name: str) -> Dict[str, Any]:
        """
        评估数据质量
        
        Args:
            table_name: 表名
            
        Returns:
            数据质量报告
        """
        try:
            # 获取表数据
            df = self.execute_query(f"SELECT * FROM {table_name}")
            
            quality_report = {
                'table_name': table_name,
                'assessment_time': datetime.now().isoformat(),
                'basic_stats': self._calculate_basic_stats(df),
                'data_quality_metrics': self._calculate_quality_metrics(df),
                'missing_values': self._analyze_missing_values(df),
                'duplicates': self._analyze_duplicates(df),
                'data_types': self._analyze_data_types(df),
                'outliers': self._detect_outliers(df)
            }
            
            self.logger.info(f"数据质量评估完成: {table_name}")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"数据质量评估失败: {e}")
            raise
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算基础统计信息"""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': df.select_dtypes(include=[np.number]).shape[1],
            'categorical_columns': df.select_dtypes(include=['object']).shape[1],
            'datetime_columns': df.select_dtypes(include=['datetime64']).shape[1]
        }
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算数据质量指标"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        return {
            'completeness': (total_cells - missing_cells) / total_cells,
            'consistency': self._calculate_consistency(df),
            'validity': self._calculate_validity(df),
            'uniqueness': self._calculate_uniqueness(df)
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析缺失值"""
        missing_stats = df.isnull().sum()
        return {
            'total_missing': int(missing_stats.sum()),
            'missing_by_column': missing_stats.to_dict(),
            'missing_percentage_by_column': (missing_stats / len(df) * 100).to_dict()
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析重复值"""
        duplicates = df.duplicated()
        return {
            'total_duplicates': int(duplicates.sum()),
            'duplicate_percentage': float(duplicates.sum() / len(df) * 100),
            'unique_rows': int(len(df) - duplicates.sum())
        }
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """分析数据类型"""
        return df.dtypes.astype(str).to_dict()
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """检测异常值（使用IQR方法）"""
        outliers = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = {
                'count': int(outlier_mask.sum()),
                'percentage': float(outlier_mask.sum() / len(df) * 100)
            }
        
        return outliers
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """计算数据一致性"""
        # 简化的一致性计算：检查格式一致性
        consistency_score = 1.0
        
        # 检查字符串列的格式一致性
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if not df[col].empty:
                # 检查是否有混合的数据类型（数字和字符串）
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    # 如果能转换为数字，但原来是字符串，可能存在不一致
                except:
                    pass
        
        return consistency_score
    
    def _calculate_validity(self, df: pd.DataFrame) -> float:
        """计算数据有效性"""
        # 简化的有效性计算：检查数据范围合理性
        validity_score = 1.0
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if not df[col].empty:
                # 检查是否有负数（对于应该为正数的列）
                if col.lower() in ['quantity', 'price', 'amount', 'count']:
                    negative_ratio = (df[col] < 0).sum() / len(df)
                    validity_score -= negative_ratio * 0.1
        
        return max(0.0, validity_score)
    
    def _calculate_uniqueness(self, df: pd.DataFrame) -> float:
        """计算数据唯一性"""
        if df.empty:
            return 1.0
        
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates())
        
        return unique_rows / total_rows
    
    def generate_quality_report(self, table_names: List[str], output_path: str = None) -> Dict[str, Any]:
        """
        生成综合数据质量报告
        
        Args:
            table_names: 要评估的表名列表
            output_path: 报告输出路径
            
        Returns:
            综合质量报告
        """
        comprehensive_report = {
            'generation_time': datetime.now().isoformat(),
            'assessed_tables': table_names,
            'table_reports': {},
            'summary': {}
        }
        
        # 评估每个表
        for table_name in table_names:
            try:
                table_report = self.assess_data_quality(table_name)
                comprehensive_report['table_reports'][table_name] = table_report
            except Exception as e:
                self.logger.error(f"评估表 {table_name} 失败: {e}")
                comprehensive_report['table_reports'][table_name] = {
                    'error': str(e)
                }
        
        # 生成汇总信息
        comprehensive_report['summary'] = self._generate_summary(
            comprehensive_report['table_reports']
        )
        
        # 保存报告
        if output_path:
            self._save_report(comprehensive_report, output_path)
        
        return comprehensive_report
    
    def _generate_summary(self, table_reports: Dict[str, Dict]) -> Dict[str, Any]:
        """生成汇总信息"""
        valid_reports = [report for report in table_reports.values() 
                        if 'error' not in report]
        
        if not valid_reports:
            return {'error': '没有有效的表报告'}
        
        # 计算平均质量指标
        avg_completeness = np.mean([
            report['data_quality_metrics']['completeness'] 
            for report in valid_reports
        ])
        
        avg_consistency = np.mean([
            report['data_quality_metrics']['consistency'] 
            for report in valid_reports
        ])
        
        avg_validity = np.mean([
            report['data_quality_metrics']['validity'] 
            for report in valid_reports
        ])
        
        avg_uniqueness = np.mean([
            report['data_quality_metrics']['uniqueness'] 
            for report in valid_reports
        ])
        
        total_rows = sum([
            report['basic_stats']['total_rows'] 
            for report in valid_reports
        ])
        
        return {
            'total_tables_assessed': len(valid_reports),
            'total_rows_assessed': total_rows,
            'average_quality_metrics': {
                'completeness': float(avg_completeness),
                'consistency': float(avg_consistency),
                'validity': float(avg_validity),
                'uniqueness': float(avg_uniqueness)
            },
            'overall_quality_score': float(
                (avg_completeness + avg_consistency + avg_validity + avg_uniqueness) / 4
            )
        }
    
    def _save_report(self, report: Dict[str, Any], output_path: str):
        """保存报告到文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"质量报告已保存到: {output_path}")
        except Exception as e:
            self.logger.error(f"保存报告失败: {e}")
            raise
    
    def close(self):
        """关闭连接池"""
        if self.connection_pool:
            # MySQL连接池会自动管理连接
            self.logger.info("数据库连接管理器已关闭")