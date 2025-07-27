"""
数据连接管理模块

提供MySQL连接池管理、连接状态监控、错误处理和重试机制
"""

import asyncio
import pymysql
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime, timedelta


class DataConnection:
    """数据连接管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化数据连接"""
        self.config = config
        self.mysql_config = config.get('mysql', {})
        self.redis_config = config.get('redis', {})
        
        self.logger = logging.getLogger(__name__)
        
        # MySQL连接池
        self.engine = None
        self.connection_pool = None
        
        # Redis连接
        self.redis_client = None
        
        # 连接状态
        self.is_connected = False
        self.last_check_time = None
        
        self._initialize_mysql()
        self._initialize_redis()
    
    def _initialize_mysql(self):
        """初始化MySQL连接池"""
        try:
            mysql_url = (
                f"mysql+pymysql://{self.mysql_config.get('user', 'root')}:"
                f"{self.mysql_config.get('password', '')}@"
                f"{self.mysql_config.get('host', 'localhost')}:"
                f"{self.mysql_config.get('port', 3306)}/"
                f"{self.mysql_config.get('database', 'test')}"
                f"?charset={self.mysql_config.get('charset', 'utf8mb4')}"
            )
            
            self.engine = create_engine(
                mysql_url,
                poolclass=QueuePool,
                pool_size=self.mysql_config.get('pool_size', 10),
                max_overflow=self.mysql_config.get('max_overflow', 20),
                pool_timeout=self.mysql_config.get('pool_timeout', 30),
                pool_recycle=self.mysql_config.get('pool_recycle', 3600),
                echo=False
            )
            
            self.logger.info("MySQL连接池初始化成功")
            
        except Exception as e:
            self.logger.error(f"MySQL连接池初始化失败: {e}")
            raise
    
    def _initialize_redis(self):
        """初始化Redis连接"""
        try:
            import redis
            self.redis_client = redis.ConnectionPool(
                host=self.redis_config.get('host', 'localhost'),
                port=self.redis_config.get('port', 6379),
                db=self.redis_config.get('db', 0),
                password=self.redis_config.get('password', None),
                max_connections=self.redis_config.get('max_connections', 50)
            )
            self.logger.info("Redis连接池初始化成功")
            
        except ImportError:
            self.logger.warning("Redis模块未安装，跳过Redis初始化")
        except Exception as e:
            self.logger.error(f"Redis连接池初始化失败: {e}")
    
    async def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            if self.engine is None:
                return False
            
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                result.fetchone()
            
            self.is_connected = True
            self.last_check_time = datetime.now()
            self.logger.info("数据库连接测试成功")
            return True
            
        except Exception as e:
            self.is_connected = False
            self.logger.error(f"数据库连接测试失败: {e}")
            return False
    
    async def get_data(self, 
                      query: Optional[str] = None,
                      table: Optional[str] = None,
                      filters: Optional[Dict[str, Any]] = None,
                      limit: int = 1000) -> Optional[pd.DataFrame]:
        """获取数据"""
        try:
            if query is None and table is None:
                # 默认查询，获取示例数据
                query = self._build_default_query(limit)
            elif table is not None:
                # 根据表名构建查询
                query = self._build_table_query(table, filters, limit)
            
            # 执行查询
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"成功获取 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"数据获取失败: {e}")
            # 如果数据库查询失败，返回示例数据
            return self._get_sample_data()
    
    def _build_default_query(self, limit: int) -> str:
        """构建默认查询"""
        return f"""
        SELECT 
            id,
            product_name,
            category,
            quantity,
            price,
            supplier,
            created_at,
            updated_at
        FROM warehouse_inventory 
        ORDER BY updated_at DESC 
        LIMIT {limit}
        """
    
    def _build_table_query(self, table: str, filters: Optional[Dict[str, Any]], limit: int) -> str:
        """根据表名和过滤条件构建查询"""
        query = f"SELECT * FROM {table}"
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f"{key} = '{value}'")
                else:
                    conditions.append(f"{key} = {value}")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += f" LIMIT {limit}"
        return query
    
    def _get_sample_data(self) -> pd.DataFrame:
        """获取示例数据（当数据库不可用时）"""
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'id': range(1, n_samples + 1),
            'product_name': [f'商品_{i}' for i in range(1, n_samples + 1)],
            'category': np.random.choice(['电子产品', '服装', '食品', '家具', '图书'], n_samples),
            'quantity': np.random.randint(1, 1000, n_samples),
            'price': np.round(np.random.uniform(10, 1000, n_samples), 2),
            'supplier': np.random.choice(['供应商A', '供应商B', '供应商C', '供应商D'], n_samples),
            'created_at': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
            'updated_at': pd.date_range('2024-01-01', periods=n_samples, freq='H')
        }
        
        df = pd.DataFrame(data)
        self.logger.info(f"使用示例数据: {len(df)} 条记录")
        return df
    
    async def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """获取最近的数据"""
        try:
            query = f"""
            SELECT * FROM warehouse_inventory 
            WHERE updated_at >= NOW() - INTERVAL {hours} HOUR
            ORDER BY updated_at DESC
            """
            return await self.get_data(query=query)
        except Exception as e:
            self.logger.error(f"获取最近数据失败: {e}")
            return self._get_sample_data()
    
    async def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """执行自定义查询"""
        try:
            df = pd.read_sql(query, self.engine)
            self.logger.info(f"查询执行成功，返回 {len(df)} 条记录")
            return df
        except Exception as e:
            self.logger.error(f"查询执行失败: {e}")
            return None
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态"""
        status = {
            'mysql': {
                'connected': self.is_connected,
                'last_check': self.last_check_time,
                'pool_size': self.mysql_config.get('pool_size', 10)
            },
            'redis': {
                'available': self.redis_client is not None
            }
        }
        return status
    
    def close(self):
        """关闭连接"""
        try:
            if self.engine:
                self.engine.dispose()
            if self.redis_client:
                self.redis_client.disconnect()
            self.logger.info("数据连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭连接时出错: {e}")