"""
配置加载器工具模块
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_name: str = "config.yml") -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_name: 配置文件名
            
        Returns:
            配置字典
        """
        try:
            config_path = self.config_dir / config_name
            
            if not config_path.exists():
                self.logger.warning(f"配置文件不存在: {config_path}")
                return self._get_default_config()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 合并环境变量
            config = self._merge_env_vars(config)
            
            self.logger.info(f"配置加载成功: {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'database': {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'database': os.getenv('DB_NAME', 'ccgl_warehouse'),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', 'password'),
                'pool_size': 10
            },
            'preprocessing': {
                'missing_strategy': 'auto',
                'outlier_method': 'iqr',
                'scaling_method': 'standard'
            },
            'analysis': {
                'clustering_methods': ['kmeans', 'dbscan', 'hierarchical'],
                'anomaly_methods': ['isolation_forest', 'one_class_svm'],
                'reduction_methods': ['pca', 'tsne']
            },
            'mcp': {
                'server_host': os.getenv('MCP_SERVER_HOST', 'localhost'),
                'server_port': int(os.getenv('MCP_SERVER_PORT', 8080)),
                'enable_ssl': os.getenv('MCP_ENABLE_SSL', 'false').lower() == 'true'
            },
            'llm': {
                'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
                'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY', ''),
                'default_model': 'gpt-3.5-turbo',
                'max_tokens': 4000
            },
            'web': {
                'host': os.getenv('WEB_HOST', '0.0.0.0'),
                'port': int(os.getenv('WEB_PORT', 8000)),
                'debug': os.getenv('WEB_DEBUG', 'false').lower() == 'true'
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': os.getenv('LOG_FILE', 'logs/ccgl_analytics.log'),
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    def _merge_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """合并环境变量到配置中"""
        env_mappings = {
            'DB_HOST': ['database', 'host'],
            'DB_PORT': ['database', 'port'],
            'DB_NAME': ['database', 'database'],
            'DB_USER': ['database', 'user'],
            'DB_PASSWORD': ['database', 'password'],
            'MCP_SERVER_HOST': ['mcp', 'server_host'],
            'MCP_SERVER_PORT': ['mcp', 'server_port'],
            'OPENAI_API_KEY': ['llm', 'openai_api_key'],
            'ANTHROPIC_API_KEY': ['llm', 'anthropic_api_key'],
            'WEB_HOST': ['web', 'host'],
            'WEB_PORT': ['web', 'port'],
            'LOG_LEVEL': ['logging', 'level']
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # 导航到配置路径
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # 设置值（尝试转换类型）
                try:
                    if config_path[-1].endswith('_port'):
                        current[config_path[-1]] = int(env_value)
                    elif env_value.lower() in ['true', 'false']:
                        current[config_path[-1]] = env_value.lower() == 'true'
                    else:
                        current[config_path[-1]] = env_value
                except ValueError:
                    current[config_path[-1]] = env_value
        
        return config