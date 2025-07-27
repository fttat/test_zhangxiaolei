"""
日志工具模块

提供统一的日志配置和管理功能
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logger(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """设置日志系统"""
    
    # 默认配置
    default_config = {
        'level': 'INFO',
        'file': 'logs/ccgl.log',
        'max_size': '100MB',
        'backup_count': 5,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
    
    # 合并配置
    if config:
        default_config.update(config)
    
    # 创建根日志器
    logger = logging.getLogger('ccgl')
    
    # 如果已经配置过，直接返回
    if logger.handlers:
        return logger
    
    # 设置日志级别
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_mapping.get(default_config['level'].upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 创建格式化器
    formatter = logging.Formatter(default_config['format'])
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    try:
        log_file = Path(default_config['file'])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 解析文件大小
        max_size = _parse_size(default_config['max_size'])
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size,
            backupCount=default_config['backup_count'],
            encoding='utf-8'
        )
        
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    except Exception as e:
        logger.warning(f"文件日志配置失败: {e}")
    
    # 防止日志重复
    logger.propagate = False
    
    logger.info("日志系统初始化完成")
    return logger


def _parse_size(size_str: str) -> int:
    """解析文件大小字符串"""
    size_str = size_str.upper().strip()
    
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        # 默认为字节
        return int(size_str)


class StructuredLogger:
    """结构化日志器"""
    
    def __init__(self, logger_name: str, config: Optional[Dict[str, Any]] = None):
        """初始化结构化日志器"""
        self.logger = setup_logger(config)
        self.logger_name = logger_name
    
    def log_event(self, level: str, event_type: str, 
                  message: str, **kwargs):
        """记录结构化事件"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'logger': self.logger_name,
            'event_type': event_type,
            'message': message,
            **kwargs
        }
        
        # 格式化日志消息
        formatted_message = f"[{event_type}] {message}"
        if kwargs:
            formatted_message += f" | Data: {kwargs}"
        
        # 记录日志
        getattr(self.logger, level.lower())(formatted_message)
    
    def log_performance(self, operation: str, duration: float, 
                       success: bool = True, **kwargs):
        """记录性能指标"""
        self.log_event(
            'info' if success else 'warning',
            'performance',
            f"Operation '{operation}' completed in {duration:.3f}s",
            operation=operation,
            duration=duration,
            success=success,
            **kwargs
        )
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """记录错误信息"""
        self.log_event(
            'error',
            'error',
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    def log_api_call(self, method: str, url: str, 
                     status_code: Optional[int] = None,
                     duration: Optional[float] = None,
                     **kwargs):
        """记录API调用"""
        message = f"API call: {method} {url}"
        if status_code:
            message += f" -> {status_code}"
        if duration:
            message += f" ({duration:.3f}s)"
        
        level = 'info'
        if status_code and status_code >= 400:
            level = 'warning' if status_code < 500 else 'error'
        
        self.log_event(
            level,
            'api_call',
            message,
            method=method,
            url=url,
            status_code=status_code,
            duration=duration,
            **kwargs
        )


class AnalyticsLogger(StructuredLogger):
    """分析专用日志器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('analytics', config)
    
    def log_data_processing(self, operation: str, 
                           input_shape: tuple, output_shape: tuple,
                           duration: float, **kwargs):
        """记录数据处理操作"""
        self.log_event(
            'info',
            'data_processing',
            f"Data processing '{operation}': {input_shape} -> {output_shape}",
            operation=operation,
            input_shape=input_shape,
            output_shape=output_shape,
            duration=duration,
            **kwargs
        )
    
    def log_model_training(self, model_name: str, 
                          training_time: float,
                          metrics: Dict[str, float],
                          **kwargs):
        """记录模型训练"""
        self.log_event(
            'info',
            'model_training',
            f"Model '{model_name}' trained successfully",
            model_name=model_name,
            training_time=training_time,
            metrics=metrics,
            **kwargs
        )
    
    def log_analysis_result(self, analysis_type: str,
                           result_summary: Dict[str, Any],
                           **kwargs):
        """记录分析结果"""
        self.log_event(
            'info',
            'analysis_result',
            f"Analysis '{analysis_type}' completed",
            analysis_type=analysis_type,
            result_summary=result_summary,
            **kwargs
        )


class MCPLogger(StructuredLogger):
    """MCP专用日志器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('mcp', config)
    
    def log_server_start(self, server_name: str, port: int, **kwargs):
        """记录服务器启动"""
        self.log_event(
            'info',
            'server_start',
            f"MCP server '{server_name}' started on port {port}",
            server_name=server_name,
            port=port,
            **kwargs
        )
    
    def log_server_stop(self, server_name: str, **kwargs):
        """记录服务器停止"""
        self.log_event(
            'info',
            'server_stop',
            f"MCP server '{server_name}' stopped",
            server_name=server_name,
            **kwargs
        )
    
    def log_client_request(self, server_name: str, method: str,
                          client_id: str, duration: Optional[float] = None,
                          **kwargs):
        """记录客户端请求"""
        message = f"MCP request to '{server_name}': {method}"
        if duration:
            message += f" ({duration:.3f}s)"
        
        self.log_event(
            'info',
            'client_request',
            message,
            server_name=server_name,
            method=method,
            client_id=client_id,
            duration=duration,
            **kwargs
        )
    
    def log_coordination_task(self, task_name: str, 
                             servers_involved: list,
                             status: str, **kwargs):
        """记录协调任务"""
        self.log_event(
            'info',
            'coordination_task',
            f"Coordination task '{task_name}' {status}",
            task_name=task_name,
            servers_involved=servers_involved,
            status=status,
            **kwargs
        )


def get_logger(name: str) -> logging.Logger:
    """获取特定名称的日志器"""
    return logging.getLogger(f'ccgl.{name}')


def set_log_level(level: str):
    """设置全局日志级别"""
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_mapping.get(level.upper(), logging.INFO)
    logging.getLogger('ccgl').setLevel(log_level)


# 预定义的日志器实例
analytics_logger = AnalyticsLogger()
mcp_logger = MCPLogger()