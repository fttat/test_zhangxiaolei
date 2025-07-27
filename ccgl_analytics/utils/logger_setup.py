"""
日志设置工具模块
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: str = "INFO",
                format_string: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 记录器名称
        log_file: 日志文件路径
        level: 日志级别
        format_string: 日志格式
        
    Returns:
        配置好的logger对象
    """
    
    # 创建logger
    logger = logging.getLogger(name)
    
    # 防止重复添加handler
    if logger.handlers:
        return logger
    
    # 设置日志级别
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # 设置日志格式
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用旋转文件处理器
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger