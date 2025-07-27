"""
Logging utility for CCGL Analytics platform
Provides structured logging with multiple output formats and levels
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Try to import loguru for enhanced logging, fallback to standard logging
try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False


class CCGLLogger:
    """Enhanced logger for CCGL Analytics platform"""
    
    def __init__(self, name: str = "ccgl_analytics", level: str = "INFO"):
        self.name = name
        self.level = level.upper()
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with appropriate configuration"""
        if LOGURU_AVAILABLE:
            return self._setup_loguru()
        else:
            return self._setup_standard_logger()
    
    def _setup_loguru(self):
        """Setup loguru-based logging"""
        # Remove default handler
        loguru_logger.remove()
        
        # Add console handler
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.level,
            colorize=True
        )
        
        # Add file handler
        log_file = Path("logs") / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        loguru_logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=self.level,
            rotation="1 day",
            retention="7 days",
            compression="zip"
        )
        
        return loguru_logger
    
    def _setup_standard_logger(self):
        """Setup standard Python logging"""
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.level))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = Path("logs") / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        if LOGURU_AVAILABLE:
            self.logger.debug(message, **kwargs)
        else:
            self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        if LOGURU_AVAILABLE:
            self.logger.info(message, **kwargs)
        else:
            self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        if LOGURU_AVAILABLE:
            self.logger.warning(message, **kwargs)
        else:
            self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        if LOGURU_AVAILABLE:
            self.logger.error(message, **kwargs)
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        if LOGURU_AVAILABLE:
            self.logger.critical(message, **kwargs)
        else:
            self.logger.critical(message, extra=kwargs)


# Global logger instance
_default_logger: Optional[CCGLLogger] = None


def get_logger(name: str = "ccgl_analytics", level: str = "INFO") -> CCGLLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        CCGLLogger instance
    """
    global _default_logger
    
    if _default_logger is None or _default_logger.name != name:
        _default_logger = CCGLLogger(name, level)
    
    return _default_logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup global logging configuration
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    global _default_logger
    _default_logger = CCGLLogger("ccgl_analytics", level)
    
    if log_file and LOGURU_AVAILABLE:
        _default_logger.logger.add(log_file, level=level)