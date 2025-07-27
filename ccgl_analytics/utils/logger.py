"""
CCGL Analytics Logging Utility
Provides structured logging capabilities for the entire system
"""

import logging
import logging.handlers
import json
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated', 
                          'thread', 'threadName', 'processName', 'process',
                          'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
                
        return json.dumps(log_entry, ensure_ascii=False)

class TextFormatter(logging.Formatter):
    """Custom text formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'color') and record.color:
            color = self.COLORS.get(record.levelname, '')
            reset = self.RESET
        else:
            color = reset = ''
            
        record.color_start = color
        record.color_end = reset
        
        return super().format(record)

def setup_logging(
    level: str = "INFO",
    format_type: str = "text",
    log_file: Optional[str] = None,
    max_size: str = "100MB",
    backup_count: int = 5
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type (text, json)
        log_file: Path to log file
        max_size: Maximum log file size
        backup_count: Number of backup files to keep
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if format_type.lower() == "json":
        console_formatter = JsonFormatter()
    else:
        console_formatter = TextFormatter(
            '%(color_start)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(color_end)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Parse max_size
        size_str = max_size.upper()
        if size_str.endswith('MB'):
            max_bytes = int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            max_bytes = int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            max_bytes = int(size_str)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        
        if format_type.lower() == "json":
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str, extra_fields: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get a logger instance with optional extra fields.
    
    Args:
        name: Logger name (typically __name__)
        extra_fields: Extra fields to include in log records
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    if extra_fields:
        # Create a custom logger adapter to include extra fields
        class ExtraFieldsAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add extra fields to kwargs
                if 'extra' not in kwargs:
                    kwargs['extra'] = {}
                kwargs['extra'].update(self.extra)
                return msg, kwargs
        
        return ExtraFieldsAdapter(logger, extra_fields)
    
    return logger

class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self):
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(
                f"{self.__class__.__module__}.{self.__class__.__name__}"
            )
        return self._logger

# Module-level logger
logger = get_logger(__name__)

# Initialize logging from environment variables if available
def init_from_env():
    """Initialize logging from environment variables."""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_format = os.getenv('LOG_FORMAT', 'text')
    log_file = os.getenv('LOG_FILE')
    max_size = os.getenv('LOG_MAX_SIZE', '100MB')
    backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))
    
    setup_logging(
        level=log_level,
        format_type=log_format,
        log_file=log_file,
        max_size=max_size,
        backup_count=backup_count
    )

# Initialize logging on module import
init_from_env()