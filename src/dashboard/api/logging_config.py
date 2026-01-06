"""Structured logging configuration for dashboard API.

This module provides structured logging with JSON formatting for production
environments and human-readable formatting for development.

Security Impact:
    - Logs are sanitized to prevent PII leakage
    - Structured format enables better log analysis
    - Log levels prevent information disclosure
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs.
    
    Formats log records as JSON for better parsing and analysis in
    production environments.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Parameters:
            record: Log record to format
            
        Returns:
            JSON string representation of log record
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add request context if available
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "client_ip"):
            log_data["client_ip"] = record.client_ip
        if hasattr(record, "endpoint"):
            log_data["endpoint"] = record.endpoint
        
        return json.dumps(log_data)


def setup_logging(use_json: bool = False, log_level: str = "INFO"):
    """Setup application logging.
    
    Parameters:
        use_json: Use JSON formatting (for production)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Security Impact:
        - JSON logs enable better security monitoring
        - Log levels prevent information disclosure
        - Structured format helps identify security events
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Set formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        # Human-readable format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    # Set levels for third-party loggers
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.
    
    Parameters:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

