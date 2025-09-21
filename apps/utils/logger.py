"""Logging utilities for Balatro detection system."""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from config.settings import settings


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Log file path. If None, uses settings default.
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Set level
    level = getattr(logging, settings.logging_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(settings.logging_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file is None:
        log_file = settings.logging_file
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=settings.logging_max_bytes,
            backupCount=settings.logging_backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)
