"""
Centralized logging configuration for the Trading Backtest System.

This module provides a unified logging setup with dual output (console and file),
configurable log levels, and rotating file handlers to manage log file sizes.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
from datetime import datetime


# Default configuration
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "trader_backtest.log"
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# Logger name hierarchy
ROOT_LOGGER_NAME = "trader_backtest"


class LoggerSetup:
    """
    Manages the logging configuration for the entire application.
    
    This class provides a centralized way to configure logging with both
    console and file handlers, supporting different log levels for each output.
    """
    
    _initialized = False
    
    @classmethod
    def setup_logging(
        cls,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        enable_console: bool = True,
        enable_file: bool = True,
    ) -> logging.Logger:
        """
        Configure the logging system with console and file handlers.
        
        Parameters
        ----------
        console_level : str
            Log level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_level : str
            Log level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file : str, optional
            Name of the log file (default: trader_backtest.log)
        log_dir : str, optional
            Directory for log files (default: logs/)
        max_bytes : int
            Maximum size of log file before rotation (default: 10MB)
        backup_count : int
            Number of backup log files to keep (default: 5)
        enable_console : bool
            Enable console logging (default: True)
        enable_file : bool
            Enable file logging (default: True)
            
        Returns
        -------
        logging.Logger
            The configured root logger
        """
        # Always allow reconfiguration - clear handlers and reinitialize
        # This ensures all parameters (including enable_console, enable_file) are respected
        if cls._initialized:
            root_logger = logging.getLogger(ROOT_LOGGER_NAME)
            root_logger.handlers.clear()
            cls._initialized = False
        
        # Get or create root logger
        root_logger = logging.getLogger(ROOT_LOGGER_NAME)
        root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter
        
        # Remove any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(cls._get_log_level(console_level))
            console_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if enable_file:
            # Create log directory if it doesn't exist
            log_directory = Path(log_dir or DEFAULT_LOG_DIR)
            log_directory.mkdir(parents=True, exist_ok=True)
            
            # Set up log file path
            log_filename = log_file or DEFAULT_LOG_FILE
            log_path = log_directory / log_filename
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_path),
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(cls._get_log_level(file_level))
            file_formatter = logging.Formatter(
                fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-35s | %(funcName)-25s:%(lineno)-4d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            # Log the initialization
            root_logger.info(f"Logging initialized - File: {log_path}, Console: {enable_console}")
        
        cls._initialized = True
        return root_logger
    
    @staticmethod
    def _get_log_level(level_str: str) -> int:
        """
        Convert string log level to logging constant.
        
        Parameters
        ----------
        level_str : str
            Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            
        Returns
        -------
        int
            Logging level constant
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        return level_map.get(level_str.upper(), logging.INFO)
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.
        
        This creates a hierarchical logger under the root logger, allowing
        for component-specific logging configuration if needed.
        
        Parameters
        ----------
        name : str
            Name of the component (e.g., 'markets', 'brokers', 'traders')
            
        Returns
        -------
        logging.Logger
            Logger instance for the component
        """
        if not cls._initialized:
            cls.setup_logging()
        
        # Create hierarchical logger name
        if name.startswith(ROOT_LOGGER_NAME):
            logger_name = name
        else:
            logger_name = f"{ROOT_LOGGER_NAME}.{name}"
        
        return logging.getLogger(logger_name)
    
    @classmethod
    def reset(cls):
        """Reset the logging configuration (useful for testing)."""
        root_logger = logging.getLogger(ROOT_LOGGER_NAME)
        root_logger.handlers.clear()
        cls._initialized = False


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger for a component.
    
    Parameters
    ----------
    name : str
        Name of the component
        
    Returns
    -------
    logging.Logger
        Logger instance
    """
    return LoggerSetup.get_logger(name)


def setup_logging(**kwargs) -> logging.Logger:
    """
    Convenience function to set up logging.
    
    Parameters
    ----------
    **kwargs
        Keyword arguments passed to LoggerSetup.setup_logging()
        
    Returns
    -------
    logging.Logger
        The configured root logger
    """
    return LoggerSetup.setup_logging(**kwargs)

# Made with Bob
