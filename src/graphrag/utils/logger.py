"""
Logging utilities for GraphRAG system.
"""

import logging
import sys
from typing import Optional
from datetime import datetime


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Create and configure a logger instance.
    
    Args:
        name: Logger name
        log_file: Optional file to write logs to
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class ExperimentLogger:
    """Logger for tracking experimental runs."""
    
    def __init__(self, experiment_name: str, log_dir: str = "results"):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir
        self.logger = get_logger(
            f"experiment_{experiment_name}",
            log_file=f"{log_dir}/experiment_{experiment_name}_{self.timestamp}.log"
        )
    
    def log_config(self, config: dict):
        """Log experiment configuration."""
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  - {key}: {value}")
    
    def log_metrics(self, metrics: dict):
        """Log computed metrics."""
        self.logger.info("Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  - {key}: {value}")
    
    def log_error(self, error: Exception):
        """Log error with traceback."""
        self.logger.exception(f"Error occurred: {str(error)}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger."""
        return self.logger
