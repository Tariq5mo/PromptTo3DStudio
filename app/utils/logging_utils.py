import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List

def setup_logging(
    log_level: int = logging.INFO,
    # Use a slightly cleaner format
    log_format: str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    log_file: Optional[str] = None,
):
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (default: INFO)
        log_format: Format string for log messages
        log_file: Path to log file (if None, logs will be in data/app.log)
    """
    # Create handlers
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    # Add file handler if requested
    if log_file is None:
        # Default log location
        log_dir = Path("data")
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "app.log")

    file_handler = logging.FileHandler(log_file)
    handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format, # Use the updated format
        handlers=handlers
    )

    logging.info(f"Logging configured at level {logging.getLevelName(log_level)} with output to {log_file}")

def log_execution_start(prompt: str) -> str:
    """
    Log the start of a pipeline execution and return a correlation ID.

    Args:
        prompt: User prompt starting the execution

    Returns:
        A correlation ID for tracking the execution across logs
    """
    correlation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"[{correlation_id}] Starting pipeline execution for prompt: '{prompt}'")
    return correlation_id

def log_execution_end(correlation_id: str, success: bool, execution_time: float) -> None:
    """
    Log the end of a pipeline execution.

    Args:
        correlation_id: The correlation ID from log_execution_start
        success: Whether the execution was successful
        execution_time: Execution time in seconds
    """
    status = "SUCCESS" if success else "FAILED"
    logging.info(f"[{correlation_id}] Pipeline execution {status} in {execution_time:.2f} seconds")