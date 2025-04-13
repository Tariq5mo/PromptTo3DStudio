import logging
import functools
import time
import uuid
from typing import Callable, Any, TypeVar, cast, Dict, Optional

T = TypeVar('T')

class ExecutionContext:
    """
    A context object for tracking pipeline execution state and generated assets.
    """
    def __init__(self, user_prompt: str, correlation_id: Optional[str] = None):
        """
        Initialize a new execution context.

        Args:
            user_prompt: Original user prompt
            correlation_id: Optional correlation ID for tracing (generated if not provided)
        """
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.user_prompt = user_prompt
        self.enhanced_prompt: Optional[str] = None
        self.image_path: Optional[str] = None
        self.image_filename: Optional[str] = None
        self.model_path: Optional[str] = None
        self.model_filename: Optional[str] = None
        self.start_time = time.time()
        self.stage = "initialized"
        self.error: Optional[Exception] = None

        # Set up a correlation ID for tracing the entire request
        logging.info(f"[{self.correlation_id}] Created execution context for prompt: '{user_prompt}'")

    def update_stage(self, stage: str) -> None:
        """Update the current pipeline stage"""
        self.stage = stage
        logging.info(f"[{self.correlation_id}] Execution stage: {stage}")

    def set_error(self, error: Exception) -> None:
        """Set error information"""
        self.error = error
        logging.error(f"[{self.correlation_id}] Error in stage '{self.stage}': {str(error)}")

    def get_execution_time(self) -> float:
        """Get the execution time so far in seconds"""
        return time.time() - self.start_time

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a dictionary summarizing the execution status"""
        return {
            "correlation_id": self.correlation_id,
            "user_prompt": self.user_prompt,
            "enhanced_prompt": self.enhanced_prompt,
            "stage": self.stage,
            "execution_time": f"{self.get_execution_time():.2f}s",
            "error": str(self.error) if self.error else None,
            "image_path": self.image_path,
            "model_path": self.model_path
        }


def safe_execute(default_return: Any = None):
    """
    Decorator that catches and logs exceptions.

    Args:
        default_return: Value to return if an exception occurs

    Returns:
        Decorated function that handles exceptions
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                return cast(T, default_return)
        return wrapper
    return decorator


def retry(max_attempts: int = 3, delay: float = 2.0, backoff: float = 2.0):
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Base delay between retries in seconds
        backoff: Multiplier for the delay on each subsequent retry

    Returns:
        Decorated function that implements retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:  # Don't sleep on the last attempt
                        logging.warning(
                            f"Attempt {attempt+1}/{max_attempts} for {func.__name__} "
                            f"failed: {str(e)}. Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logging.error(
                            f"Final attempt {max_attempts}/{max_attempts} for {func.__name__} "
                            f"failed: {str(e)}"
                        )

            # If we get here, all retries failed
            raise last_exception  # type: ignore

        return wrapper
    return decorator