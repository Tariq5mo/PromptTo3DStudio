import unittest
import time
from unittest.mock import patch, MagicMock

# Import from test config
from tests.test_config import TEST_PROMPT

# Import error handling utilities to test
from utils.error_handling import (
    ExecutionContext,
    safe_execute,
    retry
)

class TestExecutionContext(unittest.TestCase):
    """Test the ExecutionContext class for tracking pipeline execution"""

    def test_initialization(self):
        """Test that ExecutionContext initializes correctly"""
        context = ExecutionContext(TEST_PROMPT)

        self.assertEqual(context.user_prompt, TEST_PROMPT)
        self.assertEqual(context.stage, "initialized")
        self.assertIsNotNone(context.correlation_id)
        self.assertIsNone(context.error)
        self.assertIsNone(context.enhanced_prompt)
        self.assertIsNone(context.image_path)
        self.assertIsNone(context.model_path)

    def test_update_stage(self):
        """Test updating the execution stage"""
        context = ExecutionContext(TEST_PROMPT)

        context.update_stage("processing_prompt")
        self.assertEqual(context.stage, "processing_prompt")

        context.update_stage("generating_image")
        self.assertEqual(context.stage, "generating_image")

    def test_set_error(self):
        """Test setting an error in the context"""
        context = ExecutionContext(TEST_PROMPT)
        test_error = ValueError("Test error")

        context.set_error(test_error)
        self.assertEqual(context.error, test_error)

    def test_get_execution_time(self):
        """Test getting execution time"""
        context = ExecutionContext(TEST_PROMPT)

        # Sleep a bit to get a non-zero execution time
        time.sleep(0.01)

        execution_time = context.get_execution_time()
        self.assertGreater(execution_time, 0)

    def test_get_status_summary(self):
        """Test getting status summary"""
        context = ExecutionContext(TEST_PROMPT)
        context.enhanced_prompt = "Enhanced " + TEST_PROMPT
        context.image_path = "/path/to/image.png"
        context.model_path = "/path/to/model.glb"

        summary = context.get_status_summary()

        self.assertEqual(summary["user_prompt"], TEST_PROMPT)
        self.assertEqual(summary["enhanced_prompt"], "Enhanced " + TEST_PROMPT)
        self.assertEqual(summary["image_path"], "/path/to/image.png")
        self.assertEqual(summary["model_path"], "/path/to/model.glb")


class TestSafeExecuteDecorator(unittest.TestCase):
    """Test the safe_execute decorator"""

    def test_normal_execution(self):
        """Test normal function execution without errors"""
        # Define a test function
        @safe_execute()
        def test_function():
            return "success"

        self.assertEqual(test_function(), "success")

    def test_error_handling(self):
        """Test catching exceptions"""
        # Define a function that throws an exception
        @safe_execute(default_return="fallback")
        def failing_function():
            raise ValueError("Test error")

        self.assertEqual(failing_function(), "fallback")

    def test_custom_default_return(self):
        """Test custom default return value"""
        @safe_execute(default_return={"status": "error"})
        def failing_function():
            raise ValueError("Test error")

        self.assertEqual(failing_function(), {"status": "error"})


class TestRetryDecorator(unittest.TestCase):
    """Test the retry decorator"""

    def test_successful_first_attempt(self):
        """Test function that succeeds on first attempt"""
        mock_function = MagicMock(return_value="success")
        decorated = retry()(mock_function)

        result = decorated()

        self.assertEqual(result, "success")
        mock_function.assert_called_once()

    def test_retry_until_success(self):
        """Test function that fails then succeeds"""
        # Mock that fails twice then succeeds
        mock_function = MagicMock(side_effect=[
            ValueError("First failure"),
            ValueError("Second failure"),
            "success"
        ])

        # Use fast retries for testing
        decorated = retry(max_attempts=3, delay=0.01)(mock_function)

        # Don't actually sleep in tests
        with patch('time.sleep'):
            result = decorated()

        self.assertEqual(result, "success")
        self.assertEqual(mock_function.call_count, 3)

    def test_all_attempts_fail(self):
        """Test when all retry attempts fail"""
        mock_function = MagicMock(side_effect=ValueError("Persistent error"))

        # Use fast retries for testing
        decorated = retry(max_attempts=3, delay=0.01)(mock_function)

        # Don't actually sleep in tests
        with patch('time.sleep'):
            with self.assertRaises(ValueError):
                decorated()

        self.assertEqual(mock_function.call_count, 3)


if __name__ == "__main__":
    unittest.main()