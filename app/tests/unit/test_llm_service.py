import unittest
from unittest.mock import patch, MagicMock
import json

# Import from test config
from tests.test_config import TEST_PROMPT, TEST_ENHANCED_PROMPT

# Import the service to test
from services.llm_service import LLMService

class TestLLMService(unittest.TestCase):
    """Unit tests for the LLM service that interacts with Ollama"""

    def setUp(self):
        """Set up for each test"""
        self.llm_service = LLMService()

    @patch('services.llm_service.requests.post')
    def test_enhance_prompt_success(self, mock_post):
        """Test successful prompt enhancement"""
        # Mock the Ollama API response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": TEST_ENHANCED_PROMPT}
        mock_post.return_value = mock_response

        # Call the service
        result = self.llm_service.enhance_prompt(TEST_PROMPT)

        # Assertions
        self.assertEqual(result, TEST_ENHANCED_PROMPT)
        mock_post.assert_called_once()

        # Verify request format
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['model'], "deepseek-r1:8b")
        self.assertTrue(TEST_PROMPT in kwargs['json']['prompt'])

    @patch('services.llm_service.requests.post')
    def test_enhance_prompt_api_error(self, mock_post):
        """Test error handling when the API call fails"""
        # Mock a request exception
        mock_post.side_effect = Exception("API connection error")

        # Call should not raise an exception - it should fallback
        result = self.llm_service.enhance_prompt(TEST_PROMPT)

        # Should fallback to original prompt
        self.assertEqual(result, TEST_PROMPT)

    @patch('services.llm_service.requests.post')
    def test_enhance_prompt_retry_logic(self, mock_post):
        """Test that retry logic works properly"""
        # First call fails, second one succeeds
        mock_error_response = MagicMock()
        mock_error_response.raise_for_status.side_effect = Exception("Rate limit exceeded")

        mock_success_response = MagicMock()
        mock_success_response.raise_for_status.return_value = None
        mock_success_response.json.return_value = {"response": TEST_ENHANCED_PROMPT}

        mock_post.side_effect = [mock_error_response, mock_success_response]

        # Configure service with faster retries for testing
        self.llm_service.retry_delay = 0.01

        # Call should succeed after retry
        with patch('services.llm_service.time.sleep') as mock_sleep:  # Don't actually sleep in tests
            result = self.llm_service.enhance_prompt(TEST_PROMPT)

        # Check results
        self.assertEqual(result, TEST_ENHANCED_PROMPT)
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once()

    def test_init_with_custom_params(self):
        """Test initializing with custom parameters"""
        custom_service = LLMService(
            model_name="different-model",
            api_base="http://other-server:8080/api",
            temperature=0.5
        )

        self.assertEqual(custom_service.model_name, "different-model")
        self.assertEqual(custom_service.api_base, "http://other-server:8080/api")
        self.assertEqual(custom_service.temperature, 0.5)


if __name__ == '__main__':
    unittest.main()