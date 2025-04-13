import unittest
import os
import json
from unittest.mock import patch, MagicMock
import logging

# Import test configuration
from tests.test_config import (
    TEST_PROMPT,
    TEST_ENHANCED_PROMPT,
    SAMPLE_BASE64_IMAGE,
    SAMPLE_3D_MODEL,
    TEST_TEXT_TO_IMAGE_APP_ID,
    TEST_IMAGE_TO_3D_APP_ID,
    setup_test_directories
)

# Import components for integration test
from services.pipeline_service import PipelineService
from services.llm_service import LLMService
from utils.image_utils import ImageUtils
from utils.error_handling import ExecutionContext
from main import execute

# Import application model classes
from openfabric_pysdk.context import AppModel
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the entire AI pipeline"""

    def setUp(self):
        """Set up test environment"""
        # Set up test directories
        self.test_images_dir, self.test_models_dir = setup_test_directories()

        # Set up mock responses
        self.setup_mocks()

    def setup_mocks(self):
        """Set up mock objects for external services"""
        # Start patchers for external dependencies
        self.requests_patcher = patch('services.llm_service.requests.post')
        self.stub_patcher = patch('core.stub.Stub.call')
        self.mock_requests = self.requests_patcher.start()
        self.mock_stub_call = self.stub_patcher.start()

        # Set up mock response for LLM API
        mock_llm_response = MagicMock()
        mock_llm_response.raise_for_status.return_value = None
        mock_llm_response.json.return_value = {"response": TEST_ENHANCED_PROMPT}
        self.mock_requests.return_value = mock_llm_response

        # Set up mock responses for Openfabric apps
        self.mock_stub_call.side_effect = [
            {"image": SAMPLE_BASE64_IMAGE},  # Text-to-Image result
            SAMPLE_3D_MODEL                  # Image-to-3D result
        ]

    def tearDown(self):
        """Clean up after tests"""
        # Stop all patches
        self.requests_patcher.stop()
        self.stub_patcher.stop()

        # Clean up any created files
        for path in [self.test_images_dir, self.test_models_dir]:
            if path.exists():
                for file in path.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass

    @patch('main.pipeline_service')
    def test_execute_function_integration(self, mock_pipeline_service):
        """Test the main execute function with the pipeline service"""
        # Set up mock context that would be returned by pipeline
        mock_context = ExecutionContext(TEST_PROMPT)
        mock_context.enhanced_prompt = TEST_ENHANCED_PROMPT
        mock_context.image_path = "/path/to/image.png"
        mock_context.model_path = "/path/to/model.json"
        mock_context.stage = "complete"
        mock_pipeline_service.process.return_value = mock_context

        # Create the AppModel for testing
        model = AppModel()
        model.request = InputClass()
        model.request.prompt = TEST_PROMPT
        model.response = OutputClass()

        # Execute the main function
        execute(model)

        # Verify the pipeline was called with the prompt
        mock_pipeline_service.process.assert_called_once_with(TEST_PROMPT)

        # Verify the response was set correctly
        self.assertTrue(TEST_PROMPT in model.response.message)
        self.assertTrue(TEST_ENHANCED_PROMPT in model.response.message)
        self.assertTrue("/path/to/image.png" in model.response.message)
        self.assertTrue("/path/to/model.json" in model.response.message)

    @patch('services.pipeline_service.TEXT_TO_IMAGE_APP_ID', TEST_TEXT_TO_IMAGE_APP_ID)
    @patch('services.pipeline_service.IMAGE_TO_3D_APP_ID', TEST_IMAGE_TO_3D_APP_ID)
    def test_full_pipeline_integration(self):
        """
        Test the full pipeline integration from prompt to 3D model

        This test patches the external API calls but allows all internal
        components to work together as they would in production.
        """
        # Override the image utilities to use test directories
        image_utils = ImageUtils(
            images_dir=str(self.test_images_dir),
            models_dir=str(self.test_models_dir)
        )

        # Create a real pipeline service but inject our test image utils
        pipeline_service = PipelineService()
        pipeline_service.image_utils = image_utils

        # Process a prompt through the pipeline
        context = pipeline_service.process(TEST_PROMPT)

        # Assert the pipeline completed successfully
        self.assertIsNone(context.error, f"Pipeline failed with error: {context.error}")
        self.assertEqual(context.stage, "complete")
        self.assertEqual(context.enhanced_prompt, TEST_ENHANCED_PROMPT)

        # Verify image was saved
        self.assertIsNotNone(context.image_path)
        self.assertTrue(os.path.exists(context.image_path))

        # Verify model was saved
        self.assertIsNotNone(context.model_path)
        self.assertTrue(os.path.exists(context.model_path))

        # Verify model content
        with open(context.model_path, 'r') as f:
            model_data = json.load(f)
            self.assertEqual(model_data["model_data"], SAMPLE_3D_MODEL)
            self.assertEqual(model_data["metadata"]["original_prompt"], TEST_ENHANCED_PROMPT)

    @patch('main.configurations')
    def test_main_execute_with_error(self, mock_configurations):
        """Test the main execute function when the pipeline encounters an error"""
        # Set up mock pipeline service that raises an error
        with patch('main.pipeline_service') as mock_pipeline_service:
            # Create error context
            error_context = ExecutionContext(TEST_PROMPT)
            error_context.stage = "text_to_image"
            error_context.set_error(ValueError("Test pipeline error"))
            mock_pipeline_service.process.return_value = error_context

            # Create the AppModel for testing
            model = AppModel()
            model.request = InputClass()
            model.request.prompt = TEST_PROMPT
            model.response = OutputClass()

            # Execute the main function
            execute(model)

            # Verify response contains error message
            self.assertTrue("Error" in model.response.message)
            self.assertTrue("Test pipeline error" in model.response.message)


if __name__ == "__main__":
    unittest.main()