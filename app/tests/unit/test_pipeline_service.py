import unittest
from unittest.mock import patch, MagicMock
import json

# Import test configuration
from tests.test_config import (
    TEST_PROMPT,
    TEST_ENHANCED_PROMPT,
    SAMPLE_BASE64_IMAGE,
    SAMPLE_3D_MODEL,
    TEST_TEXT_TO_IMAGE_APP_ID,
    TEST_IMAGE_TO_3D_APP_ID
)

# Import the service to test
from services.pipeline_service import PipelineService
from utils.error_handling import ExecutionContext

class TestPipelineService(unittest.TestCase):
    """Unit tests for the Pipeline service that orchestrates the workflow"""

    def setUp(self):
        """Set up for each test"""
        # Create pipeline service with mocked components
        self.pipeline_service = PipelineService()

        # Add patchers for dependencies
        self.llm_patcher = patch.object(self.pipeline_service, 'llm_service')
        self.image_utils_patcher = patch.object(self.pipeline_service, 'image_utils')
        self.stub_patcher = patch('services.pipeline_service.Stub')

        # Start patchers
        self.mock_llm_service = self.llm_patcher.start()
        self.mock_image_utils = self.image_utils_patcher.start()
        self.mock_stub_class = self.stub_patcher.start()

        # Setup mock stub instance
        self.mock_stub = MagicMock()
        self.mock_stub_class.return_value = self.mock_stub

        # Setup default mock behaviors
        self.mock_llm_service.enhance_prompt.return_value = TEST_ENHANCED_PROMPT

        self.mock_stub.call.side_effect = [
            {"image": SAMPLE_BASE64_IMAGE},  # Text-to-Image result
            SAMPLE_3D_MODEL                  # Image-to-3D result
        ]

        self.mock_image_utils.save_base64_image.return_value = ("/path/to/image.png", "image.png")
        self.mock_image_utils.save_3d_model.return_value = ("/path/to/model.glb", "model.glb")

    def tearDown(self):
        """Clean up after each test"""
        self.llm_patcher.stop()
        self.image_utils_patcher.stop()
        self.stub_patcher.stop()

    def test_process_success(self):
        """Test successful processing through the entire pipeline"""
        # Call the process method
        context = self.pipeline_service.process(TEST_PROMPT)

        # Verify each step was called correctly
        self.mock_llm_service.enhance_prompt.assert_called_once_with(TEST_PROMPT)

        # Verify Text-to-Image call
        self.mock_stub_class.assert_any_call([TEST_TEXT_TO_IMAGE_APP_ID])
        self.mock_stub.call.assert_any_call(
            TEST_TEXT_TO_IMAGE_APP_ID,
            {"prompt": TEST_ENHANCED_PROMPT}
        )

        # Verify image was saved
        self.mock_image_utils.save_base64_image.assert_called_once_with(
            SAMPLE_BASE64_IMAGE,
            TEST_ENHANCED_PROMPT,
            "text2img"
        )

        # Verify Image-to-3D call
        self.mock_stub_class.assert_any_call([TEST_IMAGE_TO_3D_APP_ID])
        self.mock_stub.call.assert_any_call(
            TEST_IMAGE_TO_3D_APP_ID,
            {"image": SAMPLE_BASE64_IMAGE}
        )

        # Verify 3D model was saved
        self.mock_image_utils.save_3d_model.assert_called_once()

        # Verify context was updated correctly
        self.assertEqual(context.enhanced_prompt, TEST_ENHANCED_PROMPT)
        self.assertEqual(context.image_path, "/path/to/image.png")
        self.assertEqual(context.image_filename, "image.png")
        self.assertEqual(context.model_path, "/path/to/model.glb")
        self.assertEqual(context.model_filename, "model.glb")
        self.assertEqual(context.stage, "complete")
        self.assertIsNone(context.error)

    def test_llm_enhancement_failure(self):
        """Test handling of LLM enhancement failure"""
        # Mock LLM service to raise an exception
        self.mock_llm_service.enhance_prompt.side_effect = Exception("LLM service unavailable")

        # Call the process method
        context = self.pipeline_service.process(TEST_PROMPT)

        # Verify error was captured
        self.assertIsNotNone(context.error)
        self.assertEqual(context.stage, "prompt_enhancement")
        self.assertEqual(str(context.error), "LLM service unavailable")

        # Verify later steps were not called
        self.mock_stub.call.assert_not_called()
        self.mock_image_utils.save_base64_image.assert_not_called()
        self.mock_image_utils.save_3d_model.assert_not_called()

    def test_text_to_image_failure(self):
        """Test handling of Text-to-Image API failure"""
        # Mock stub to return invalid response
        self.mock_stub.call.side_effect = [
            {},  # Missing "image" field
            SAMPLE_3D_MODEL
        ]

        # Call the process method
        context = self.pipeline_service.process(TEST_PROMPT)

        # Verify error was captured
        self.assertIsNotNone(context.error)
        self.assertEqual(context.stage, "text_to_image")
        self.assertTrue("Failed to generate image from text" in str(context.error))

        # Verify LLM was called but later steps were not
        self.mock_llm_service.enhance_prompt.assert_called_once()
        self.mock_stub.call.assert_called_once()
        self.mock_image_utils.save_base64_image.assert_not_called()
        self.mock_image_utils.save_3d_model.assert_not_called()

    def test_image_to_3d_failure(self):
        """Test handling of Image-to-3D API failure"""
        # Mock stub to return valid response for first call, then None
        self.mock_stub.call.side_effect = [
            {"image": SAMPLE_BASE64_IMAGE},  # Good Text-to-Image result
            None                             # Failed Image-to-3D result
        ]

        # Call the process method
        context = self.pipeline_service.process(TEST_PROMPT)

        # Verify error was captured
        self.assertIsNotNone(context.error)
        self.assertEqual(context.stage, "image_to_3d")
        self.assertTrue("Failed to generate 3D model from image" in str(context.error))

        # Verify earlier steps were called but later steps were not
        self.mock_llm_service.enhance_prompt.assert_called_once()
        self.assertEqual(self.mock_stub.call.call_count, 2)
        self.mock_image_utils.save_base64_image.assert_called_once()
        self.mock_image_utils.save_3d_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()