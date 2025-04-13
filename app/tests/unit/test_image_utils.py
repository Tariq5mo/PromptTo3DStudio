import unittest
import os
import json
import base64
from pathlib import Path
from unittest.mock import patch, mock_open

# Import test configuration
from tests.test_config import (
    setup_test_directories,
    TEST_PROMPT,
    SAMPLE_BASE64_IMAGE,
    SAMPLE_3D_MODEL
)

# Import the utilities to test
from utils.image_utils import ImageUtils

class TestImageUtils(unittest.TestCase):
    """Unit tests for the image utilities"""

    def setUp(self):
        """Set up test environment"""
        # Set up test directories
        self.test_images_dir, self.test_models_dir = setup_test_directories()

        # Create instance with test directories
        self.image_utils = ImageUtils(
            images_dir=str(self.test_images_dir),
            models_dir=str(self.test_models_dir)
        )

        # Clean up any leftover test files
        for file in self.test_images_dir.glob("*"):
            file.unlink()
        for file in self.test_models_dir.glob("*"):
            file.unlink()

    def test_init_creates_directories(self):
        """Test that directories are created on initialization"""
        # Create a temporary path
        temp_dir = Path("temp_test_dir")

        # Create instance that should create directories
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            ImageUtils(
                images_dir=str(temp_dir / "images"),
                models_dir=str(temp_dir / "models")
            )

            # Check that mkdir was called twice with parents=True
            self.assertEqual(mock_mkdir.call_count, 2)
            for call in mock_mkdir.call_args_list:
                self.assertEqual(call.kwargs, {"parents": True, "exist_ok": True})

    def test_save_base64_image(self):
        """Test saving a base64 encoded image"""
        # Call the method
        filepath, filename = self.image_utils.save_base64_image(
            SAMPLE_BASE64_IMAGE,
            TEST_PROMPT,
            "test_image"
        )

        # Check that the file was created
        self.assertTrue(os.path.exists(filepath))

        # Check that metadata file was created
        metadata_path = filepath.replace('.png', '.json')
        self.assertTrue(os.path.exists(metadata_path))

        # Verify metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.assertEqual(metadata["prompt"], TEST_PROMPT)
            self.assertEqual(metadata["filename"], filename)

    def test_load_image_as_base64(self):
        """Test loading an image as base64"""
        # First save an image
        filepath, _ = self.image_utils.save_base64_image(
            SAMPLE_BASE64_IMAGE,
            TEST_PROMPT,
            "test_image"
        )

        # Then load it
        loaded_image = self.image_utils.load_image_as_base64(filepath)

        # The loaded image should match the original
        # (There might be minor encoding differences, so check the decoded data)
        original_decoded = base64.b64decode(SAMPLE_BASE64_IMAGE)
        loaded_decoded = base64.b64decode(loaded_image)
        self.assertEqual(original_decoded, loaded_decoded)

    def test_load_nonexistent_image(self):
        """Test loading an image that doesn't exist"""
        with self.assertRaises(FileNotFoundError):
            self.image_utils.load_image_as_base64("/path/to/nonexistent/image.png")

    def test_save_3d_model(self):
        """Test saving a 3D model"""
        # Test input data
        image_path = str(self.test_images_dir / "source_image.png")

        # Call the method
        filepath, filename = self.image_utils.save_3d_model(
            SAMPLE_3D_MODEL,
            TEST_PROMPT,
            image_path,
            "test_model"
        )

        # Check that the file was created
        self.assertTrue(os.path.exists(filepath))

        # Read and verify content
        with open(filepath, 'r') as f:
            saved_data = json.load(f)

            # Check model data
            self.assertEqual(saved_data["model_data"], SAMPLE_3D_MODEL)

            # Check metadata
            self.assertEqual(saved_data["metadata"]["original_prompt"], TEST_PROMPT)
            self.assertEqual(saved_data["metadata"]["source_image"], image_path)


if __name__ == "__main__":
    unittest.main()