import os
import sys
import logging
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Add app directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Setup logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants for testing
TEST_TEXT_TO_IMAGE_APP_ID = "f0997a01-d6d3-a5fe-53d8-561300318557"
TEST_IMAGE_TO_3D_APP_ID = "69543f29-4d41-4afc-7f29-3d51591f11eb"
TEST_PROMPT = "A futuristic city with flying cars and tall buildings"
TEST_ENHANCED_PROMPT = "A breathtaking futuristic metropolis with gleaming skyscrapers reaching toward the clouds, flying vehicles zipping between buildings, neon lights reflecting off glass surfaces, and elevated walkways connecting towers, all under a sunset sky with rich purple and orange hues creating dramatic shadows across the cityscape."

# Sample base64 image (tiny 1x1 pixel transparent PNG)
SAMPLE_BASE64_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeAJcI9BX2AAAAABJRU5ErkJggg=="

# Sample 3D model data
SAMPLE_3D_MODEL = {
    "format": "glb",
    "model": "base64_data_would_go_here",
    "preview": "base64_preview_image"
}

def get_test_data_dir():
    """Get directory for test data, creating it if it doesn't exist"""
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir

def setup_test_directories():
    """Create test directories for images and models"""
    test_data_dir = get_test_data_dir()
    images_dir = test_data_dir / "images"
    models_dir = test_data_dir / "models"

    images_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    return images_dir, models_dir