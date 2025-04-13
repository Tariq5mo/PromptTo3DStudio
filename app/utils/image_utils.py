import os
import base64
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

class ImageUtils:
    """
    Utility class for handling image operations including base64 encoding/decoding,
    file operations, and metadata management.
    """

    def __init__(self, images_dir: str = "data/images", models_dir: str = "data/models"):
        """
        Initialize the image utilities with directories for storing assets.

        Args:
            images_dir: Directory path for storing generated images
            models_dir: Directory path for storing generated 3D models
        """
        self.images_dir = Path(images_dir)
        self.models_dir = Path(models_dir)

        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"ImageUtils initialized with directories: images={self.images_dir}, models={self.models_dir}")

    def save_base64_image(self, base64_data: str, prompt: str, prefix: str = "image") -> Tuple[str, str]:
        """
        Save a base64-encoded image to disk and return the file path.

        Args:
            base64_data: Base64 encoded image data
            prompt: The prompt used to generate the image (for metadata)
            prefix: Filename prefix for the saved image

        Returns:
            Tuple containing (filepath, filename)
        """
        try:
            # Remove data URL prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]

            # Decode the base64 data
            image_data = base64.b64decode(base64_data)

            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.png"
            filepath = self.images_dir / filename

            # Write the image data to file
            with open(filepath, "wb") as f:
                f.write(image_data)

            # Save associated metadata
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    "prompt": prompt,
                    "timestamp": timestamp,
                    "filename": filename
                }, f, indent=2)

            logging.info(f"Saved image to {filepath} with metadata")
            return str(filepath), filename

        except Exception as e:
            logging.error(f"Error saving base64 image: {str(e)}")
            raise

    def load_image_as_base64(self, filepath: str) -> str:
        """
        Load an image from disk and return it as base64.

        Args:
            filepath: Path to the image file to load

        Returns:
            Base64 encoded string representation of the image
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"Image file not found: {filepath}")

            with open(filepath, "rb") as f:
                image_data = f.read()

            base64_data = base64.b64encode(image_data).decode("utf-8")
            logging.info(f"Loaded image from {filepath} as base64")
            return base64_data

        except Exception as e:
            logging.error(f"Error loading image as base64: {str(e)}")
            raise

    def save_3d_model(self, model_data: Dict[str, Any], prompt: str, image_path: str, prefix: str = "model") -> Tuple[str, str]:
        """
        Save a 3D model and associated metadata.

        Args:
            model_data: Dictionary containing the 3D model data
            prompt: Original or enhanced prompt used for generation
            image_path: Path to the source image used for 3D generation
            prefix: Filename prefix for the saved model

        Returns:
            Tuple containing (filepath, filename)
        """
        try:
            # Create a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.json"
            filepath = self.models_dir / filename

            # Extract the model data and add metadata
            model_with_metadata = {
                "model_data": model_data,
                "metadata": {
                    "original_prompt": prompt,
                    "source_image": image_path,
                    "timestamp": timestamp,
                    "generation_time": datetime.now().isoformat()
                }
            }

            # Write model with metadata to file
            with open(filepath, "w") as f:
                json.dump(model_with_metadata, f, indent=2)

            logging.info(f"Saved 3D model to {filepath} with metadata")
            return str(filepath), filename

        except Exception as e:
            logging.error(f"Error saving 3D model: {str(e)}")
            raise