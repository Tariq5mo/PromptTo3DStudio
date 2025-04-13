import logging
import time
from typing import Dict, Any, Optional, Tuple

from core.stub import Stub
from services.llm_service import LLMService
from utils.image_utils import ImageUtils
from utils.error_handling import ExecutionContext, retry

# App IDs for the Openfabric services
TEXT_TO_IMAGE_APP_ID = "f0997a01-d6d3-a5fe-53d8-561300318557"
IMAGE_TO_3D_APP_ID = "69543f29-4d41-4afc-7f29-3d51591f11eb"

class PipelineService:
    """
    Service that coordinates the entire text to 3D model pipeline.
    This service orchestrates:
    1. LLM prompt enhancement
    2. Text-to-Image conversion
    3. Image-to-3D model conversion
    """

    def __init__(self):
        """Initialize the pipeline service with required components."""
        self.llm_service = LLMService()
        self.image_utils = ImageUtils()
        logging.info("Pipeline service initialized")

    def process(self, user_prompt: str) -> ExecutionContext:
        """
        Process a user prompt through the entire pipeline.

        Args:
            user_prompt: The original prompt from the user

        Returns:
            An ExecutionContext containing results and status information
        """
        # Initialize execution context for tracking
        context = ExecutionContext(user_prompt)
        correlation_id = context.correlation_id # Get the ID

        try:
            # Step 1: Enhance prompt with LLM
            context.update_stage("prompt_enhancement")
            # Pass correlation_id here
            enhanced_prompt = self.llm_service.enhance_prompt(user_prompt, correlation_id)
            context.enhanced_prompt = enhanced_prompt

            # Step 2: Generate image from enhanced prompt
            context.update_stage("text_to_image")
            # Pass correlation_id here
            image_result = self._generate_image_from_text(enhanced_prompt, correlation_id)

            if not image_result or "image" not in image_result:
                raise ValueError("Failed to generate image from text")

            # Save the image to disk
            context.update_stage("image_saving")
            image_path, image_filename = self.image_utils.save_base64_image(
                image_result["image"],
                enhanced_prompt,
                "text2img"
            )
            context.image_path = image_path
            context.image_filename = image_filename

            # Step 3: Generate 3D model from image
            context.update_stage("image_to_3d")
            # Pass correlation_id here
            model_result = self._generate_3d_from_image(image_result["image"], correlation_id)

            if not model_result:
                raise ValueError("Failed to generate 3D model from image")

            # Save the 3D model data
            context.update_stage("model_saving")
            model_path, model_filename = self.image_utils.save_3d_model(
                model_result,
                enhanced_prompt,
                image_path,
                "model"
            )
            context.model_path = model_path
            context.model_filename = model_filename

            # Final stage complete
            context.update_stage("complete")
            logging.info(f"[{context.correlation_id}] Pipeline execution completed successfully in {context.get_execution_time():.2f}s")

        except Exception as e:
            context.set_error(e)
            logging.error(f"[{context.correlation_id}] Pipeline execution failed: {str(e)}", exc_info=True)

        return context

    @retry(max_attempts=3, delay=2.0)
    # Add correlation_id parameter
    def _generate_image_from_text(self, prompt: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an image from text using the Text-to-Image app.

        Args:
            prompt: The text prompt to generate an image from
            correlation_id: Optional correlation ID for logging traceability

        Returns:
            Dictionary containing the app response
        """
        log_prefix = f"[{correlation_id}] " if correlation_id else ""
        logging.info(f"{log_prefix}Calling Text-to-Image app with prompt: '{prompt}'")
        start_time = time.time()

        # Initialize the Stub with the Text-to-Image app ID
        stub = Stub([TEXT_TO_IMAGE_APP_ID])

        # Fetch schema for dynamic validation (optional but good practice)
        input_schema = stub.schema(TEXT_TO_IMAGE_APP_ID, 'input')
        logging.debug(f"{log_prefix}Text-to-Image input schema: {input_schema}")

        # Call the app with the prompt
        result = stub.call(TEXT_TO_IMAGE_APP_ID, {"prompt": prompt})

        # Log performance metrics
        elapsed_time = time.time() - start_time
        logging.info(f"{log_prefix}Text-to-Image generation completed in {elapsed_time:.2f} seconds")

        if not result or "image" not in result:
            raise ValueError("Failed to generate image from text")

        return result

    @retry(max_attempts=3, delay=2.0)
    # Add correlation_id parameter
    def _generate_3d_from_image(self, image_base64: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a 3D model from an image using the Image-to-3D app.

        Args:
            image_base64: Base64-encoded image
            correlation_id: Optional correlation ID for logging traceability

        Returns:
            Dictionary containing the app response
        """
        log_prefix = f"[{correlation_id}] " if correlation_id else ""
        logging.info(f"{log_prefix}Calling Image-to-3D app with image")
        start_time = time.time()

        # Initialize the Stub with the Image-to-3D app ID
        stub = Stub([IMAGE_TO_3D_APP_ID])

        # Fetch schema for dynamic validation (optional but good practice)
        input_schema = stub.schema(IMAGE_TO_3D_APP_ID, 'input')
        logging.debug(f"{log_prefix}Image-to-3D input schema: {input_schema}")

        # Call the app with the image
        result = stub.call(IMAGE_TO_3D_APP_ID, {"image": image_base64})

        # Log performance metrics
        elapsed_time = time.time() - start_time
        logging.info(f"{log_prefix}Image-to-3D generation completed in {elapsed_time:.2f} seconds")

        if not result:
            raise ValueError("Failed to generate 3D model from image")

        return result