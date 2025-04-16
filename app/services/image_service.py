import logging
import os
import base64
import time
import requests
from typing import Optional, Dict, Any, Union


class TextToImageService:
    """
    Service for converting text prompts to images using the Openfabric Text-to-Image app.
    """

    # Working example app ID from README
    TEXT_TO_IMAGE_APP_ID = "c25dcd829d134ea98f5ae4dd311d13bc"

    # Original app ID (not currently working)
    # TEXT_TO_IMAGE_APP_ID = "f0997a01-d6d3-a5fe-53d8-561300318557"

    def __init__(self, stub):
        """
        Initialize the TextToImageService.

        Args:
            stub: An initialized Stub instance for making calls to Openfabric apps
        """
        self.stub = stub
        self.app_url = self._format_app_url(self.TEXT_TO_IMAGE_APP_ID)
        logging.info(
            f"TextToImageService initialized with app URL: {self.app_url}")

    @staticmethod
    def _format_app_url(app_id: str) -> str:
        """
        Format an app ID into the proper URL format for Openfabric services.

        Args:
            app_id (str): The raw app ID

        Returns:
            str: The properly formatted URL
        """
        # Format according to README example: app_id.node3.openfabric.network
        return f"{app_id}.node3.openfabric.network"

    def generate_image(self, prompt: str, user_id: str = 'super-user') -> Dict[str, Any]:
        """Generate an image from a text prompt."""
        logging.info(f"Generating image for prompt: '{prompt}'")

        # Add retries for resilience
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Prepare the request data
                request_data = {'prompt': prompt}

                # Set timeout to avoid hanging indefinitely
                timeout_seconds = 30
                start_time = time.time()

                # Make the call with timeout monitoring
                try:
                    # Make the call to the Text-to-Image app
                    response_data = self.stub.call(self.app_url, request_data, user_id)

                    # Request succeeded
                    if not response_data:
                        raise Exception("Empty response from Text-to-Image service")

                    logging.info(f"Image generated successfully after {attempt+1} attempts")
                    return response_data

                except Exception as call_error:
                    # Check if we've exceeded our timeout
                    if time.time() - start_time > timeout_seconds:
                        raise Exception(f"Request timed out after {timeout_seconds} seconds")
                    raise call_error

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logging.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"Image generation failed after {max_retries} attempts: {str(e)}")

                    # Try alternate app ID as fallback
                    if self.app_url == self._format_app_url(self.TEXT_TO_IMAGE_APP_ID):
                        logging.info("Trying alternate Text-to-Image service...")

                        # Use backup app ID (switching between the two known IDs)
                        alternate_app_id = "c25dcd829d134ea98f5ae4dd311d13bc" \
                            if self.TEXT_TO_IMAGE_APP_ID != "c25dcd829d134ea98f5ae4dd311d13bc" \
                            else "f0997a01-d6d3-a5fe-53d8-561300318557"

                        self.app_url = self._format_app_url(alternate_app_id)
                        try:
                            # Final attempt with alternate service
                            response_data = self.stub.call(self.app_url, request_data, user_id)
                            if response_data:
                                logging.info(f"Image generated using alternate service")
                                return response_data
                        except Exception as alt_error:
                            logging.error(f"Alternate service also failed: {str(alt_error)}")

                    # Return a mock/placeholder response for graceful degradation
                    return self._generate_mock_response(prompt)

    def _generate_mock_response(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a mock response when all API calls fail.
        This allows the application to continue functioning with degraded capability.

        Args:
            prompt (str): The original prompt

        Returns:
            Dict[str, Any]: A mock response structure
        """
        logging.warning("Generating mock response due to service failure")
        return {
            "result": f"mock_image_{int(time.time())}",
            "_mock": True,
            "prompt": prompt,
            "error": "Generated mock response due to service unavailability"
        }

    def save_image_reference(self, image_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Save the image reference to disk.

        Args:
            image_data (Dict[str, Any]): The response data containing the image reference
            filename (Optional[str]): Optional custom filename

        Returns:
            str: The path to the saved reference file

        Raises:
            Exception: If saving the image reference fails
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "output", "images")
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename if not provided
            if not filename:
                timestamp = int(time.time())
                filename = f"reference_{timestamp}.txt"

            # Determine the full path
            reference_path = os.path.join(output_dir, filename)

            # Extract the image reference from the response
            image_reference = None
            if 'result' in image_data:
                image_reference = image_data['result']
            else:
                # Try to find a suitable reference in the response
                for key, value in image_data.items():
                    if isinstance(value, str):
                        image_reference = value
                        break

            if not image_reference:
                raise Exception("No image reference found in response")

            # Save the reference to a file
            with open(reference_path, 'w') as f:
                f.write(f"Image reference: {image_reference}\n")
                f.write(
                    f"Original prompt timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Response data: {str(image_data)}\n")

            logging.info(f"Image reference saved to: {reference_path}")

            # Return the path to the saved reference file
            return reference_path

        except Exception as e:
            logging.error(f"Failed to save image reference: {str(e)}")
            raise Exception(f"Failed to save image reference: {str(e)}")

    def fetch_image_data(self, image_reference: str) -> bytes:
        """
        Fetch the actual image data using the reference ID.

        Note: This is a placeholder method. In a real implementation, you would use
        the Openfabric API to fetch the actual image data using the reference.

        Args:
            image_reference (str): The image reference ID from the Text-to-Image service

        Returns:
            bytes: The actual image data

        Raises:
            NotImplementedError: This method needs to be implemented based on Openfabric's API
        """
        # This is where you would implement the logic to fetch the actual image data
        # using the Openfabric API and the reference ID
        raise NotImplementedError(
            "Fetching actual image data from reference ID is not yet implemented. "
            "This requires additional information about Openfabric's API for retrieving binary data."
        )
