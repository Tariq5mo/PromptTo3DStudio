import requests
import logging
import time
from typing import Dict, Any, Optional, List, Union

class LLMService:
    """
    Service class for interacting with Ollama's API to run local LLM inference
    with phi model.
    """

    def __init__(
        self,
        model_name: str = "phi",
        api_base: str = "http://localhost:11434/api",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: int = 2,
        request_timeout: int = 60
    ):
        """
        Initialize the LLM service.

        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for Ollama API
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Base delay between retries in seconds (with exponential backoff)
            request_timeout: Timeout in seconds for API requests
        """
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout

        logging.info(f"LLM Service initialized with model: {model_name}")

    def enhance_prompt(self, prompt: str, correlation_id: Optional[str] = None) -> str:
        """
        Enhance a user prompt with creative details using the LLM.

        This function sends the prompt to the local LLM to expand a simple user
        description into a detailed, visually rich prompt that will work well
        for image generation and subsequent 3D modeling.

        Args:
            prompt: The original user prompt to enhance
            correlation_id: Optional correlation ID for logging traceability

        Returns:
            The enhanced prompt with additional details
        """
        system_prompt = """You are a 3D artist enhancing user descriptions for a text-to-3D pipeline.

Transform simple prompts into detailed visual descriptions that:
- Include specific visual elements (colors, materials, lighting)
- Emphasize 3D-friendly features (depth, form, structure)
- Maintain artistic coherence while expanding creatively

Output a single detailed paragraph that will drive high-quality text-to-image generation and subsequent 3D modeling.
Provide only the enhanced description without meta-text or formatting."""

        log_prefix = f"[{correlation_id}] " if correlation_id else ""

        try:
            # Pass correlation_id to the internal method
            return self._call_llm_with_retry(system_prompt, prompt, correlation_id)
        except Exception as e:
            logging.error(f"{log_prefix}Error enhancing prompt with LLM: {str(e)}")
            logging.warning(f"{log_prefix}Falling back to original prompt: '{prompt}'")
            # Fallback: return original prompt if enhancement fails
            return prompt

    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str, correlation_id: Optional[str] = None) -> str:
        """
        Call the LLM API with retry logic in case of failures.

        Args:
            system_prompt: System prompt to guide the LLM's behavior
            user_prompt: User input prompt
            correlation_id: Optional correlation ID for logging traceability

        Returns:
            The LLM's response text

        Raises:
            Exception: If all retry attempts fail
        """
        request_payload = {
            "model": self.model_name,
            "prompt": f"{system_prompt}\n\nUser description: {user_prompt}\n\nEnhanced description:",
            "stream": False,
            "temperature": self.temperature
        }

        log_prefix = f"[{correlation_id}] " if correlation_id else ""

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base}/generate",
                    json=request_payload,
                    timeout=self.request_timeout  # Use the configured timeout
                )
                response.raise_for_status()
                result = response.json()
                enhanced_prompt = result.get("response", "").strip()

                # Log performance metrics
                elapsed_time = time.time() - start_time
                logging.info(f"{log_prefix}LLM inference completed in {elapsed_time:.2f} seconds")

                # Log prompt transformation
                logging.info(f"{log_prefix}Original prompt: '{user_prompt}'")
                logging.info(f"{log_prefix}Enhanced prompt: '{enhanced_prompt}'")

                return enhanced_prompt

            except requests.exceptions.RequestException as e:
                last_exception = e
                retry_wait = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"{log_prefix}LLM API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                logging.info(f"{log_prefix}Retrying in {retry_wait} seconds...")
                time.sleep(retry_wait)

        # If we get here, all retries failed
        raise Exception(f"Failed to call LLM API after {self.max_retries} attempts: {str(last_exception)}")