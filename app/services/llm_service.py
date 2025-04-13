import requests
import logging
import time
from typing import Dict, Any, Optional, List, Union

class LLMService:
    """
    Service class for interacting with Ollama's API to run local LLM inference
    with DeepSeek-r1:8b model.
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        api_base: str = "http://localhost:11434/api",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the LLM service.

        Args:
            model_name: Name of the Ollama model to use
            api_base: Base URL for Ollama API
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_retries: Maximum number of retry attempts on failure
            retry_delay: Base delay between retries in seconds (with exponential backoff)
        """
        self.model_name = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logging.info(f"LLM Service initialized with model: {model_name}")

    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance a user prompt with creative details using the LLM.

        This function sends the prompt to the local LLM to expand a simple user
        description into a detailed, visually rich prompt that will work well
        for image generation and subsequent 3D modeling.

        Args:
            prompt: The original user prompt to enhance

        Returns:
            The enhanced prompt with additional details
        """
        system_prompt = """You are an expert visual artist and creative writer specializing in 3D modeling concepts.

Your task is to enhance and expand simple user descriptions into vivid, detailed prompts that will:
1. Work well for text-to-image generation
2. Ultimately be converted to 3D models

Focus on:
- Visual elements (colors, textures, materials)
- Lighting and shadows
- Spatial relationships and depth
- Form and structure details that will translate well to 3D
- Keeping a cohesive artistic style

Maintain the core subject and intent of the original prompt while enhancing it with rich details.
Your output should be a single detailed paragraph without using bullet points or numbered lists.
Do not include phrases like "here's an enhanced description" - just provide the enhanced description directly."""

        try:
            return self._call_llm_with_retry(system_prompt, prompt)
        except Exception as e:
            logging.error(f"Error enhancing prompt with LLM: {str(e)}")
            logging.warning(f"Falling back to original prompt: '{prompt}'")
            # Fallback: return original prompt if enhancement fails
            return prompt

    def _call_llm_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM API with retry logic in case of failures.

        Args:
            system_prompt: System prompt to guide the LLM's behavior
            user_prompt: User input prompt

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

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base}/generate",
                    json=request_payload,
                    timeout=30  # 30 second timeout
                )
                response.raise_for_status()
                result = response.json()
                enhanced_prompt = result.get("response", "").strip()

                # Log performance metrics
                elapsed_time = time.time() - start_time
                logging.info(f"LLM inference completed in {elapsed_time:.2f} seconds")

                # Log prompt transformation
                logging.info(f"Original prompt: '{user_prompt}'")
                logging.info(f"Enhanced prompt: '{enhanced_prompt}'")

                return enhanced_prompt

            except requests.exceptions.RequestException as e:
                last_exception = e
                retry_wait = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"LLM API request failed (attempt {attempt+1}/{self.max_retries}): {str(e)}")
                logging.info(f"Retrying in {retry_wait} seconds...")
                time.sleep(retry_wait)

        # If we get here, all retries failed
        raise Exception(f"Failed to call LLM API after {self.max_retries} attempts: {str(last_exception)}")