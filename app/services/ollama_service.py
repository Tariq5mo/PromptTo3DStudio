import json
import logging
import requests
import time
from typing import Dict, List, Optional, Any

from services.llm_service import LLMService


class OllamaService(LLMService):
    """
    Implementation of the LLM service using Ollama and the phi-2.7b model.
    """

    def __init__(self,
                 model_name: str = "phi:2.7b",
                 host: str = "http://localhost:11434",
                 temperature: float = 0.7,
                 max_retries: int = 3,
                 timeout: int = 90):  # Increased timeout from 30 to 90 seconds
        """
        Initialize the Ollama service with configuration parameters.

        Args:
            model_name (str): Name of the model to use
            host (str): Host URL for the Ollama service
            temperature (float): Sampling temperature (0.0-1.0), higher is more creative
            max_retries (int): Maximum retry attempts on failure
            timeout (int): Request timeout in seconds (default: 90 seconds)
        """
        self.model_name = model_name
        self.host = host
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout  # Now 90 seconds
        self.api_url = f"{host}/api/generate"

        # Verify the model is available
        self._verify_model_availability()

    def _verify_model_availability(self) -> None:
        """Checks if the specified model is available on the Ollama server."""
        try:
            response = requests.get(f"{self.host}/api/tags")
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]

            if self.model_name not in available_models:
                logging.warning(
                    f"Model {self.model_name} not found. Available models: {available_models}")
                logging.info(
                    f"Attempting to use closest match or default model...")
            else:
                logging.info(
                    f"Model {self.model_name} is available and ready for use")
        except Exception as e:
            logging.error(f"Error checking model availability: {e}")
            logging.warning(
                "Proceeding with requested model, but it may fail if not available")

    def _call_ollama_api(self, prompt: str, system_prompt: str = None) -> str:
        """
        Makes the actual API call to Ollama with retry logic.

        Args:
            prompt (str): The prompt to send to the model
            system_prompt (str, optional): System prompt for context/instructions

        Returns:
            str: The generated response text

        Raises:
            Exception: If all retry attempts fail
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }

        if system_prompt:
            payload["system"] = system_prompt

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json().get("response", "")

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                logging.error(
                    f"Attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                if attempt + 1 < self.max_retries:
                    backoff_time = 2 ** attempt  # Exponential backoff
                    logging.info(f"Retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                else:
                    logging.error("All retry attempts failed")
                    raise Exception(
                        f"Failed to communicate with Ollama API: {str(e)}")

    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhances a user prompt with additional details to improve image generation quality.

        Args:
            prompt (str): The original user prompt

        Returns:
            str: Enhanced prompt with additional details
        """
        system_prompt = """
        You are a creative prompt enhancer for text-to-image generation. Your task is to take a simple user prompt
        and enhance it with rich, detailed descriptions that will help generate high-quality images.

        Focus on adding:
        - Visual details (colors, textures, lighting, perspective)
        - Artistic style references
        - Mood and atmosphere elements
        - Technical parameters that would be helpful

        Keep your response focused on the enhancement only, without explanations or preambles.
        Your enhanced prompt should be coherent, descriptive and well-structured.
        """

        full_prompt = f"""
        Transform this simple prompt into a rich, detailed description for an image generator:

        PROMPT: "{prompt}"

        Enhanced prompt:
        """

        try:
            enhanced = self._call_ollama_api(full_prompt, system_prompt)
            if not enhanced or len(enhanced) < len(prompt):
                logging.warning(
                    "Enhanced prompt was shorter than original, returning original prompt")
                return prompt
            return enhanced
        except Exception as e:
            logging.error(f"Prompt enhancement failed: {str(e)}")
            return prompt  # Return original prompt on failure

    def generate_description(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generates a detailed description based on the user prompt and optional context.

        Args:
            prompt (str): The user prompt
            context (Optional[List[Dict[str, Any]]]): Optional context from previous interactions

        Returns:
            str: Generated detailed description
        """
        system_prompt = """
        You are a highly descriptive AI that specializes in creating rich, detailed descriptions
        for 3D model generation. Focus on physical characteristics, spatial relationships,
        textures, materials, and structural details that would be important for a 3D model.

        Provide details about:
        - Shape and form
        - Materials and textures
        - Proportions and scale
        - Component relationships
        - Surface details

        Your description should be useful for generating a 3D model from an image,
        so focus on physical and structural details rather than abstract concepts.
        """

        # Include context in prompt if provided
        context_text = ""
        if context:
            context_text = "Based on our previous conversation:\n"
            for item in context[-3:]:  # Use last 3 context items at most
                if "prompt" in item:
                    context_text += f"- Previous prompt: {item['prompt']}\n"
                if "response" in item:
                    context_text += f"- Previous response: {item['response']}\n"
            context_text += "\n"

        full_prompt = f"""
        {context_text}
        Create a rich, detailed description for generating a 3D model based on this prompt:

        "{prompt}"

        Detailed description:
        """

        try:
            return self._call_ollama_api(full_prompt, system_prompt)
        except Exception as e:
            logging.error(f"Description generation failed: {str(e)}")
            return f"Failed to generate description for: {prompt}"

    def validate_output(self, generated_text: str) -> bool:
        """
        Validates the generated text to ensure it meets quality standards.

        Args:
            generated_text (str): The text generated by the LLM

        Returns:
            bool: True if the text passes validation, False otherwise
        """
        # Basic validation rules
        min_length = 50
        max_length = 2000
        required_elements = ["color", "texture", "shape", "detail"]

        # Check length
        if len(generated_text) < min_length:
            logging.warning(
                f"Generated text too short: {len(generated_text)} chars")
            return False

        if len(generated_text) > max_length:
            logging.warning(
                f"Generated text too long: {len(generated_text)} chars")
            return False

        # Check for required descriptive elements
        lower_text = generated_text.lower()
        missing_elements = [element for element in required_elements
                            if element not in lower_text]

        if missing_elements:
            logging.warning(
                f"Generated text missing required elements: {missing_elements}")
            return False

        return True
