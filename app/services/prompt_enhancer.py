import logging
from typing import Dict, List, Optional, Any

from services.llm_service import LLMService
from services.prompt_strategies import PromptStrategyFactory


class PromptEnhancer:
    """
    Enhanced prompt processing system that combines LLM capabilities with
    specialized prompt strategies to generate optimal descriptions for
    text-to-image and image-to-3D conversion.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initialize the prompt enhancer with an LLM service.

        Args:
            llm_service (LLMService): The LLM service implementation to use
        """
        self.llm_service = llm_service

    def process(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Process a user prompt through the complete enhancement pipeline."""
        logging.info(f"Processing prompt: '{prompt}'")

        # Step 1: Apply specialized strategy
        strategy = PromptStrategyFactory.get_strategy(prompt)
        strategy_type = strategy.__class__.__name__
        logging.info(f"Selected strategy: {strategy_type}")

        strategy_prompt = strategy.enhance(prompt)
        logging.info(f"Strategy enhanced prompt: '{strategy_prompt}'")

        # Step 2: Enhance with LLM
        llm_enhanced = self.llm_service.enhance_prompt(strategy_prompt)
        logging.info(f"LLM enhanced prompt: '{llm_enhanced}'")

        # Step 3: Validate and optimize output
        if not self.llm_service.validate_output(llm_enhanced):
            logging.warning("Enhanced prompt failed validation, using backup method")
            final_prompt = self._create_backup_prompt(prompt)
        else:
            final_prompt = llm_enhanced

        # Intelligently trim the prompt if it's too long
        max_length = 1000  # Reduced from 2000 to ensure better handling
        if len(final_prompt) > max_length:
            logging.warning(f"Prompt too long ({len(final_prompt)} chars). Trimming intelligently.")
            final_prompt = self._intelligent_trim(final_prompt, max_length)

        logging.info(f"Final enhanced prompt: '{final_prompt}'")
        return final_prompt

    def _intelligent_trim(self, text: str, max_length: int = 1000) -> str:
        """
        Intelligently trim the text to the specified maximum length.
        Preserves complete sentences where possible.

        Args:
            text (str): The text to trim
            max_length (int): The maximum allowed length

        Returns:
            str: The trimmed text
        """
        if len(text) <= max_length:
            return text

        # Find a good breaking point (end of sentence)
        breakpoint = max_length
        while breakpoint > max_length // 2:
            if text[breakpoint] in ['.', '!', '?'] and text[breakpoint-1] not in ['.', '!', '?']:
                return text[:breakpoint+1].strip()
            breakpoint -= 1

        # If no good sentence break found, break at a space
        breakpoint = max_length
        while breakpoint > max_length // 2:
            if text[breakpoint] == ' ':
                return text[:breakpoint].strip()
            breakpoint -= 1

        # Last resort: hard break at max_length
        return text[:max_length].strip()

    def _create_backup_prompt(self, prompt: str) -> str:
        """
        Create a backup enhanced prompt when the LLM enhancement fails.

        Args:
            prompt (str): The original user prompt

        Returns:
            str: A simple enhanced prompt
        """
        # Simple template-based enhancement
        backup_template = """
        A highly detailed, professional 3D model of {prompt}.
        Include rich textures, proper lighting, and careful attention to detail.
        The model should be well-proportioned with clearly defined surfaces and materials.
        Render with high quality settings, soft shadows, and proper perspective.
        """
        return backup_template.format(prompt=prompt)
