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
        """
        Process a user prompt through the complete enhancement pipeline:
        1. Apply specialized strategy based on prompt type
        2. Enhance with LLM for creative details
        3. Validate and refine the output

        Args:
            prompt (str): The original user prompt
            context (Optional[List[Dict[str, Any]]]): Optional context from previous interactions

        Returns:
            str: Fully enhanced prompt ready for image generation
        """
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

        # Step 3: Validate output
        if not self.llm_service.validate_output(llm_enhanced):
            logging.warning(
                "Enhanced prompt failed validation, using backup method")
            # Fallback to simpler enhancement if the full enhancement fails validation
            final_prompt = self._create_backup_prompt(prompt)
        else:
            final_prompt = llm_enhanced

        # For very long prompts, trim to a reasonable length
        if len(final_prompt) > 1000:
            logging.info("Trimming prompt to reasonable length")
            final_prompt = final_prompt[:1000]

        logging.info(f"Final enhanced prompt: '{final_prompt}'")
        return final_prompt

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
