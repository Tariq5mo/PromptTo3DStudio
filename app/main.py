import logging
import json
import os
from typing import Dict, List, Optional, Any

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub
from services.ollama_service import OllamaService
from services.prompt_enhancer import PromptEnhancer
from services.image_service import TextToImageService

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()
# Memory storage for context across requests
memory: Dict[str, List[Dict[str, Any]]] = dict()

# Initialize our services - do this at module level to persist between calls
llm_service = OllamaService(
    model_name="phi:2.7b",
    host="http://localhost:11434",
    temperature=0.7
)
prompt_enhancer = PromptEnhancer(llm_service)

############################################################
# Config callback function
############################################################


def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf


############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """

    # Retrieve input
    request: InputClass = model.request

    # Fix for the model.context issue - use try/except to handle potential attribute error
    try:
        user_id = model.context.user_id if hasattr(model, 'context') and hasattr(
            model.context, 'user_id') else 'super-user'
    except Exception:
        user_id = 'super-user'

    logging.info(f"Processing request from user: {user_id}")

    # Initialize memory for this user if not exists
    if user_id not in memory:
        memory[user_id] = []

    # Retrieve user config
    user_config: ConfigClass = configurations.get(
        user_id, configurations.get('super-user'))
    logging.info(f"Using config: {user_config}")

    # Get user prompt
    user_prompt = request.prompt
    if not user_prompt:
        model.response.message = "Please provide a prompt to generate a 3D model."
        return

    try:
        # Get context from memory for this user
        user_context = memory.get(user_id, [])

        # Step 1: Enhanced prompt generation using LLM
        logging.info(f"Enhancing user prompt: '{user_prompt}'")
        enhanced_prompt = prompt_enhancer.process(user_prompt, user_context)
        logging.info(f"Enhanced prompt: '{enhanced_prompt}'")

        # Initialize response message
        response_message = f"Enhanced prompt: {enhanced_prompt}\n\n"

        # Step 2: Format app IDs properly for Openfabric connection
        # We're using the example app ID from README that we confirmed works
        text_to_image_app_id = "c25dcd829d134ea98f5ae4dd311d13bc"
        image_to_3d_app_id = "69543f29-4afc-7f29-3d51591f11eb"

        # Format app IDs as URLs
        app_ids = [
            f"{text_to_image_app_id}.node3.openfabric.network",
            f"{image_to_3d_app_id}.node3.openfabric.network"
        ]

        # Initialize the Stub with correctly formatted app IDs
        stub = Stub(app_ids)

        # Step 3: Initialize the TextToImageService with our stub
        text_to_image_service = TextToImageService(stub)

        # Step 4: Generate image from enhanced prompt
        try:
            # Generate the image
            image_result = text_to_image_service.generate_image(
                enhanced_prompt, user_id)

            # Save the image reference
            reference_path = text_to_image_service.save_image_reference(
                image_result)

            # Add to response
            response_message += f"Successfully generated image reference! Saved to: {reference_path}\n\n"

            # Add to memory
            user_context.append({
                "prompt": user_prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_reference": image_result.get('result', ''),
                "reference_path": reference_path,
                "timestamp": int(os.path.basename(reference_path).split('_')[1].split('.')[0]) if '_' in os.path.basename(reference_path) else int(time.time())
            })

        except Exception as e:
            response_message += f"Failed to generate image: {str(e)}\n\n"
            # Still store prompt in memory even if image generation failed
            user_context.append({
                "prompt": user_prompt,
                "enhanced_prompt": enhanced_prompt,
                "error": str(e)
            })

        # Limit memory to last 10 interactions
        if len(user_context) > 10:
            user_context = user_context[-10:]
        memory[user_id] = user_context

        # Prepare final response
        model.response.message = response_message

    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")
        model.response.message = f"An error occurred during processing: {str(e)}"
