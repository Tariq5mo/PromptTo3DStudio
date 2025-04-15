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
    user_id = model.context.user_id or 'super-user'
    logging.info(f"Processing request from user: {user_id}")

    # Initialize memory for this user if not exists
    if user_id not in memory:
        memory[user_id] = []

    # Retrieve user config
    user_config: ConfigClass = configurations.get(user_id, configurations.get('super-user'))
    logging.info(f"Using config: {user_config}")

    # Initialize the Stub with app IDs
    app_ids = user_config.app_ids if user_config else [
        # Default app IDs if not configured
        "f0997a01-d6d3-a5fe-53d8-561300318557",  # Text-to-Image
        "69543f29-4d41-4afc-7f29-3d51591f11eb"   # Image-to-3D
    ]
    stub = Stub(app_ids)

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

        # Store this prompt and enhancement in memory
        user_context.append({
            "prompt": user_prompt,
            "enhanced_prompt": enhanced_prompt
        })

        # Prepare response with the enhanced prompt
        response: OutputClass = model.response
        response.message = f"Enhanced prompt: {enhanced_prompt}\n\nNext steps would be to use this enhanced prompt with the Text-to-Image and Image-to-3D APIs."

        # Limit memory to last 10 interactions
        if len(user_context) > 10:
            user_context = user_context[-10:]
        memory[user_id] = user_context

    except Exception as e:
        logging.error(f"Error during prompt enhancement: {str(e)}")
        model.response.message = f"An error occurred during processing: {str(e)}"