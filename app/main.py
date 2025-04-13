import logging
from typing import Dict

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub

from services.pipeline_service import PipelineService

# Initialize the pipeline service
pipeline_service = PipelineService()

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

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

    if not request or not request.prompt:
        logging.error("No prompt provided in request")
        model.response.message = "Error: No prompt provided"
        return

    user_prompt = request.prompt
    logging.info(f"Received prompt: '{user_prompt}'")

    # Retrieve user config
    user_config: ConfigClass = configurations.get('super-user', None)
    logging.info(f"{configurations}")

    # Initialize the Stub with app IDs
    app_ids = user_config.app_ids if user_config else []
    stub = Stub(app_ids)

    # Process the prompt through the pipeline
    context = pipeline_service.process(user_prompt)

    # Create a formatted response message based on execution results
    if context.error:
        # Error case
        error_message = str(context.error)
        logging.error(f"Pipeline execution failed: {error_message}")
        model.response.message = f"Error: {error_message}"
    else:
        # Success case - construct a nicely formatted response
        response_message = (
            f"Successfully created 3D model from prompt!\n\n"
            f"Original Prompt: '{user_prompt}'\n\n"
            f"Enhanced Prompt: '{context.enhanced_prompt}'\n\n"
            f"Image saved to: {context.image_path}\n\n"
            f"3D Model saved to: {context.model_path}\n\n"
            f"Processing time: {context.get_execution_time():.2f} seconds"
        )

        # Prepare response
        response: OutputClass = model.response
        response.message = response_message

    # Log completion
    logging.info(f"Pipeline execution completed in {context.get_execution_time():.2f} seconds")