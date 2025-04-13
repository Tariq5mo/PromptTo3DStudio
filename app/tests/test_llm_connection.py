import sys
import os
import logging

# Add app directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the service
from services.llm_service import LLMService
from utils.logging_utils import setup_logging

# Setup basic logging
setup_logging()

def test_connection():
    """Tests the connection and basic functionality of the LLMService."""
    logging.info("Starting LLM connection test...")

    try:
        # Initialize the service with longer timeout (120 seconds instead of default 30)
        # First request to Ollama often takes longer as it loads the model into memory
        llm_service = LLMService(request_timeout=120)

        # Define a simple test prompt
        test_prompt = "A simple red cube"
        logging.info(f"Using test prompt: '{test_prompt}'")

        print("\nConnecting to Ollama. This might take a while if the model is loading for the first time...")

        # Call the enhance_prompt method
        enhanced_prompt = llm_service.enhance_prompt(test_prompt)

        # Print the results
        print("\n--- LLM Test Results ---")
        print(f"Original Prompt: {test_prompt}")
        print(f"Enhanced Prompt: {enhanced_prompt}")
        print("------------------------\n")

        if enhanced_prompt != test_prompt:
            logging.info("LLM enhancement successful!")
        else:
            logging.warning("LLM enhancement failed or returned the original prompt. Check Ollama server and logs.")

    except Exception as e:
        logging.error(f"LLM connection test failed: {str(e)}", exc_info=True)
        print("\n--- LLM Test Failed ---")
        print("Could not connect to or get a response from the LLM.")
        print("Ensure the Ollama server is running and the 'deepseek-r1:8b' model is available.")
        print("-----------------------\n")

if __name__ == "__main__":
    test_connection()