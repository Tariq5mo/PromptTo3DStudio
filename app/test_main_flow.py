#!/usr/bin/env python3
from main import execute, llm_service, prompt_enhancer
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from openfabric_pysdk.context import AppModel
import logging
import sys
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import necessary components

# Import the execute function from main


def test_main_flow(prompt="A dragon perched on a castle tower at sunset"):
    """
    Test the main application flow with prompt enhancement

    Args:
        prompt (str): The user's prompt to process
    """
    print("=" * 80)
    print(f"TESTING MAIN FLOW WITH PROMPT: '{prompt}'")
    print("=" * 80)

    # Create an AppModel instance to simulate a request
    model = AppModel()

    # Set up the request with the user's prompt
    model.request = InputClass()
    model.request.prompt = prompt

    # Set up an empty response
    model.response = OutputClass()

    try:
        # Execute the main application flow
        print("Executing main application flow...")
        execute(model)

        # Print the response
        print("\n" + "=" * 80)
        print("RESPONSE:")
        print("=" * 80)
        print(model.response.message)

        return True
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    # Get prompt from command line if provided, otherwise use default
    user_prompt = " ".join(sys.argv[1:]) if len(
        sys.argv) > 1 else "A dragon perched on a castle tower at sunset"

    # Check if the LLM service is ready
    print("Checking LLM service...")
    try:
        # Test the LLM with a simple prompt enhancement
        test_response = llm_service.enhance_prompt("Test prompt")
        print(f"✅ LLM service responded: {test_response[:100]}..." if len(
            test_response) > 100 else f"✅ LLM service responded: {test_response}")
    except Exception as e:
        print(f"❌ LLM service error: {str(e)}")
        print("\nPlease make sure the Ollama service is running with the phi model:")
        print("  docker run -d -p 11434:11434 --name ollama ollama/ollama")
        print("  docker exec -it ollama ollama pull phi:2.7b")
        sys.exit(1)

    # Run the test
    test_main_flow(user_prompt)
