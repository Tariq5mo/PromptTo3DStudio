#!/usr/bin/env python3
from services.prompt_enhancer import PromptEnhancer
from services.ollama_service import OllamaService
import logging
import sys

# Configure logging to see detailed output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Import our LLM service implementation


def test_ollama_connection():
    """Test basic connectivity to the Ollama server"""
    logging.info("Testing connection to Ollama server...")

    try:
        # Initialize the Ollama service
        ollama = OllamaService(
            model_name="phi:2.7b",
            host="http://localhost:11434",
            temperature=0.7
        )
        logging.info("✓ Connection established successfully")
        return ollama
    except Exception as e:
        logging.error(f"✗ Connection failed: {str(e)}")
        return None


def test_prompt_enhancement(ollama, prompt="A castle on a hill"):
    """Test the prompt enhancement functionality"""
    logging.info(f"Testing prompt enhancement with: '{prompt}'")

    try:
        enhanced = ollama.enhance_prompt(prompt)
        logging.info(f"✓ Prompt enhanced successfully!")
        logging.info(f"Original: '{prompt}'")
        logging.info(f"Enhanced: '{enhanced}'")
        return enhanced
    except Exception as e:
        logging.error(f"✗ Prompt enhancement failed: {str(e)}")
        return None


def test_full_pipeline(prompt="A castle on a hill"):
    """Test the complete prompt enhancement pipeline including strategies"""
    logging.info(f"Testing full prompt enhancement pipeline with: '{prompt}'")

    try:
        # Initialize the service
        ollama = OllamaService(
            model_name="phi:2.7b",
            host="http://localhost:11434",
            temperature=0.7
        )

        # Create the prompt enhancer
        enhancer = PromptEnhancer(ollama)

        # Process the prompt
        enhanced = enhancer.process(prompt)

        logging.info(f"✓ Full pipeline completed successfully!")
        logging.info(f"Original: '{prompt}'")
        logging.info(f"Enhanced: '{enhanced}'")
        return enhanced
    except Exception as e:
        logging.error(f"✗ Pipeline failed: {str(e)}")
        return None


if __name__ == "__main__":
    # Get command line arguments for the prompt if provided
    if len(sys.argv) > 1:
        test_prompt = " ".join(sys.argv[1:])
    else:
        test_prompt = "A castle on a hill"

    print("=" * 80)
    print("OLLAMA INTEGRATION TEST")
    print("=" * 80)

    # Test basic connection
    ollama = test_ollama_connection()

    if ollama:
        print("\n" + "=" * 80)
        print("DIRECT PROMPT ENHANCEMENT TEST")
        print("=" * 80)
        test_prompt_enhancement(ollama, test_prompt)

    print("\n" + "=" * 80)
    print("FULL PIPELINE TEST")
    print("=" * 80)
    test_full_pipeline(test_prompt)
