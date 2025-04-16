#!/usr/bin/env python3
import logging
import sys
from core.stub import Stub
from services.image_service import TextToImageService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


def test_text_to_image(prompt="A medieval castle on a cliff at sunset"):
    """
    Test the text-to-image service with a given prompt.

    Args:
        prompt (str): The prompt to convert to an image
    """
    print("=" * 80)
    print(f"TESTING TEXT-TO-IMAGE SERVICE WITH PROMPT: '{prompt}'")
    print("=" * 80)

    # Step 1: Format app IDs properly according to the README example
    text_to_image_app_id = "c25dcd829d134ea98f5ae4dd311d13bc"
    app_url = f"{text_to_image_app_id}.node3.openfabric.network"

    print(f"Using formatted app URL: {app_url}")

    try:
        # Step 2: Initialize Stub with the formatted app URL
        app_ids = [app_url]
        stub = Stub(app_ids)

        # Step 3: Create TextToImageService
        text_to_image_service = TextToImageService(stub)

        # Step 4: Generate image from prompt
        print(f"Generating image from prompt: '{prompt}'")
        image_result = text_to_image_service.generate_image(prompt)

        # Step 5: Save the generated image
        print("Image generated successfully, saving to disk...")
        image_path = text_to_image_service.save_image_reference(image_result)

        print(f"\n✅ SUCCESS! Image saved to: {image_path}")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False


if __name__ == "__main__":
    # Get prompt from command line if provided, otherwise use default
    user_prompt = " ".join(sys.argv[1:]) if len(
        sys.argv) > 1 else "A medieval castle on a cliff at sunset"

    # Run the test
    test_text_to_image(user_prompt)
