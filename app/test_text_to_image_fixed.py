#!/usr/bin/env python3
import logging
import sys
import os
import base64
import time
from core.stub import Stub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def test_text_to_image(prompt="A magical castle with dragons flying overhead"):
    """
    Test the text-to-image feature using the README example format.

    Args:
        prompt (str): The text prompt to convert to an image
    """
    print("=" * 80)
    print(f"TESTING TEXT-TO-IMAGE WITH PROMPT: '{prompt}'")
    print("=" * 80)

    # Use the example app ID from the README - this is known to work
    example_app_id = 'c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network'
    print(f"Using example app ID from README: {example_app_id}")

    try:
        # Initialize the Stub with the example app ID
        print("Initializing Stub...")
        stub = Stub([example_app_id])

        # Call the text-to-image service with the user's prompt
        print(f"Calling text-to-image service with prompt: '{prompt}'")
        result = stub.call(example_app_id, {'prompt': prompt}, 'super-user')

        if not result:
            print("❌ Received empty response")
            return False

        print(f"✅ Received response with keys: {list(result.keys())}")

        # Extract the image data/reference
        if 'result' in result:
            image_data = result['result']
            print(f"✅ Found data in 'result' field: {image_data[:100]}..." if len(
                str(image_data)) > 100 else f"✅ Found data in 'result' field: {image_data}")

            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "output", "images")
            os.makedirs(output_dir, exist_ok=True)

            # Generate a filename based on timestamp
            timestamp = int(time.time())
            filename = f"image_{timestamp}.png"
            file_path = os.path.join(output_dir, filename)

            # Determine if the result is binary data, base64, or a reference
            if isinstance(image_data, bytes):
                # Direct binary data
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                print(f"✅ Saved binary image to: {file_path}")
            elif isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # Base64 data URL
                    image_content = image_data.split(',')[1]
                    with open(file_path, 'wb') as f:
                        f.write(base64.b64decode(image_content))
                    print(f"✅ Saved base64 image to: {file_path}")
                elif len(image_data) > 100 and '/' in image_data:
                    # Likely a reference to a blob or file path
                    reference_file = os.path.join(
                        output_dir, f"reference_{timestamp}.txt")
                    with open(reference_file, 'w') as f:
                        f.write(f"Image reference: {image_data}\n")
                        f.write(
                            f"To retrieve: Use this reference to fetch the actual image data")
                    print(f"ℹ️ Saved image reference to: {reference_file}")
                    print(
                        f"NOTE: This is a reference ID. You need to retrieve the actual image from the Openfabric platform.")
                else:
                    # Try to decode as base64 anyway
                    try:
                        with open(file_path, 'wb') as f:
                            f.write(base64.b64decode(image_data))
                        print(f"✅ Saved decoded image to: {file_path}")
                    except:
                        # Just save as text
                        with open(file_path + '.txt', 'w') as f:
                            f.write(image_data)
                        print(f"ℹ️ Saved raw response to: {file_path}.txt")

            return True
        else:
            print(
                f"❌ No 'result' field found in response. Available keys: {list(result.keys())}")
            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    # Get prompt from command line if provided, otherwise use default
    user_prompt = " ".join(sys.argv[1:]) if len(
        sys.argv) > 1 else "A medieval castle on a cliff at sunset"

    # Run the test
    test_text_to_image(user_prompt)
