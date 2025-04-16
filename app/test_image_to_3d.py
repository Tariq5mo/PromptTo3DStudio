#!/usr/bin/env python3
import logging
import sys
import os
import time
from core.stub import Stub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Constants
IMAGE_TO_3D_APP_ID = "69543f29-4afc-7f29-3d51591f11eb"


def format_app_url(app_id: str) -> str:
    """
    Format an app ID into the proper URL format for Openfabric services.

    Args:
        app_id (str): The raw app ID

    Returns:
        str: The properly formatted URL
    """
    return f"{app_id}.node3.openfabric.network"


def test_image_to_3d(image_path=None):
    """
    Test the Image-to-3D service with a given image file.

    Args:
        image_path (str, optional): Path to the image file to convert. If None, a test image will be used.

    Returns:
        bool: True if the test was successful, False otherwise
    """
    print("=" * 80)
    print("TESTING IMAGE-TO-3D SERVICE")
    print("=" * 80)

    # Format the app URL
    app_url = format_app_url(IMAGE_TO_3D_APP_ID)
    print(f"Using app URL: {app_url}")

    # Use the provided image or a default one
    if not image_path:
        # Look for any existing image in the output directory
        output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "output", "images")

        # If no images found, notify the user
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            print(
                "❌ No images found in the output directory. Please run test_text_to_image.py first.")
            return False

        # Find the most recent image file
        image_files = [f for f in os.listdir(
            output_dir) if f.endswith('.png') or f.endswith('.jpg')]
        if not image_files:
            print(
                "❌ No image files found in output directory. Please run test_text_to_image.py first.")
            return False

        # Use the most recent image
        image_path = os.path.join(output_dir, sorted(image_files)[-1])

    print(f"Using image file: {image_path}")

    # Read the image file
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        print(f"✅ Successfully read image file ({len(image_data)} bytes)")
    except Exception as e:
        print(f"❌ Failed to read image file: {str(e)}")
        return False

    try:
        # Initialize the Stub
        stub = Stub([app_url])
        print("✅ Stub initialized")

        # Prepare request data - this may need to be adjusted based on the app's input schema
        request_data = {
            'image': image_data
        }

        # Call the Image-to-3D service
        print("Calling Image-to-3D service...")
        response = stub.call(app_url, request_data, 'super-user')

        if not response:
            print("❌ Received empty response from Image-to-3D service")
            return False

        print(f"✅ Received response: {response}")

        # Process the response (this will depend on the service's output format)
        # For now, we'll just log the response keys
        print(f"Response keys: {list(response.keys())}")

        # Create output directory for 3D models if it doesn't exist
        output_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), "output", "models")
        os.makedirs(output_dir, exist_ok=True)

        # Save any 3D model data if present
        if 'model' in response:
            # Adjust file extension as needed
            model_path = os.path.join(
                output_dir, f"model_{int(time.time())}.glb")
            with open(model_path, 'wb') as f:
                f.write(response['model'])
            print(f"✅ Model saved to: {model_path}")
        else:
            print(
                "❓ No model data found in response. Available keys: {list(response.keys())}")

            # Save the response data to a file for inspection
            response_path = os.path.join(
                output_dir, f"response_{int(time.time())}.txt")
            with open(response_path, 'w') as f:
                f.write(str(response))
            print(f"ℹ️ Response data saved to: {response_path}")

        return True

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Get optional image path from command line
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Run the test
    test_image_to_3d(image_path)
