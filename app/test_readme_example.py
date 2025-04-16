#!/usr/bin/env python3
from core.stub import Stub
import logging
import sys
import os
from typing import Dict, Any
import requests
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import our core components


def test_example_from_readme():
    """
    Test the example from the README file:

    object = stub.call('c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network',
                       {'prompt': 'Hello World!'}, 'super-user')
    image = object.get('result')
    with open('output.png', 'wb') as f:
        f.write(image)
    """
    print("=" * 80)
    print("TESTING EXAMPLE FROM README")
    print("=" * 80)

    # Use the example app ID from the README
    example_app_id = 'c25dcd829d134ea98f5ae4dd311d13bc.node3.openfabric.network'
    print(f"Using example app ID: {example_app_id}")

    try:
        # Initialize the Stub with the example app ID
        stub = Stub([example_app_id])

        # Print manifest if available
        manifest = stub.manifest(example_app_id)
        if manifest:
            print(f"✅ Successfully loaded manifest: {manifest}")
        else:
            print("❌ Failed to load manifest")

        # Test the call with the example prompt
        print("Calling app with prompt: 'Hello World!'")
        object_result = stub.call(
            example_app_id, {'prompt': 'Hello World!'}, 'super-user')

        if not object_result:
            print("❌ Received empty response from app")
            return False

        print(f"✅ Received response: {object_result}")

        # Extract the image data reference
        if 'result' in object_result:
            image_reference = object_result['result']
            print(
                f"✅ Found image reference in 'result' field: {image_reference}")

            # The image reference is a path or ID, not actual bytes
            # In a real implementation, you would need to fetch the actual image data
            # using this reference from the Openfabric platform

            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), "output", "images")
            os.makedirs(output_dir, exist_ok=True)

            # Save a placeholder or mock image for testing
            output_path = os.path.join(output_dir, "readme_example_output.txt")
            with open(output_path, "w") as f:
                f.write(f"Image reference: {image_reference}")

            print(f"✅ Successfully saved image reference to: {output_path}")
            print(f"NOTE: This is just the reference ID, not the actual image data.")
            print(
                f"To get the actual image, you would need to fetch it from the Openfabric platform")
            print(f"using the reference ID: {image_reference}")
            return True
        else:
            print(
                f"❌ No 'result' field found in response. Available fields: {list(object_result.keys())}")

            # Try to find any binary data in the response that might be the image
            for key, value in object_result.items():
                if isinstance(value, bytes) or (isinstance(value, str) and len(value) > 1000):
                    print(f"Found possible image data in field: '{key}'")

            return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


if __name__ == "__main__":
    # Run the test
    test_example_from_readme()
