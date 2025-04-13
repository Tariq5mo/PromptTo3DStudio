#!/usr/bin/env python3
import unittest
import sys
import os

# Add the parent directory to the path so imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_all_tests():
    """Run all unit and integration tests"""
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests and return the result
    result = test_runner.run(test_suite)
    return result

def run_unit_tests():
    """Run only unit tests"""
    # Discover and run unit tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.join(os.path.dirname(__file__), "unit"), pattern="test_*.py")

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests and return the result
    result = test_runner.run(test_suite)
    return result

def run_integration_tests():
    """Run only integration tests"""
    # Discover and run integration tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.join(os.path.dirname(__file__), "integration"), pattern="test_*.py")

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests and return the result
    result = test_runner.run(test_suite)
    return result

if __name__ == "__main__":
    # Check command-line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "unit":
            result = run_unit_tests()
        elif test_type == "integration":
            result = run_integration_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python run_tests.py [unit|integration|all]")
            sys.exit(1)
    else:
        # Default to running all tests
        result = run_all_tests()

    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful())