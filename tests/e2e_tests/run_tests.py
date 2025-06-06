#!/usr/bin/env python
"""
End-to-end test runner for LangChain SAP HANA integration.

This script discovers and runs all end-to-end tests, generating
a comprehensive report of the results.
"""

import os
import sys
import time
import json
import logging
import argparse
import unittest
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import xmlrunner


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run end-to-end tests for LangChain SAP HANA integration")
    
    parser.add_argument("--backend-url", dest="backend_url", 
                        default=os.environ.get("E2E_BACKEND_URL", "http://localhost:8000"),
                        help="Backend API URL (default: http://localhost:8000)")
    
    parser.add_argument("--api-key", dest="api_key",
                        default=os.environ.get("E2E_API_KEY", "test-api-key"),
                        help="API key for authentication")
    
    parser.add_argument("--run-local", dest="run_local",
                        action="store_true", default=os.environ.get("E2E_RUN_LOCAL", "true").lower() == "true",
                        help="Start a local server for testing")
    
    parser.add_argument("--timeout", dest="timeout",
                        type=int, default=int(os.environ.get("E2E_TEST_TIMEOUT", "30")),
                        help="Test request timeout in seconds")
    
    parser.add_argument("--test-hana", dest="test_hana",
                        action="store_true", default=os.environ.get("E2E_TEST_HANA", "false").lower() == "true",
                        help="Run SAP HANA integration tests")
    
    parser.add_argument("--output-dir", dest="output_dir",
                        default=os.environ.get("E2E_OUTPUT_DIR", "test_results"),
                        help="Directory for test reports")
    
    parser.add_argument("--verbose", "-v", dest="verbose",
                        action="store_true", default=False,
                        help="Enable verbose output")
    
    parser.add_argument("--pattern", dest="pattern",
                        default="test_*.py",
                        help="Pattern for test files to run")
    
    return parser.parse_args()


def setup_environment(args):
    """Set up the test environment based on command line arguments."""
    # Set environment variables for tests
    os.environ["E2E_BACKEND_URL"] = args.backend_url
    os.environ["E2E_API_KEY"] = args.api_key
    os.environ["E2E_RUN_LOCAL"] = str(args.run_local).lower()
    os.environ["E2E_TEST_TIMEOUT"] = str(args.timeout)
    os.environ["E2E_TEST_HANA"] = str(args.test_hana).lower()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Return logger for use in main function
    return logging.getLogger("e2e_test_runner")


def discover_and_run_tests(args, logger):
    """Discover and run tests based on the given pattern."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info(f"Discovering tests in {current_dir} with pattern {args.pattern}")
    test_suite = unittest.defaultTestLoader.discover(
        start_dir=current_dir,
        pattern=args.pattern
    )
    
    # Count test cases
    test_count = sum(1 for _ in iter_tests(test_suite))
    logger.info(f"Found {test_count} test cases")
    
    # Run tests and generate reports
    start_time = time.time()
    
    # XML runner for JUnit-compatible reports
    xml_report_file = os.path.join(args.output_dir, "e2e_test_results.xml")
    
    with open(os.path.join(args.output_dir, "e2e_test_results.txt"), "w") as f:
        # Run tests with both text and XML output
        runner = xmlrunner.XMLTestRunner(
            output=args.output_dir,
            verbosity=2 if args.verbose else 1,
            stream=f
        )
        
        result = runner.run(test_suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Generate summary report
    generate_summary_report(args, result, duration, test_count, logger)
    
    return result


def iter_tests(test_suite):
    """
    Recursively iterate through all test cases in a test suite.
    
    Args:
        test_suite: A TestSuite object
        
    Yields:
        Individual TestCase objects
    """
    for test in test_suite:
        if isinstance(test, unittest.TestSuite):
            yield from iter_tests(test)
        else:
            yield test


def generate_summary_report(args, result, duration, test_count, logger):
    """
    Generate a summary report of the test results.
    
    Args:
        args: Command line arguments
        result: TestResult object
        duration: Test duration in seconds
        test_count: Total number of tests
        logger: Logger instance
    """
    # Create summary data
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "backend_url": args.backend_url,
        "run_local": args.run_local,
        "test_hana": args.test_hana,
        "duration_seconds": duration,
        "total_tests": test_count,
        "successful_tests": test_count - len(result.failures) - len(result.errors) - len(result.skipped),
        "failed_tests": len(result.failures),
        "error_tests": len(result.errors),
        "skipped_tests": len(result.skipped),
        "success_rate": (test_count - len(result.failures) - len(result.errors) - len(result.skipped)) / test_count if test_count > 0 else 0,
    }
    
    # Add details for failures and errors
    if result.failures:
        summary["failures"] = [
            {
                "test": str(test),
                "message": message
            }
            for test, message in result.failures
        ]
    
    if result.errors:
        summary["errors"] = [
            {
                "test": str(test),
                "message": message
            }
            for test, message in result.errors
        ]
    
    # Write JSON report
    json_path = os.path.join(args.output_dir, "e2e_test_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Log summary
    logger.info(f"Test run completed in {duration:.2f} seconds")
    logger.info(f"Total tests: {test_count}")
    logger.info(f"Successful tests: {summary['successful_tests']}")
    logger.info(f"Failed tests: {summary['failed_tests']}")
    logger.info(f"Tests with errors: {summary['error_tests']}")
    logger.info(f"Skipped tests: {summary['skipped_tests']}")
    logger.info(f"Success rate: {summary['success_rate'] * 100:.2f}%")
    logger.info(f"Summary report written to {json_path}")


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up environment and get logger
    logger = setup_environment(args)
    
    logger.info(f"Starting end-to-end tests against {args.backend_url}")
    logger.info(f"Run local server: {args.run_local}")
    logger.info(f"Test HANA integration: {args.test_hana}")
    
    # Run tests
    result = discover_and_run_tests(args, logger)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())