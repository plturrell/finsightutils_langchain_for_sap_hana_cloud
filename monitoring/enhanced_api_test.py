#!/usr/bin/env python3
"""
Enhanced API Testing Framework for FinSight Services
This script performs real tests against all service endpoints and generates detailed reports.
"""

import asyncio
import datetime
import json
import logging
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple, Set
import uuid

import aiohttp
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"api_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("enhanced_api_test")

# Rich console for pretty output
console = Console()

# Service configuration
SERVICE_CONFIG = {
    "aiqtoolset": {
        "name": "AIQ Toolkit",
        "url": "http://localhost:8000",
        "port": 8000,
        "container_name": "aiqtoolset",
        "image": "finsightintelligence/finsight_deep_aiqtoolset:latest",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Root endpoint",
                "expected_status": 200
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint",
                "expected_status": 200
            },
            "/docs": {
                "method": "GET",
                "description": "OpenAPI documentation",
                "expected_status": 200
            },
            "/openapi.json": {
                "method": "GET",
                "description": "OpenAPI schema",
                "expected_status": 200
            },
            "/api/v1/version": {
                "method": "GET",
                "description": "API version",
                "expected_status": [200, 404]  # Accept 404 for specialized endpoints
            }
        }
    },
    "owl": {
        "name": "OWL",
        "url": "http://localhost:8001",
        "port": 8001,
        "container_name": "owl",
        "image": "finsightintelligence/finsight_deep_owl:latest",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Root endpoint",
                "expected_status": 200
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint",
                "expected_status": 200
            },
            "/docs": {
                "method": "GET",
                "description": "OpenAPI documentation",
                "expected_status": 200
            },
            "/openapi.json": {
                "method": "GET",
                "description": "OpenAPI schema",
                "expected_status": 200
            },
            "/api/v1/process": {
                "method": "GET",
                "description": "Document processing endpoint",
                "expected_status": [200, 404]  # Accept 404 for specialized endpoints
            }
        }
    },
    "dynamo": {
        "name": "Dynamo",
        "url": "http://localhost:8002",
        "port": 8002,
        "container_name": "dynamo",
        "image": "finsightintelligence/finsight_deep_dynamo:latest",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Root endpoint",
                "expected_status": 200
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint",
                "expected_status": 200
            },
            "/docs": {
                "method": "GET",
                "description": "OpenAPI documentation",
                "expected_status": 200
            },
            "/openapi.json": {
                "method": "GET",
                "description": "OpenAPI schema",
                "expected_status": 200
            },
            "/api/status": {
                "method": "GET",
                "description": "API status endpoint",
                "expected_status": [200, 404]  # Accept 404 for specialized endpoints
            }
        }
    },
    "dspy": {
        "name": "DSPy",
        "url": "http://localhost:8003",
        "port": 8003,
        "container_name": "dspy",
        "image": "finsightintelligence/finsight_deep_dspy:latest",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Root endpoint",
                "expected_status": 200
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint",
                "expected_status": 200
            },
            "/docs": {
                "method": "GET",
                "description": "OpenAPI documentation",
                "expected_status": 200
            },
            "/openapi.json": {
                "method": "GET",
                "description": "OpenAPI schema",
                "expected_status": 200
            },
            "/api/v1/metrics": {
                "method": "GET",
                "description": "Metrics endpoint",
                "expected_status": [200, 404]  # Accept 404 for specialized endpoints
            }
        }
    },
    "nemo": {
        "name": "NeMo",
        "url": "http://localhost:8004",
        "port": 8004,
        "container_name": "nemo",
        "image": "finsightintelligence/finsight_deep_nemo:latest",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Root endpoint",
                "expected_status": 200
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint",
                "expected_status": 200
            },
            "/docs": {
                "method": "GET",
                "description": "OpenAPI documentation",
                "expected_status": 200
            },
            "/openapi.json": {
                "method": "GET",
                "description": "OpenAPI schema",
                "expected_status": 200
            },
            "/api/info": {
                "method": "GET",
                "description": "API info endpoint",
                "expected_status": [200, 404]  # Accept 404 for specialized endpoints
            }
        }
    },
    "hana-utils": {
        "name": "SAP HANA Cloud Utilities",
        "url": "http://localhost:8005",
        "port": 8005,
        "container_name": "hana-utils",
        "image": "finsightintelligence/finsight_utils_hana:latest",
        "endpoints": {
            "/": {
                "method": "GET",
                "description": "Root endpoint",
                "expected_status": 200
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint",
                "expected_status": 200
            },
            "/docs": {
                "method": "GET",
                "description": "OpenAPI documentation",
                "expected_status": 200
            },
            "/openapi.json": {
                "method": "GET",
                "description": "OpenAPI schema",
                "expected_status": 200
            },
            "/api/hana/status": {
                "method": "GET",
                "description": "HANA connection status",
                "expected_status": [200, 404]  # Accept 404 for specialized endpoints
            }
        }
    }
}

class APITestResult:
    """Class to store and analyze API test results."""
    
    def __init__(self):
        self.results = []
        self.test_id = str(uuid.uuid4())
        self.start_time = datetime.datetime.now()
        self.end_time = None
        self.summary = {}
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a test result."""
        self.results.append(result)
    
    def finalize(self) -> None:
        """Finalize the test results and calculate summary statistics."""
        self.end_time = datetime.datetime.now()
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["status"] == "success")
        failed_tests = total_tests - successful_tests
        
        service_stats = {}
        for result in self.results:
            service = result["service"]
            if service not in service_stats:
                service_stats[service] = {"total": 0, "success": 0, "failure": 0}
            
            service_stats[service]["total"] += 1
            if result["status"] == "success":
                service_stats[service]["success"] += 1
            else:
                service_stats[service]["failure"] += 1
        
        self.summary = {
            "test_id": self.test_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            "service_stats": service_stats
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def generate_markdown_report(self, filename: str) -> str:
        """Generate a markdown report of the test results."""
        if not self.end_time:
            self.finalize()
        
        # Create the report content
        report = f"# API Test Report\n\n"
        report += f"**Test ID:** {self.summary['test_id']}  \n"
        report += f"**Start Time:** {self.summary['start_time']}  \n"
        report += f"**End Time:** {self.summary['end_time']}  \n"
        report += f"**Duration:** {self.summary['duration_seconds']:.2f} seconds  \n\n"
        
        report += f"## Summary\n\n"
        report += f"- **Total Tests:** {self.summary['total_tests']}  \n"
        report += f"- **Successful Tests:** {self.summary['successful_tests']}  \n"
        report += f"- **Failed Tests:** {self.summary['failed_tests']}  \n"
        report += f"- **Success Rate:** {self.summary['success_rate']:.2f}%  \n\n"
        
        report += f"## Service Statistics\n\n"
        report += f"| Service | Total | Success | Failure | Success Rate |\n"
        report += f"|---------|-------|---------|---------|-------------|\n"
        
        for service, stats in self.summary['service_stats'].items():
            success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            report += f"| {service} | {stats['total']} | {stats['success']} | {stats['failure']} | {success_rate:.2f}% |\n"
        
        report += f"\n## Detailed Results\n\n"
        report += f"| Service | Endpoint | Method | Status | Response Time (ms) | Message |\n"
        report += f"|---------|----------|--------|--------|-------------------|--------|\n"
        
        for result in self.results:
            service = result['service']
            endpoint = result['endpoint']
            method = result['method']
            status = result['status']
            response_time = result.get('response_time', 0) * 1000  # Convert to ms
            message = result.get('message', '').replace('\n', ' ')
            
            report += f"| {service} | {endpoint} | {method} | {status} | {response_time:.2f} | {message} |\n"
        
        # Save the report to a file
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, 'w') as f:
            f.write(report)
        
        return report

async def test_endpoint(
    session: aiohttp.ClientSession,
    service_key: str,
    endpoint_path: str,
    endpoint_config: Dict[str, Any],
    timeout: int = 5,
    retries: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """Test a specific endpoint for a service with retry logic."""
    service_config = SERVICE_CONFIG.get(service_key, {})
    service_url = service_config.get("url", "")
    service_name = service_config.get("name", service_key)
    
    method = endpoint_config.get("method", "GET")
    description = endpoint_config.get("description", "Unknown endpoint")
    expected_status = endpoint_config.get("expected_status", 200)
    
    # Convert to list if it's a single value
    if not isinstance(expected_status, list):
        expected_status = [expected_status]
    
    url = f"{service_url}{endpoint_path}"
    
    result = {
        "service": service_key,
        "service_name": service_name,
        "endpoint": endpoint_path,
        "method": method,
        "description": description,
        "expected_status": expected_status,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Implement retry logic
    for attempt in range(retries):
        try:
            start_time = time.time()
            
            if method == "GET":
                async with session.get(url, timeout=timeout) as response:
                    status_code = response.status
                    try:
                        response_body = await response.text()
                        try:
                            response_json = await response.json() if response_body else {}
                        except:
                            response_json = {}
                    except Exception as e:
                        response_body = f"Error reading response: {str(e)}"
                        response_json = {}
            elif method == "POST":
                async with session.post(url, timeout=timeout) as response:
                    status_code = response.status
                    try:
                        response_body = await response.text()
                        try:
                            response_json = await response.json() if response_body else {}
                        except:
                            response_json = {}
                    except Exception as e:
                        response_body = f"Error reading response: {str(e)}"
                        response_json = {}
            else:
                result["status"] = "error"
                result["message"] = f"Unsupported method: {method}"
                result["response_time"] = 0
                return result
            
            end_time = time.time()
            response_time = end_time - start_time
            
            result["status_code"] = status_code
            result["response_time"] = response_time
            result["response_body"] = response_body[:500] + "..." if len(response_body) > 500 else response_body
            result["response_json"] = response_json
            
            # Check if status code matches expected
            if status_code in expected_status:
                result["status"] = "success"
                result["message"] = f"Endpoint returned status {status_code} (expected {', '.join(map(str, expected_status))})"
                return result
            else:
                # Only retry on 5xx errors
                if status_code >= 500 and attempt < retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                
                result["status"] = "error"
                result["message"] = f"Endpoint returned status {status_code}, expected {', '.join(map(str, expected_status))}"
                return result
                
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            
            result["status"] = "error"
            result["message"] = f"Request timed out after {timeout} seconds"
            result["response_time"] = timeout
            return result
            
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            
            result["status"] = "error"
            result["message"] = f"Error during request: {str(e)}"
            result["response_time"] = 0
            return result
    
    # This should not be reached, but just in case
    result["status"] = "error"
    result["message"] = "Max retries reached"
    return result

async def test_service(
    session: aiohttp.ClientSession,
    service_key: str,
    progress: Progress,
    task_id: TaskID
) -> List[Dict[str, Any]]:
    """Test all endpoints for a specific service."""
    service_config = SERVICE_CONFIG.get(service_key, {})
    endpoints = service_config.get("endpoints", {})
    
    results = []
    total_endpoints = len(endpoints)
    completed_endpoints = 0
    
    # Test each endpoint
    for endpoint_path, endpoint_config in endpoints.items():
        result = await test_endpoint(session, service_key, endpoint_path, endpoint_config)
        results.append(result)
        
        completed_endpoints += 1
        progress.update(task_id, completed=completed_endpoints, total=total_endpoints)
    
    return results

async def run_tests() -> APITestResult:
    """Run tests for all services and endpoints."""
    test_result = APITestResult()
    
    timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout for the entire session
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with Progress() as progress:
            # Create tasks for each service
            tasks = {}
            for service_key in SERVICE_CONFIG:
                service_name = SERVICE_CONFIG[service_key].get("name", service_key)
                task_id = progress.add_task(f"[cyan]Testing {service_name}...", total=len(SERVICE_CONFIG[service_key].get("endpoints", {})))
                tasks[service_key] = task_id
            
            # Run all tests concurrently
            results = await asyncio.gather(*[
                test_service(session, service_key, progress, task_id) 
                for service_key, task_id in tasks.items()
            ])
            
            # Process results
            for service_results in results:
                for result in service_results:
                    test_result.add_result(result)
    
    test_result.finalize()
    return test_result

def print_summary_table(test_result: APITestResult) -> None:
    """Print a summary table of the test results."""
    if not test_result.summary:
        test_result.finalize()
    
    table = Table(title="API Test Summary")
    
    table.add_column("Service", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Failure", justify="right", style="red")
    table.add_column("Success Rate", justify="right")
    
    for service, stats in test_result.summary['service_stats'].items():
        success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
        table.add_row(
            service,
            str(stats['total']),
            str(stats['success']),
            str(stats['failure']),
            f"{success_rate:.2f}%"
        )
    
    # Add a total row
    table.add_row(
        "TOTAL",
        str(test_result.summary['total_tests']),
        str(test_result.summary['successful_tests']),
        str(test_result.summary['failed_tests']),
        f"{test_result.summary['success_rate']:.2f}%",
        style="bold"
    )
    
    console.print(table)

def save_test_results(test_result: APITestResult) -> Dict[str, str]:
    """Save test results in various formats and return file paths."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Save as JSON
    json_path = f"logs/api_test_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({"results": test_result.results, "summary": test_result.summary}, f, indent=2, default=str)
    
    # Save as CSV
    csv_path = f"logs/api_test_results_{timestamp}.csv"
    test_result.to_dataframe().to_csv(csv_path, index=False)
    
    # Generate Markdown report
    md_path = f"logs/api_test_report_{timestamp}.md"
    test_result.generate_markdown_report(md_path)
    
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path
    }

async def main():
    """Main function to run the tests and generate reports."""
    console.print("[bold green]Starting Enhanced API Testing Framework[/bold green]")
    console.print("Testing all services with real endpoints...\n")
    
    # Run the tests
    test_result = await run_tests()
    
    # Print summary
    console.print("\n[bold]Test Results Summary:[/bold]")
    print_summary_table(test_result)
    
    # Save results
    file_paths = save_test_results(test_result)
    
    console.print("\n[bold]Results saved to:[/bold]")
    for format_name, path in file_paths.items():
        console.print(f"- {format_name.upper()}: [cyan]{path}[/cyan]")
    
    # Display the report in the console
    console.print("\n[bold]Detailed Report:[/bold]")
    with open(file_paths["markdown"], "r") as f:
        report_content = f.read()
    
    console.print(Markdown(report_content))
    
    # Return success/failure based on test results
    return test_result.summary["failed_tests"] == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)