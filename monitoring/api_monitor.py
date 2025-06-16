#!/usr/bin/env python3
"""
API Continuous Monitoring Script for FinSight Services

This script runs the enhanced API testing framework at regular intervals
and keeps track of service health over time.
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Any, Optional, Set
import uuid

# Import the enhanced API testing module
import enhanced_api_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"api_monitor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("api_monitor")

# Monitoring configuration
DEFAULT_INTERVAL = 300  # 5 minutes
DEFAULT_LOG_DIR = "monitor_logs"
DEFAULT_ALERT_THRESHOLD = 80.0  # Alert if success rate falls below this percentage

class APIMonitor:
    """API Monitor for continuous testing of services."""
    
    def __init__(
        self, 
        interval: int = DEFAULT_INTERVAL, 
        log_dir: str = DEFAULT_LOG_DIR,
        alert_threshold: float = DEFAULT_ALERT_THRESHOLD
    ):
        """Initialize the API monitor.
        
        Args:
            interval: Time between test runs in seconds
            log_dir: Directory to store monitoring logs
            alert_threshold: Success rate threshold for alerts (percentage)
        """
        self.interval = interval
        self.log_dir = log_dir
        self.alert_threshold = alert_threshold
        self.running = False
        self.results_history = []
        self.current_run = None
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
    async def run_tests(self) -> enhanced_api_test.APITestResult:
        """Run a single test suite using the enhanced API testing framework."""
        logger.info("Starting API test run...")
        test_result = await enhanced_api_test.run_tests()
        logger.info(f"Test run completed. Success rate: {test_result.summary['success_rate']:.2f}%")
        return test_result
    
    def save_results(self, test_result: enhanced_api_test.APITestResult) -> Dict[str, str]:
        """Save the test results to the log directory."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = os.path.join(self.log_dir, f"api_test_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump({"results": test_result.results, "summary": test_result.summary}, f, indent=2, default=str)
        
        # Generate Markdown report
        md_path = os.path.join(self.log_dir, f"api_test_report_{timestamp}.md")
        test_result.generate_markdown_report(md_path)
        
        # Save a summary of recent test runs
        self.results_history.append({
            "timestamp": timestamp,
            "success_rate": test_result.summary["success_rate"],
            "total_tests": test_result.summary["total_tests"],
            "successful_tests": test_result.summary["successful_tests"],
            "failed_tests": test_result.summary["failed_tests"],
            "service_stats": test_result.summary["service_stats"]
        })
        
        # Keep only the last 50 results
        if len(self.results_history) > 50:
            self.results_history = self.results_history[-50:]
        
        # Save history to a JSON file
        history_path = os.path.join(self.log_dir, "monitor_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=str)
        
        return {
            "json": json_path,
            "markdown": md_path,
            "history": history_path
        }
    
    def check_alerts(self, test_result: enhanced_api_test.APITestResult) -> List[Dict[str, Any]]:
        """Check for any alert conditions and return alert details."""
        alerts = []
        
        # Check overall success rate
        if test_result.summary["success_rate"] < self.alert_threshold:
            alerts.append({
                "level": "ERROR",
                "message": f"Overall success rate {test_result.summary['success_rate']:.2f}% is below threshold of {self.alert_threshold}%",
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        # Check per-service success rates
        for service, stats in test_result.summary["service_stats"].items():
            success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            if success_rate < self.alert_threshold:
                alerts.append({
                    "level": "WARNING",
                    "message": f"Service '{service}' success rate {success_rate:.2f}% is below threshold of {self.alert_threshold}%",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "service": service
                })
        
        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert['message']}")
        
        # Save alerts to a file if there are any
        if alerts:
            alerts_file = os.path.join(self.log_dir, "alerts.json")
            
            # Read existing alerts if the file exists
            existing_alerts = []
            if os.path.exists(alerts_file):
                try:
                    with open(alerts_file, 'r') as f:
                        existing_alerts = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Error reading alerts file, starting with empty alerts")
            
            # Add new alerts and save
            existing_alerts.extend(alerts)
            with open(alerts_file, 'w') as f:
                json.dump(existing_alerts, f, indent=2, default=str)
        
        return alerts
    
    async def monitor_loop(self):
        """Main monitoring loop that runs tests at regular intervals."""
        self.running = True
        
        logger.info(f"Starting API monitoring loop with {self.interval} second interval")
        logger.info(f"Logs will be saved to {self.log_dir}")
        logger.info(f"Alert threshold set to {self.alert_threshold}%")
        
        try:
            while self.running:
                self.current_run = uuid.uuid4()
                
                # Run the tests
                start_time = time.time()
                test_result = await self.run_tests()
                
                # Save results
                file_paths = self.save_results(test_result)
                
                # Check for alerts
                alerts = self.check_alerts(test_result)
                
                # Log summary
                duration = time.time() - start_time
                logger.info(f"Test run completed in {duration:.2f} seconds")
                logger.info(f"Results saved to {file_paths['markdown']}")
                if alerts:
                    logger.warning(f"Generated {len(alerts)} alerts")
                
                # Sleep until the next interval
                next_run = start_time + self.interval
                sleep_time = max(0, next_run - time.time())
                
                if sleep_time > 0:
                    logger.info(f"Sleeping for {sleep_time:.2f} seconds until next run")
                    # Use a cancelable sleep so we can stop cleanly
                    try:
                        await asyncio.sleep(sleep_time)
                    except asyncio.CancelledError:
                        logger.info("Sleep interrupted")
                        break
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.exception(f"Error in monitoring loop: {str(e)}")
        finally:
            self.running = False
            logger.info("Monitoring loop stopped")
    
    def stop(self):
        """Stop the monitoring loop."""
        self.running = False
        logger.info("Stopping monitoring loop")

async def main():
    """Main function to start the API monitor."""
    parser = argparse.ArgumentParser(description="API Continuous Monitoring for FinSight Services")
    parser.add_argument(
        "--interval", 
        type=int, 
        default=DEFAULT_INTERVAL,
        help=f"Time between test runs in seconds (default: {DEFAULT_INTERVAL})"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=DEFAULT_LOG_DIR,
        help=f"Directory to store monitoring logs (default: {DEFAULT_LOG_DIR})"
    )
    parser.add_argument(
        "--alert-threshold", 
        type=float, 
        default=DEFAULT_ALERT_THRESHOLD,
        help=f"Success rate threshold for alerts (default: {DEFAULT_ALERT_THRESHOLD}%)"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run tests once and exit (don't start monitoring loop)"
    )
    
    args = parser.parse_args()
    
    # Create the monitor
    monitor = APIMonitor(
        interval=args.interval,
        log_dir=args.log_dir,
        alert_threshold=args.alert_threshold
    )
    
    # Handle interrupts for clean shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Interrupt received, shutting down...")
        monitor.stop()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    try:
        if args.run_once:
            # Just run the tests once
            test_result = await monitor.run_tests()
            file_paths = monitor.save_results(test_result)
            alerts = monitor.check_alerts(test_result)
            
            print(f"\nTest Results Summary:")
            enhanced_api_test.print_summary_table(test_result)
            
            print(f"\nResults saved to:")
            for format_name, path in file_paths.items():
                print(f"- {format_name.upper()}: {path}")
            
            if alerts:
                print(f"\nAlerts Generated:")
                for alert in alerts:
                    print(f"- {alert['level']}: {alert['message']}")
        else:
            # Start the monitoring loop
            await monitor.monitor_loop()
    finally:
        # Remove signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)

if __name__ == "__main__":
    asyncio.run(main())