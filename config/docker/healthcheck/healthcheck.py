#!/usr/bin/env python3
"""
Blue-Green Deployment Health Checker for SAP HANA LangChain Integration

This service monitors the health of blue and green deployments and manages traffic switching
via Traefik for zero-downtime deployments.
"""

import os
import time
import json
import logging
import signal
import sys
import subprocess
import requests
from typing import Dict, Any, Optional, Tuple, List
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bg-healthcheck")

# Configuration from environment
BLUE_URL = os.getenv("BLUE_URL", "http://api-blue:8000/health/status")
GREEN_URL = os.getenv("GREEN_URL", "http://api-green:8000/health/status")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))
TRAEFIK_API_URL = os.getenv("TRAEFIK_API_URL", "http://traefik:8080/api")
SWITCH_THRESHOLD = int(os.getenv("SWITCH_THRESHOLD", "3"))  # Consecutive successful health checks required

# Global state
current_active = "blue"  # Initial active deployment
green_ready_count = 0    # Count of consecutive green health checks
blue_ready_count = 0     # Count of consecutive blue health checks


def check_health(url: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Check the health of a deployment.
    
    Args:
        url: The health check URL
        
    Returns:
        Tuple of (is_healthy, details)
    """
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            details = response.json()
            # Verify essential services are healthy
            if details.get("status") == "healthy" and details.get("gpu", {}).get("status") == "healthy":
                return True, details
        return False, {"error": f"Unhealthy response: {response.status_code}", "data": response.text}
    except Exception as e:
        return False, {"error": str(e)}


def get_current_routing() -> str:
    """
    Get the currently active deployment from Traefik.
    
    Returns:
        Current active deployment color ("blue" or "green")
    """
    try:
        # Get routers from Traefik API
        response = requests.get(f"{TRAEFIK_API_URL}/http/routers", timeout=5)
        if response.status_code != 200:
            logger.error(f"Failed to get routers: {response.status_code} {response.text}")
            return current_active
            
        routers = response.json()
        
        # Find active API router
        for router in routers:
            if router.get("service", "").startswith("api-") and router.get("status") == "enabled":
                color = router.get("service").split("-")[1]
                return color
                
        logger.warning("No active router found, defaulting to blue")
        return "blue"
    except Exception as e:
        logger.error(f"Error getting current routing: {e}")
        return current_active


def switch_traffic(to_color: str) -> bool:
    """
    Switch traffic to the specified deployment.
    
    Args:
        to_color: The deployment color to switch to ("blue" or "green")
        
    Returns:
        Success status
    """
    if to_color not in ["blue", "green"]:
        logger.error(f"Invalid color: {to_color}")
        return False
        
    try:
        client = docker.from_env()
        
        # Get containers
        traefik_container = None
        blue_container = None
        green_container = None
        
        for container in client.containers.list():
            if "traefik" in container.name:
                traefik_container = container
            elif "api-blue" in container.name:
                blue_container = container
            elif "api-green" in container.name:
                green_container = container
                
        if not all([traefik_container, blue_container, green_container]):
            logger.error("Could not find all required containers")
            return False
            
        # Enable target color in Traefik
        if to_color == "blue":
            blue_container.update(labels={"traefik.enable": "true"})
            green_container.update(labels={"traefik.enable": "false"})
        else:
            blue_container.update(labels={"traefik.enable": "false"})
            green_container.update(labels={"traefik.enable": "true"})
            
        logger.info(f"Switched traffic to {to_color} deployment")
        return True
    except Exception as e:
        logger.error(f"Error switching traffic: {e}")
        return False


def manage_deployment():
    """Main deployment management function."""
    global current_active, green_ready_count, blue_ready_count
    
    # Get current state
    current_active = get_current_routing()
    logger.info(f"Current active deployment: {current_active}")
    
    # Check health of both deployments
    blue_healthy, blue_details = check_health(BLUE_URL)
    green_healthy, green_details = check_health(GREEN_URL)
    
    # Log status
    logger.info(f"Blue deployment: {'Healthy' if blue_healthy else 'Unhealthy'}")
    logger.info(f"Green deployment: {'Healthy' if green_healthy else 'Unhealthy'}")
    
    # Update ready counters
    blue_ready_count = blue_ready_count + 1 if blue_healthy else 0
    green_ready_count = green_ready_count + 1 if green_healthy else 0
    
    # Handle blue-green switching logic
    if current_active == "blue" and green_healthy:
        # If green is consistently healthy and has a newer version, switch to it
        if green_ready_count >= SWITCH_THRESHOLD:
            # Check if green has newer version
            blue_version = blue_details.get("version", "0.0.0")
            green_version = green_details.get("version", "0.0.0")
            
            if green_version > blue_version:
                logger.info(f"Green deployment consistently healthy with newer version ({green_version} > {blue_version})")
                if switch_traffic("green"):
                    current_active = "green"
                    # Reset counters
                    blue_ready_count = 0
                    green_ready_count = 0
    
    elif current_active == "green" and blue_healthy:
        # If blue is consistently healthy and has a newer version, switch to it
        if blue_ready_count >= SWITCH_THRESHOLD:
            # Check if blue has newer version
            blue_version = blue_details.get("version", "0.0.0")
            green_version = green_details.get("version", "0.0.0")
            
            if blue_version > green_version:
                logger.info(f"Blue deployment consistently healthy with newer version ({blue_version} > {green_version})")
                if switch_traffic("blue"):
                    current_active = "blue"
                    # Reset counters
                    blue_ready_count = 0
                    green_ready_count = 0
    
    # Failover logic - if active deployment is unhealthy but other is healthy
    if current_active == "blue" and not blue_healthy and green_healthy and green_ready_count >= 1:
        logger.warning("Failover: Blue deployment unhealthy, switching to green")
        if switch_traffic("green"):
            current_active = "green"
            blue_ready_count = 0
            green_ready_count = 0
    
    elif current_active == "green" and not green_healthy and blue_healthy and blue_ready_count >= 1:
        logger.warning("Failover: Green deployment unhealthy, switching to blue")
        if switch_traffic("blue"):
            current_active = "blue"
            blue_ready_count = 0
            green_ready_count = 0


def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info("Received termination signal, shutting down...")
    sys.exit(0)


def main():
    """Main function."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting Blue-Green deployment health checker")
    
    while True:
        try:
            manage_deployment()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()