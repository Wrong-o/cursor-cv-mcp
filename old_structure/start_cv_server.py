#!/usr/bin/env python3
"""
Start the Cursor CV MCP server manually.
This can be useful for debugging and troubleshooting.
"""

import importlib
import sys
import logging
import threading
import time
import requests
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def verify_server(max_attempts=5, delay=2):
    """Verify the server is running and responding to requests."""
    base_url = "http://127.0.0.1:8001"
    
    # First check health endpoint
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                logger.info("✅ Server health check successful")
                break
        except requests.RequestException:
            logger.info(f"Waiting for server to start (attempt {attempt+1}/{max_attempts})...")
            time.sleep(delay)
    else:
        logger.error("❌ Failed to connect to server after multiple attempts")
        return False
    
    # Test screenshot endpoint with both route formats
    success = False
    
    # Try the main route
    try:
        response = requests.post(
            f"{base_url}/mcp_screenshot_capture", 
            json={"monitor": 1, "debug": True},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info(f"✅ Screenshot captured successfully: {result.get('screenshot_path')}")
                success = True
            else:
                logger.error(f"❌ Screenshot capture failed: {result.get('error')}")
        else:
            logger.error(f"❌ /mcp_screenshot_capture endpoint returned status {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"❌ Failed to test screenshot endpoint: {e}")
    
    # Try the namespaced route
    if not success:
        try:
            response = requests.post(
                f"{base_url}/mcp_cv_tool/mcp_screenshot_capture", 
                json={"monitor": 1, "debug": True},
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    logger.info(f"✅ Screenshot captured successfully: {result.get('screenshot_path')}")
                    success = True
                else:
                    logger.error(f"❌ Screenshot capture failed: {result.get('error')}")
            else:
                logger.error(f"❌ /mcp_cv_tool/mcp_screenshot_capture endpoint returned status {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"❌ Failed to test namespaced screenshot endpoint: {e}")
    
    return success

def start_server():
    """Start the Cursor CV MCP server."""
    try:
        # Check if module exists
        module_name = "mcp_cv_tool.server"
        logger.info(f"Trying to import {module_name}")
        server_module = importlib.import_module(module_name)
        
        # Start the server in a separate thread
        logger.info("Starting server...")
        threading.Thread(target=server_module.run_server, daemon=True).start()
        
        # Verify the server is running
        if verify_server():
            logger.info("Server is running correctly")
            
            # Keep the main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Server stopped by user")
        else:
            logger.error("Server verification failed")
            sys.exit(1)
            
    except ModuleNotFoundError:
        logger.error(f"Module {module_name} not found. Is the package installed?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Cursor CV MCP Server Manual Start ===")
    print("This will start the CV server manually for troubleshooting.")
    print("Press Ctrl+C to stop the server.")
    start_server() 