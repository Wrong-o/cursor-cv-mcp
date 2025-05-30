#!/usr/bin/env python3
"""
Direct test script for the Cursor CV MCP endpoints.
This bypasses Cursor's MCP mechanism and directly calls the API.
"""

import requests
import json
import time
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_screenshot_endpoint():
    """Test the screenshot endpoint directly."""
    base_url = "http://127.0.0.1:8001"
    
    # Test health endpoint first
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            logger.info("✅ Server is healthy")
        else:
            logger.error(f"❌ Server health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Cannot connect to server: {e}")
        logger.info("   Is the server running? Try running 'python -m mcp_cv_tool.server'")
        return False
    
    # Test the screenshot endpoint
    try:
        data = {
            "monitor": 1,
            "debug": True,
            "analyze": True
        }
        
        logger.info(f"Sending request to /mcp_screenshot_capture: {json.dumps(data)}")
        
        response = requests.post(
            f"{base_url}/mcp_screenshot_capture",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        logger.info(f"Response status: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Response body: {json.dumps(result, indent=2)}")
            
            if result.get("success"):
                logger.info(f"✅ Screenshot captured successfully: {result.get('screenshot_path')}")
                return True
            else:
                logger.error(f"❌ Screenshot capture failed: {result.get('error')}")
                return False
        else:
            logger.error(f"❌ Screenshot endpoint returned status {response.status_code}")
            logger.error(f"Response text: {response.text}")
            return False
    except requests.RequestException as e:
        logger.error(f"❌ Failed to test screenshot endpoint: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Direct Test for Cursor CV MCP ===")
    
    # Start the server if needed
    if len(sys.argv) > 1 and sys.argv[1] == "--start-server":
        import subprocess
        
        logger.info("Starting server...")
        server_process = subprocess.Popen(
            [sys.executable, "-m", "mcp_cv_tool.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give the server time to start
        logger.info("Waiting for server to start...")
        time.sleep(3)
    
    # Test the endpoints
    if test_screenshot_endpoint():
        logger.info("✅ Test completed successfully")
    else:
        logger.error("❌ Test failed")
        sys.exit(1) 