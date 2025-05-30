#!/usr/bin/env python3
"""
Test script to directly check the FastAPI routes.
"""

import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_routes():
    """Test the FastAPI routes."""
    try:
        # Import the module
        from mcp_cv_tool.server import app
        
        # Print all registered routes
        logger.info("Registered routes:")
        for route in app.routes:
            logger.info(f"  {route.path}")
        
    except Exception as e:
        logger.error(f"Error testing routes: {e}")

if __name__ == "__main__":
    test_routes() 