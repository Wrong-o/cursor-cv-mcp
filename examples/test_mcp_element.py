#!/usr/bin/env python3
"""
Test script for MCP element detection.
This script tests the MCP find_element function with various element types.
"""

import os
import sys
import time
import requests
import json
from pprint import pprint

# Server URL
SERVER_URL = "http://localhost:8001"

def call_mcp_function(function_name, params=None):
    """Call an MCP function with parameters."""
    if params is None:
        params = {}
        
    try:
        print(f"Calling {function_name} with params: {json.dumps(params, indent=2)}")
        response = requests.post(
            f"{SERVER_URL}/mcp/call_function",
            json={
                "function_name": function_name,
                "params": params
            },
            timeout=30  # Longer timeout for CV operations
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response status code: {response.status_code}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling MCP function: {e}")
        return None

def open_test_page():
    """Open the test HTML page in the default browser."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(script_dir, "test_page.html")
    html_path = os.path.abspath(html_path)
    
    print(f"Opening test page: {html_path}")
    
    if sys.platform.startswith('linux'):
        subprocess.Popen(['xdg-open', html_path])
    elif sys.platform == 'darwin':  # macOS
        subprocess.Popen(['open', html_path])
    elif sys.platform == 'win32':  # Windows
        subprocess.Popen(['start', html_path], shell=True)
    else:
        print(f"Unsupported platform: {sys.platform}")
        return False
    
    # Wait for browser to open
    time.sleep(3)
    return True

def test_element_detection():
    """Test the MCP element detection functionality."""
    print("\n=== Testing MCP Element Detection ===\n")
    
    # Take a screenshot for element detection
    print("Taking a screenshot...")
    capture_result = call_mcp_function("mcp_screenshot_capture", {
        "debug": True
    })
    
    if not capture_result or not capture_result.get("success"):
        print("Failed to capture screenshot")
        return
    
    screenshot_path = capture_result.get("screenshot_path")
    print(f"Screenshot captured: {screenshot_path}")
    
    # Test 1: Find button
    print("\n--- Test 1: Find button ---")
    button_result = call_mcp_function("mcp_find_element", {
        "screenshot_path": screenshot_path,
        "element_type": "button",
        "confidence": 0.3,
        "debug": True
    })
    
    print("Button detection result:")
    pprint(button_result)
    
    # Test 2: Find text
    print("\n--- Test 2: Find text ---")
    text_result = call_mcp_function("mcp_find_element", {
        "screenshot_path": screenshot_path,
        "text": "Submit",
        "confidence": 0.3,
        "debug": True
    })
    
    print("Text detection result:")
    pprint(text_result)
    
    # Test 3: Find checkbox
    print("\n--- Test 3: Find checkbox ---")
    checkbox_result = call_mcp_function("mcp_find_element", {
        "screenshot_path": screenshot_path,
        "element_type": "checkbox",
        "confidence": 0.3,
        "debug": True
    })
    
    print("Checkbox detection result:")
    pprint(checkbox_result)
    
    print("\nAll MCP element detection tests completed!")

if __name__ == "__main__":
    import subprocess
    print("Starting MCP element detection tests...")
    # Optionally open the test page first
    # open_test_page()
    test_element_detection() 