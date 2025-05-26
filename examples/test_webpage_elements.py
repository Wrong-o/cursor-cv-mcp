#!/usr/bin/env python3
"""
Test script for finding and interacting with elements on a web page.
This script opens the test HTML page in a browser and tries to find and interact with elements.
"""

import os
import sys
import time
import requests
import json
import subprocess
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

def find_element_by_text(text, confidence=0.5):
    """Find an element by its text content."""
    print(f"\nLooking for element with text: '{text}'")
    
    # Capture screenshot
    capture_result = call_mcp_function("mcp_screenshot_capture", {
        "debug": True
    })
    
    if not capture_result or not capture_result.get("success"):
        print("Failed to capture screenshot")
        return None
    
    screenshot_path = capture_result.get("screenshot_path")
    print(f"Screenshot captured: {screenshot_path}")
    
    # Find element by text
    find_result = call_mcp_function("mcp_find_element", {
        "screenshot_path": screenshot_path,
        "text": text,
        "confidence": confidence,
        "debug": True
    })
    
    if not find_result or not find_result.get("success"):
        print(f"Failed to find element with text '{text}':")
        pprint(find_result)
        return None
    
    element = find_result.get("element", {})
    print(f"Found element with text '{text}':")
    pprint(element)
    return element

def find_element_by_type(element_type, confidence=0.5):
    """Find an element by its type."""
    print(f"\nLooking for element of type: '{element_type}'")
    
    # Capture screenshot
    capture_result = call_mcp_function("mcp_screenshot_capture", {
        "debug": True
    })
    
    if not capture_result or not capture_result.get("success"):
        print("Failed to capture screenshot")
        return None
    
    screenshot_path = capture_result.get("screenshot_path")
    print(f"Screenshot captured: {screenshot_path}")
    
    # Find element by type
    find_result = call_mcp_function("mcp_find_element", {
        "screenshot_path": screenshot_path,
        "element_type": element_type,
        "confidence": confidence,
        "debug": True
    })
    
    if not find_result or not find_result.get("success"):
        print(f"Failed to find element of type '{element_type}':")
        pprint(find_result)
        return None
    
    element = find_result.get("element", {})
    print(f"Found element of type '{element_type}':")
    pprint(element)
    return element

def click_element(element):
    """Click on an element."""
    if not element:
        print("No element provided to click")
        return False
    
    center_x = element.get("center_x", 0)
    center_y = element.get("center_y", 0)
    
    click_result = call_mcp_function("mcp_mouse_click", {
        "x": center_x,
        "y": center_y
    })
    
    if not click_result or not click_result.get("success"):
        print(f"Failed to click on element at ({center_x}, {center_y})")
        return False
    
    print(f"Successfully clicked on element at ({center_x}, {center_y})")
    return True

def type_in_element(element, text):
    """Click on an element and type text."""
    if not click_element(element):
        return False
    
    # Type text
    type_result = call_mcp_function("mcp_type_text", {
        "text": text
    })
    
    if not type_result or not type_result.get("success"):
        print(f"Failed to type text: '{text}'")
        return False
    
    print(f"Successfully typed text: '{text}'")
    return True

def test_webpage_elements():
    """Test finding and interacting with elements on the test page."""
    print("\n=== Testing Web Page Element Detection ===\n")
    
    # Open the test page
    if not open_test_page():
        print("Failed to open test page")
        return
    
    # Allow time for the page to fully load
    time.sleep(2)
    
    # Test 1: Find and click Submit button by text
    print("\n--- Test 1: Find and click Submit button by text ---")
    submit_element = find_element_by_text("Submit")
    if submit_element:
        click_element(submit_element)
    
    time.sleep(2)  # Wait for any alert dialog
    
    # Test 2: Find and click Reset button by text
    print("\n--- Test 2: Find and click Reset button by text ---")
    reset_element = find_element_by_text("Reset")
    if reset_element:
        click_element(reset_element)
    
    time.sleep(1)
    
    # Test 3: Find and interact with Name input
    print("\n--- Test 3: Find input field and type name ---")
    name_input = find_element_by_type("input")
    if name_input:
        type_in_element(name_input, "John Doe")
    
    time.sleep(1)
    
    # Test 4: Find and check a checkbox
    print("\n--- Test 4: Find and toggle checkbox ---")
    checkbox = find_element_by_type("checkbox")
    if checkbox:
        click_element(checkbox)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    print("Starting webpage element detection tests...")
    test_webpage_elements() 