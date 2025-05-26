#!/usr/bin/env python3
"""
Test script for the element detection functionality.
This demonstrates how to use the enhanced element detection to find and interact with UI elements.
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
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling MCP function: {e}")
        return None

def find_and_click_element(element_type=None, text=None, reference_image=None):
    """Find a UI element and click on it."""
    # First, capture a screenshot
    capture_result = call_mcp_function("mcp_screenshot_capture", {
        "debug": True
    })
    
    if not capture_result or not capture_result.get("success"):
        print("Failed to capture screenshot")
        return False
    
    screenshot_path = capture_result.get("screenshot_path")
    print(f"Screenshot captured: {screenshot_path}")
    
    # Find the element
    find_params = {
        "screenshot_path": screenshot_path,
        "confidence": 0.5,  # Lower threshold for testing
        "debug": True
    }
    
    if element_type:
        find_params["element_type"] = element_type
    if text:
        find_params["text"] = text
    if reference_image:
        find_params["reference_image"] = reference_image
    
    find_result = call_mcp_function("mcp_find_element", find_params)
    
    if not find_result or not find_result.get("success"):
        print("Failed to find element:")
        pprint(find_result)
        return False
    
    # Extract element information
    element = find_result.get("element", {})
    center_x = element.get("center_x", 0)
    center_y = element.get("center_y", 0)
    
    print(f"Found element at ({center_x}, {center_y}):")
    pprint(element)
    
    # Click on the element
    click_result = call_mcp_function("mcp_mouse_click", {
        "x": center_x,
        "y": center_y
    })
    
    if not click_result or not click_result.get("success"):
        print("Failed to click on element")
        return False
    
    print(f"Successfully clicked on element at ({center_x}, {center_y})")
    return True

def test_element_detection():
    """Run a series of element detection tests."""
    print("\n=== Testing Element Detection ===\n")
    
    # Test 1: Find by element type (button)
    print("\n--- Test 1: Find a button ---\n")
    find_and_click_element(element_type="button")
    
    time.sleep(2)
    
    # Test 2: Find by text content
    print("\n--- Test 2: Find element by text ---\n")
    find_and_click_element(text="Submit")  # Change to text that appears on your screen
    
    time.sleep(2)
    
    # Test 3: Find input field and type text
    print("\n--- Test 3: Find and type in input field ---\n")
    if find_and_click_element(element_type="input"):
        # Type some text
        call_mcp_function("mcp_type_text", {
            "text": "This is a test input"
        })
    
    time.sleep(2)
    
    # Test 4: Find checkbox and toggle it
    print("\n--- Test 4: Find and toggle checkbox ---\n")
    find_and_click_element(element_type="checkbox")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    print("Starting element detection tests...")
    test_element_detection() 