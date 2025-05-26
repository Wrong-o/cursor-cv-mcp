#!/usr/bin/env python3
"""
Direct test of the find_element function without using the MCP API.
This bypasses any potential issues with the API layer.
"""

import os
import sys
import time
import pyautogui
import cv2
from pprint import pprint

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_cv_tool.automation import find_element

def test_direct_find_element():
    """Test the find_element function directly."""
    print("\n=== Direct Testing of find_element ===\n")
    
    # Capture a screenshot
    print("Capturing screenshot...")
    screenshot = pyautogui.screenshot()
    screenshot_path = "direct_test_screenshot.jpg"
    screenshot.save(screenshot_path)
    print(f"Screenshot saved to: {screenshot_path}")
    
    # Test 1: Find button
    print("\n--- Test 1: Find button ---")
    button = find_element(
        screenshot_path=screenshot_path,
        element_type="button",
        confidence=0.3,
        debug=True
    )
    
    print("Button detection result:")
    pprint(button)
    
    # Test 2: Find text
    print("\n--- Test 2: Find text ---")
    text_element = find_element(
        screenshot_path=screenshot_path,
        text="Submit",
        confidence=0.3,
        debug=True
    )
    
    print("Text detection result:")
    pprint(text_element)
    
    # Test 3: Find checkbox
    print("\n--- Test 3: Find checkbox ---")
    checkbox = find_element(
        screenshot_path=screenshot_path,
        element_type="checkbox",
        confidence=0.3,
        debug=True
    )
    
    print("Checkbox detection result:")
    pprint(checkbox)
    
    print("\nAll direct find_element tests completed!")

if __name__ == "__main__":
    print("Starting direct find_element tests...")
    test_direct_find_element() 