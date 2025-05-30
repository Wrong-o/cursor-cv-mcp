#!/usr/bin/env python
"""
Test script for the find_input function in the automation module.
This script takes a screenshot and attempts to find input elements.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import pyautogui
from mcp_cv_tool.automation import find_input

def test_find_input():
    """Test the find_input function with a screenshot."""
    print("Testing find_input function...")
    
    # Take a screenshot
    print("Taking a screenshot...")
    screenshot = pyautogui.screenshot()
    screenshot_path = "test_find_input_screenshot.jpg"
    screenshot.save(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")
    
    # Load the screenshot as a grayscale image
    image = cv2.imread(screenshot_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Print image dimensions
    print(f"Image dimensions: {gray.shape}")
    
    # Call find_input with different confidence levels
    for confidence in [0.5, 0.6, 0.7, 0.8]:
        print(f"\nTesting with confidence: {confidence}")
        result = find_input(gray, confidence, debug=True)
        
        if result:
            print(f"Found input field at: ({result['x']}, {result['y']})")
            print(f"Dimensions: {result['width']}x{result['height']}")
            print(f"Confidence: {result['confidence']}")
            
            # Highlight the found input in the image
            output_image = image.copy()
            cv2.rectangle(
                output_image, 
                (result['x'], result['y']), 
                (result['x'] + result['width'], result['y'] + result['height']), 
                (0, 255, 0), 
                2
            )
            
            # Save the highlighted image
            highlight_path = f"test_find_input_highlight_{confidence}.jpg"
            cv2.imwrite(highlight_path, output_image)
            print(f"Highlighted image saved to {highlight_path}")
        else:
            print(f"No input field found with confidence {confidence}")

if __name__ == "__main__":
    # Make sure virtual environment is activated
    if not os.environ.get('VIRTUAL_ENV'):
        print("Warning: Virtual environment not detected.")
        print("Activate the virtual environment first with:")
        print("source venv/bin/activate")
    
    # Check for X11 display
    if not os.environ.get('DISPLAY'):
        print("Warning: DISPLAY environment variable not set.")
        print("Setting DISPLAY=:0")
        os.environ['DISPLAY'] = ':0'
    
    test_find_input() 