#!/usr/bin/env python
"""
Test script for detecting input elements in an HTML page.
This script opens a test HTML page and tries to detect input elements.
"""

import os
import sys
import time
import subprocess
import cv2
import numpy as np
from PIL import Image
import pyautogui
import webbrowser
from mcp_cv_tool.automation import find_input

def open_test_page():
    """Open the test HTML page in the default browser."""
    # Get the absolute path of the test page
    test_page_path = os.path.abspath("test_page.html")
    test_page_url = f"file://{test_page_path}"
    
    print(f"Opening test page: {test_page_url}")
    
    # Open the test page in the default browser
    webbrowser.open(test_page_url)
    
    # Give the browser some time to open
    print("Waiting for the browser to open...")
    time.sleep(5)

def find_all_inputs(screenshot_path, min_confidence=0.5):
    """Find all input elements in the screenshot."""
    print(f"Looking for input elements in {screenshot_path}")
    
    # Load the screenshot
    image = cv2.imread(screenshot_path)
    if image is None:
        print(f"ERROR: Could not load image from {screenshot_path}")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Print image dimensions
    print(f"Image dimensions: {gray.shape}")
    
    # Modified input detection for various aspect ratios
    inputs = []
    
    # Try different confidence levels
    for confidence in [0.5, 0.6, 0.7]:
        print(f"\nTesting with confidence: {confidence}")
        
        # Try different aspect ratios to find all kinds of input elements
        aspect_ranges = [
            (2.0, 20.0, 80, 20, 50),  # Standard text inputs
            (1.0, 3.0, 80, 20, 50),   # Shorter inputs
            (1.0, 10.0, 80, 50, 150)  # Larger inputs like textareas
        ]
        
        for aspect_min, aspect_max, min_width, min_height, max_height in aspect_ranges:
            # Customize find_input parameters by directly searching for rectangles
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If the polygon has 4 points, it might be a rectangle/input
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Input fields usually have a specific aspect ratio
                    aspect_ratio = float(w) / h
                    
                    # Check if it matches our current criteria
                    if (aspect_min <= aspect_ratio <= aspect_max and 
                        w >= min_width and 
                        min_height <= h <= max_height):
                        
                        # Calculate confidence
                        aspect_confidence = 1.0 - min(abs(aspect_ratio - (aspect_min + aspect_max) / 2) / (aspect_max - aspect_min), 1.0)
                        size_confidence = min(w / (min_width * 2), 1.0) if w < min_width * 2 else min((min_width * 2) / w, 1.0)
                        input_confidence = aspect_confidence * 0.5 + size_confidence * 0.5
                        
                        if input_confidence >= confidence:
                            # Check if this input overlaps with any existing one
                            is_duplicate = False
                            for existing in inputs:
                                ex, ey, ew, eh = existing['x'], existing['y'], existing['width'], existing['height']
                                # Check for overlap
                                if (x < ex + ew and x + w > ex and 
                                    y < ey + eh and y + h > ey):
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                inputs.append({
                                    'x': x,
                                    'y': y,
                                    'width': w,
                                    'height': h,
                                    'confidence': input_confidence,
                                    'method': 'custom_input_detection',
                                    'center_x': x + w // 2,
                                    'center_y': y + h // 2,
                                    'aspect_ratio': aspect_ratio
                                })
    
    # Sort by confidence
    inputs.sort(key=lambda x: x['confidence'], reverse=True)
    
    return inputs

def test_input_detection():
    """Main test function."""
    # Open the test page
    open_test_page()
    
    # Take a screenshot
    print("Taking a screenshot...")
    screenshot = pyautogui.screenshot()
    screenshot_path = "test_input_detection_screenshot.jpg"
    screenshot.save(screenshot_path)
    print(f"Screenshot saved to {screenshot_path}")
    
    # Find input elements
    inputs = find_all_inputs(screenshot_path)
    
    if not inputs:
        print("No input elements found!")
        return
    
    print(f"\nFound {len(inputs)} input elements:")
    
    # Load the screenshot for highlighting
    image = cv2.imread(screenshot_path)
    output_image = image.copy()
    
    # Highlight all found inputs
    for i, input_elem in enumerate(inputs):
        print(f"\nInput #{i+1}:")
        print(f"  Position: ({input_elem['x']}, {input_elem['y']})")
        print(f"  Dimensions: {input_elem['width']}x{input_elem['height']}")
        print(f"  Confidence: {input_elem['confidence']}")
        print(f"  Aspect Ratio: {input_elem['aspect_ratio']}")
        
        # Use different colors for each input
        color = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255)   # Magenta
        ][i % 5]
        
        # Draw rectangle
        cv2.rectangle(
            output_image, 
            (input_elem['x'], input_elem['y']), 
            (input_elem['x'] + input_elem['width'], input_elem['y'] + input_elem['height']), 
            color, 
            2
        )
        
        # Add label
        cv2.putText(
            output_image,
            f"Input #{i+1}",
            (input_elem['x'], input_elem['y'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
    
    # Save the highlighted image
    highlight_path = "test_input_detection_highlighted.jpg"
    cv2.imwrite(highlight_path, output_image)
    print(f"\nHighlighted image saved to {highlight_path}")

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
    
    test_input_detection() 