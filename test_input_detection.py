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
    test_page_path = os.path.abspath("examples/test_page.html")
    
    # Create the page if it doesn't exist
    if not os.path.exists(test_page_path):
        create_test_page(test_page_path)
    
    test_page_url = f"file://{test_page_path}"
    
    print(f"Opening test page: {test_page_url}")
    
    # Open the test page in the default browser
    webbrowser.open(test_page_url)
    
    # Give the browser some time to open
    print("Waiting for the browser to open...")
    time.sleep(5)

def create_test_page(file_path):
    """Create a test HTML page with input elements."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Input Element Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
        }
        input[type="text"], 
        input[type="email"],
        input[type="password"],
        textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        textarea {
            height: 120px;
            resize: vertical;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Input Element Test Page</h1>
        
        <div class="form-group">
            <label for="name">Name:</label>
            <input type="text" id="name" name="name" placeholder="Enter your name">
        </div>
        
        <div class="form-group">
            <label for="email">Email:</label>
            <input type="email" id="email" name="email" placeholder="Enter your email">
        </div>
        
        <div class="form-group">
            <label for="password">Password:</label>
            <input type="password" id="password" name="password" placeholder="Enter your password">
        </div>
        
        <div class="form-group">
            <label for="message">Message:</label>
            <textarea id="message" name="message" placeholder="Enter your message"></textarea>
        </div>
        
        <div class="form-group">
            <label for="search">Search:</label>
            <input type="text" id="search" name="search" placeholder="Search...">
        </div>
        
        <button type="submit">Submit</button>
    </div>
</body>
</html>"""
    
    with open(file_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created test page at {file_path}")

def find_all_inputs(screenshot_path, min_confidence=0.4):
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
    
    # Enhanced preprocessing for better edge detection
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Modified input detection for various aspect ratios
    inputs = []
    
    # Try different aspect ratios to find all kinds of input elements
    aspect_ranges = [
        (2.0, 30.0, 80, 15, 50),   # Standard text inputs
        (1.0, 5.0, 60, 15, 50),    # Shorter inputs
        (1.0, 15.0, 80, 30, 200)   # Larger inputs like textareas
    ]
    
    # Apply edge detection with different parameters
    for low_threshold in [30, 50, 70]:
        for high_threshold in [100, 150, 200]:
            edges = cv2.Canny(blurred, low_threshold, high_threshold)
            
            # Find contours in the edge-detected image
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # If the polygon has 4 points (or close to it), it might be a rectangle/input
                if 4 <= len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Skip very small rectangles
                    if w < 30 or h < 10:
                        continue
                    
                    # Input fields usually have a specific aspect ratio
                    aspect_ratio = float(w) / h
                    
                    # Check against all our aspect ratio ranges
                    for aspect_min, aspect_max, min_width, min_height, max_height in aspect_ranges:
                        # Check if it matches our current criteria
                        if (aspect_min <= aspect_ratio <= aspect_max and 
                            w >= min_width and 
                            min_height <= h <= max_height):
                            
                            # Calculate confidence
                            aspect_confidence = 1.0 - min(abs(aspect_ratio - (aspect_min + aspect_max) / 2) / (aspect_max - aspect_min), 1.0)
                            size_confidence = min(w / (min_width * 2), 1.0) if w < min_width * 2 else min((min_width * 2) / w, 1.0)
                            input_confidence = aspect_confidence * 0.5 + size_confidence * 0.5
                            
                            if input_confidence >= min_confidence:
                                # Check if this input overlaps with any existing one
                                is_duplicate = False
                                for existing in inputs:
                                    ex, ey, ew, eh = existing['x'], existing['y'], existing['width'], existing['height']
                                    # Check for significant overlap (>50%)
                                    overlap_area = max(0, min(ex + ew, x + w) - max(ex, x)) * max(0, min(ey + eh, y + h) - max(ey, y))
                                    if overlap_area > 0.5 * min(ew * eh, w * h):
                                        is_duplicate = True
                                        # If new one has higher confidence, replace the old one
                                        if input_confidence > existing['confidence']:
                                            inputs.remove(existing)
                                            is_duplicate = False
                                        break
                                
                                if not is_duplicate:
                                    inputs.append({
                                        'x': x,
                                        'y': y,
                                        'width': w,
                                        'height': h,
                                        'confidence': input_confidence,
                                        'method': 'enhanced_input_detection',
                                        'center_x': x + w // 2,
                                        'center_y': y + h // 2,
                                        'aspect_ratio': aspect_ratio
                                    })
    
    # Sort by confidence
    inputs.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Limit to top 5 results
    return inputs[:5]

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