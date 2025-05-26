#!/usr/bin/env python3
"""
Direct element detection test script.
This bypasses the MCP API and directly uses OpenCV to test element detection.
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import pytesseract
import pyautogui
import subprocess
from pprint import pprint

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

def capture_screenshot(output_path):
    """Capture a screenshot and save it to the specified path."""
    screenshot = pyautogui.screenshot()
    screenshot.save(output_path)
    print(f"Screenshot saved to: {output_path}")
    return output_path

def find_buttons(image_path, confidence=0.5):
    """Find button elements in the image."""
    print(f"\nLooking for buttons in: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Save edge detection result for debugging
    cv2.imwrite("edges.jpg", edges)
    print(f"Edge detection saved to: edges.jpg")
    
    # Draw all contours for debugging
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite("contours.jpg", contour_image)
    print(f"Contour image saved to: contours.jpg")
    
    buttons = []
    for i, contour in enumerate(contours):
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the polygon has 4 points, it might be a rectangle/button
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            
            # Buttons usually have a certain aspect ratio
            aspect_ratio = float(w) / h
            
            # Typical buttons have aspect ratios between 1.5 and 5
            if 1.0 <= aspect_ratio <= 5.0 and w >= 30 and h >= 15:
                # Calculate a confidence score based on how "button-like" it is
                aspect_confidence = 1.0 - min(abs(aspect_ratio - 3.0) / 3.0, 1.0)
                size_confidence = min(w * h / 5000.0, 1.0)
                rect_confidence = aspect_confidence * 0.6 + size_confidence * 0.4
                
                if rect_confidence >= confidence:
                    button = {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': rect_confidence,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    }
                    buttons.append(button)
                    
                    # Draw rectangle for debugging
                    cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Save button detection result
    cv2.imwrite("buttons.jpg", contour_image)
    print(f"Button detection saved to: buttons.jpg")
    
    print(f"Found {len(buttons)} potential buttons")
    for i, button in enumerate(buttons):
        print(f"Button {i+1}: ({button['x']}, {button['y']}, {button['width']}x{button['height']}), confidence: {button['confidence']:.2f}")
    
    return buttons

def find_text(image_path, target_text, confidence=0.5):
    """Find text in the image using OCR."""
    print(f"\nLooking for text '{target_text}' in: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []
    
    # Convert to PIL Image for OCR
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Perform OCR
    ocr_result = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    
    # Debug: print all detected text
    print("All detected text:")
    for i, word in enumerate(ocr_result['text']):
        if word.strip():
            conf = int(ocr_result['conf'][i])
            print(f"  '{word}' (conf: {conf}%)")
    
    # Create a debug image
    debug_image = image.copy()
    
    # Search for text
    matches = []
    for i, word in enumerate(ocr_result['text']):
        # Skip empty results
        if not word.strip():
            continue
            
        # Check if this word matches our target text
        # Using lowercase and partial matching for more robust results
        if target_text.lower() in word.lower() or word.lower() in target_text.lower():
            confidence_score = float(ocr_result['conf'][i]) / 100.0
            
            # Only consider matches above confidence threshold
            if confidence_score >= confidence:
                x = ocr_result['left'][i]
                y = ocr_result['top'][i]
                w = ocr_result['width'][i]
                h = ocr_result['height'][i]
                
                match = {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': confidence_score,
                    'text': word,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2
                }
                matches.append(match)
                
                # Draw rectangle for debugging
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(debug_image, word, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save text detection result
    cv2.imwrite("text_detection.jpg", debug_image)
    print(f"Text detection saved to: text_detection.jpg")
    
    print(f"Found {len(matches)} matches for text '{target_text}'")
    for i, match in enumerate(matches):
        print(f"Match {i+1}: '{match['text']}' at ({match['x']}, {match['y']}, {match['width']}x{match['height']}), confidence: {match['confidence']:.2f}")
    
    return matches

def find_checkboxes(image_path, confidence=0.5):
    """Find checkbox elements in the image."""
    print(f"\nLooking for checkboxes in: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use edge detection to find contours
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    checkboxes = []
    debug_image = image.copy()
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the polygon has 4 points, it might be a square/checkbox
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            
            # Checkboxes are usually square
            aspect_ratio = float(w) / h
            
            # Typical checkboxes are square and small
            if 0.8 <= aspect_ratio <= 1.2 and 10 <= w <= 40 and 10 <= h <= 40:
                # Calculate confidence
                aspect_confidence = 1.0 - min(abs(aspect_ratio - 1.0) / 0.5, 1.0)
                size_confidence = 1.0 - min(abs(w - 20) / 30.0, 1.0)
                checkbox_confidence = aspect_confidence * 0.7 + size_confidence * 0.3
                
                if checkbox_confidence >= confidence:
                    checkbox = {
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': checkbox_confidence,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    }
                    checkboxes.append(checkbox)
                    
                    # Draw rectangle for debugging
                    cv2.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Save checkbox detection result
    cv2.imwrite("checkboxes.jpg", debug_image)
    print(f"Checkbox detection saved to: checkboxes.jpg")
    
    print(f"Found {len(checkboxes)} potential checkboxes")
    for i, checkbox in enumerate(checkboxes):
        print(f"Checkbox {i+1}: ({checkbox['x']}, {checkbox['y']}, {checkbox['width']}x{checkbox['height']}), confidence: {checkbox['confidence']:.2f}")
    
    return checkboxes

def mouse_click(x, y):
    """Click at the specified coordinates."""
    print(f"Clicking at ({x}, {y})")
    pyautogui.click(x, y)
    return True

def main():
    """Run direct element detection tests."""
    print("Starting direct element detection tests...")
    
    # Open the test page
    if not open_test_page():
        print("Failed to open test page")
        return
    
    # Allow time for the page to fully load
    time.sleep(2)
    
    # Capture a screenshot
    screenshot_path = capture_screenshot("direct_test.jpg")
    
    # Test 1: Find buttons
    buttons = find_buttons(screenshot_path, confidence=0.3)
    if buttons:
        # Click the first button
        mouse_click(buttons[0]['center_x'], buttons[0]['center_y'])
    
    time.sleep(1)
    
    # Test 2: Find text
    text_matches = find_text(screenshot_path, "Submit", confidence=0.3)
    if text_matches:
        # Click on the text
        mouse_click(text_matches[0]['center_x'], text_matches[0]['center_y'])
    
    time.sleep(1)
    
    # Test 3: Find checkboxes
    checkboxes = find_checkboxes(screenshot_path, confidence=0.3)
    if checkboxes:
        # Click the first checkbox
        mouse_click(checkboxes[0]['center_x'], checkboxes[0]['center_y'])
    
    print("\nAll direct tests completed!")

if __name__ == "__main__":
    main() 