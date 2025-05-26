"""
Automation module for screen interaction capabilities.
Provides functions to click and input text on the screen.
"""

import time
import subprocess
import os
import sys
from typing import Tuple, Optional, Dict, Any, List

# Primary automation tool - PyAutoGUI
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Computer vision and image processing
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pytesseract
    HAS_CV = True
except ImportError:
    HAS_CV = False

# Fallback options for Linux
USING_FALLBACK = False
HAS_XDOTOOL = False

# Check if xdotool is available (Linux-only fallback)
try:
    if sys.platform.startswith('linux'):
        result = subprocess.run(['which', 'xdotool'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            HAS_XDOTOOL = True
            print("xdotool found, will use as fallback if PyAutoGUI fails")
except Exception:
    pass

def run_xdotool(command):
    """Run an xdotool command and return its output."""
    try:
        full_command = ['xdotool'] + command
        result = subprocess.run(full_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"xdotool error: {result.stderr}")
            return None
        return result.stdout.strip()
    except Exception as e:
        print(f"Error running xdotool: {e}")
        return None

def check_requirements():
    """Check if required dependencies are installed."""
    global USING_FALLBACK
    
    if not HAS_PYAUTOGUI:
        print("PyAutoGUI is not installed. Installing...")
        subprocess.run(["pip", "install", "pyautogui"], check=True)
        print("PyAutoGUI installed successfully.")
    
    if not HAS_CV:
        print("Computer vision dependencies not installed. Installing...")
        subprocess.run(["pip", "install", "opencv-python pytesseract pillow"], check=True)
        print("Computer vision dependencies installed successfully.")
    
    # Check for X11 display
    display = os.environ.get('DISPLAY')
    if not display:
        print("WARNING: DISPLAY environment variable not set. X11 operations may fail.")
        print("Trying to set DISPLAY=:0 as a fallback...")
        os.environ['DISPLAY'] = ':0'
    else:
        print(f"Using DISPLAY={display}")
    
    # Check for GUI environment
    try:
        screen_size = pyautogui.size()
        print(f"Screen size detected: {screen_size.width}x{screen_size.height}")
    except Exception as e:
        print(f"WARNING: Could not get screen size: {e}")
        print("This likely means PyAutoGUI cannot access the X11 display.")
        print("Try running the server with 'export DISPLAY=:0' first.")
        
        # Try fallback method
        if HAS_XDOTOOL:
            print("Attempting to use xdotool as a fallback...")
            USING_FALLBACK = True
        else:
            print("No fallback available. Install xdotool: sudo apt-get install xdotool")

def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
    """
    Click at a specific screen position.
    
    Args:
        x: X coordinate
        y: Y coordinate
        button: Mouse button ('left', 'right', 'middle')
        clicks: Number of clicks
        
    Returns:
        Success status
    """
    check_requirements()
    
    # Try xdotool fallback if PyAutoGUI failed to initialize
    if USING_FALLBACK and HAS_XDOTOOL:
        try:
            # Move mouse
            print(f"Using xdotool to move to ({x}, {y})...")
            run_xdotool(['mousemove', str(x), str(y)])
            
            # Click
            button_arg = {'left': 1, 'middle': 2, 'right': 3}.get(button, 1)
            for _ in range(clicks):
                run_xdotool(['click', str(button_arg)])
                time.sleep(0.1)
            
            print(f"Clicked at position ({x}, {y}) with {button} button using xdotool")
            return True
        except Exception as e:
            print(f"Error using xdotool: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Use PyAutoGUI if available
    try:
        # Get screen size
        screen_width, screen_height = pyautogui.size()
        print(f"Screen size: {screen_width}x{screen_height}")
        
        # Get current position
        current_x, current_y = pyautogui.position()
        print(f"Current mouse position before move: ({current_x}, {current_y})")
        
        # Validate coordinates
        if x < 0 or x >= screen_width or y < 0 or y >= screen_height:
            print(f"Warning: Coordinates ({x}, {y}) are outside screen bounds {screen_width}x{screen_height}")
            # Continue anyway - this might be for a multi-monitor setup
        
        # Move to position
        print(f"Moving mouse to ({x}, {y})...")
        pyautogui.moveTo(x, y, duration=0.5)
        
        # Verify position after move
        after_x, after_y = pyautogui.position()
        print(f"Mouse position after move: ({after_x}, {after_y})")
        
        # Click
        print(f"Clicking with {button} button, {clicks} times...")
        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        
        print(f"Clicked at position ({x}, {y}) with {button} button")
        return True
    except Exception as e:
        print(f"Error clicking at position ({x}, {y}): {e}")
        import traceback
        traceback.print_exc()
        return False

def type_text(text: str, interval: float = 0.05) -> bool:
    """
    Type text at the current cursor position.
    
    Args:
        text: Text to type
        interval: Time interval between keystrokes
        
    Returns:
        Success status
    """
    check_requirements()
    
    # Try xdotool fallback if PyAutoGUI failed to initialize
    if USING_FALLBACK and HAS_XDOTOOL:
        try:
            print(f"Using xdotool to type: '{text[:20]}{'...' if len(text) > 20 else ''}'")
            run_xdotool(['type', text])
            print(f"Typed text using xdotool")
            return True
        except Exception as e:
            print(f"Error using xdotool: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Use PyAutoGUI if available
    try:
        # Get current position for reference
        current_x, current_y = pyautogui.position()
        print(f"Current mouse position: ({current_x}, {current_y})")
        
        # Type text
        print(f"Typing text with interval {interval}s: '{text[:20]}{'...' if len(text) > 20 else ''}'")
        pyautogui.typewrite(text, interval=interval)
        
        print(f"Typed text: '{text[:20]}{'...' if len(text) > 20 else ''}'")
        return True
    except Exception as e:
        print(f"Error typing text: {e}")
        import traceback
        traceback.print_exc()
        return False

def press_key(key: str) -> bool:
    """
    Press a specific key.
    
    Args:
        key: Key to press (e.g., 'enter', 'tab', 'esc')
        
    Returns:
        Success status
    """
    check_requirements()
    
    # Try xdotool fallback if PyAutoGUI failed to initialize
    if USING_FALLBACK and HAS_XDOTOOL:
        try:
            print(f"Using xdotool to press key: {key}")
            # Map some common keys
            key_map = {
                'enter': 'Return',
                'tab': 'Tab',
                'esc': 'Escape',
                'space': 'space'
            }
            xdotool_key = key_map.get(key.lower(), key)
            run_xdotool(['key', xdotool_key])
            print(f"Pressed key using xdotool: {key}")
            return True
        except Exception as e:
            print(f"Error using xdotool: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Use PyAutoGUI if available
    try:
        # Press key
        pyautogui.press(key)
        
        print(f"Pressed key: {key}")
        return True
    except Exception as e:
        print(f"Error pressing key {key}: {e}")
        return False

def find_and_click_element(
    screenshot_path: str, 
    element_type: str = "button",
    text: Optional[str] = None,
    confidence: float = 0.7
) -> bool:
    """
    Find an element in a screenshot and click it.
    
    Args:
        screenshot_path: Path to screenshot image
        element_type: Type of element ('button', 'link', 'input', etc.)
        text: Text on or near the element
        confidence: Confidence threshold (0.0-1.0)
        
    Returns:
        Success status
    """
    check_requirements()
    
    try:
        # This would require image recognition capabilities
        # For a basic implementation, we could use template matching
        # or OCR to find elements based on visual appearance or text
        
        # For now, just simulate finding an element
        print(f"Searching for {element_type} with text '{text}'...")
        print("Element recognition not yet implemented - this would require computer vision.")
        print("Please provide coordinates for clicking instead.")
        
        return False
    except Exception as e:
        print(f"Error finding element: {e}")
        return False

def get_screen_position() -> Tuple[int, int]:
    """
    Get current mouse position.
    
    Returns:
        Tuple of (x, y) coordinates
    """
    check_requirements()
    
    try:
        x, y = pyautogui.position()
        print(f"Current mouse position: ({x}, {y})")
        return (x, y)
    except Exception as e:
        print(f"Error getting mouse position: {e}")
        return (0, 0)

def highlight_area(x: int, y: int, width: int, height: int, duration: float = 1.0) -> bool:
    """
    Highlight an area on the screen for debugging purposes.
    
    Args:
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner
        width: Width of area
        height: Height of area
        duration: Duration of highlight in seconds
        
    Returns:
        Success status
    """
    check_requirements()
    
    try:
        # This would require drawing on screen capability
        # For now, we'll just move the mouse around the perimeter
        
        # Top edge
        pyautogui.moveTo(x, y, duration=duration/4)
        # Right edge
        pyautogui.moveTo(x + width, y, duration=duration/4)
        # Bottom edge
        pyautogui.moveTo(x + width, y + height, duration=duration/4)
        # Left edge
        pyautogui.moveTo(x, y + height, duration=duration/4)
        # Back to start
        pyautogui.moveTo(x, y, duration=duration/4)
        
        print(f"Highlighted area at ({x}, {y}) with size {width}x{height}")
        return True
    except Exception as e:
        print(f"Error highlighting area: {e}")
        return False

def find_element(
    screenshot_path: str = None,
    element_type: str = None,
    text: str = None,
    reference_image: str = None,
    search_area: List[int] = None,
    confidence: float = 0.7,
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Find a UI element on the screen using computer vision techniques.
    
    Args:
        screenshot_path: Path to screenshot image (if None, a new screenshot will be taken)
        element_type: Type of element to find ('button', 'input', 'checkbox', etc.)
        text: Text content to match (using OCR)
        reference_image: Path to reference image of the element
        search_area: [x, y, width, height] to limit search area
        confidence: Confidence threshold (0.0-1.0)
        debug: Enable debug mode
        
    Returns:
        Dictionary with element information or None if not found
    """
    check_requirements()
    
    if not HAS_CV:
        print("Computer vision dependencies not installed. Cannot perform element detection.")
        return None
    
    if not any([element_type, text, reference_image]):
        print("ERROR: At least one of element_type, text, or reference_image must be provided")
        return None
    
    try:
        # Get screenshot if not provided
        if screenshot_path is None:
            # Take a screenshot
            screenshot = pyautogui.screenshot()
            screenshot_path = "temp_screenshot.png"
            screenshot.save(screenshot_path)
            print(f"Captured screenshot: {screenshot_path}")
        
        # Load the screenshot
        image = cv2.imread(screenshot_path)
        if image is None:
            print(f"ERROR: Could not load image from {screenshot_path}")
            return None
        
        # Convert to RGB for better processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply search area if provided
        if search_area and len(search_area) == 4:
            x, y, w, h = search_area
            image_rgb = image_rgb[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            offset_x, offset_y = 0, 0
        
        # Create a copy for visualization if debug is enabled
        if debug:
            debug_image = image_rgb.copy()
        
        # Initialize result
        result = None
        
        # Method 1: Find by reference image (template matching)
        if reference_image:
            result = find_by_template(image_rgb, reference_image, confidence, debug)
            if result:
                result['x'] += offset_x
                result['y'] += offset_y
                print(f"Found element by template matching at ({result['x']}, {result['y']})")
                return result
        
        # Method 2: Find by text using OCR
        if text:
            result = find_by_text(image_rgb, text, confidence, debug)
            if result:
                result['x'] += offset_x
                result['y'] += offset_y
                print(f"Found element by text '{text}' at ({result['x']}, {result['y']})")
                return result
        
        # Method 3: Find by element type (heuristics and feature detection)
        if element_type:
            result = find_by_element_type(image_rgb, element_type, confidence, debug)
            if result:
                result['x'] += offset_x
                result['y'] += offset_y
                print(f"Found element of type '{element_type}' at ({result['x']}, {result['y']})")
                return result
        
        print("Element not found")
        return None
    
    except Exception as e:
        print(f"Error finding element: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temporary screenshot if we created one
        if screenshot_path == "temp_screenshot.png" and os.path.exists(screenshot_path):
            os.remove(screenshot_path)

def find_by_template(image: np.ndarray, template_path: str, confidence: float, debug: bool) -> Optional[Dict[str, Any]]:
    """Find element by template matching."""
    try:
        # Load template image
        template = cv2.imread(template_path)
        if template is None:
            print(f"ERROR: Could not load template image from {template_path}")
            return None
        
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        
        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # Get best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Check if the match is good enough
        if max_val >= confidence:
            # Get template dimensions
            h, w = template.shape[:2]
            
            # Create result
            return {
                'x': max_loc[0],
                'y': max_loc[1],
                'width': w,
                'height': h,
                'confidence': float(max_val),
                'method': 'template',
                'center_x': max_loc[0] + w // 2,
                'center_y': max_loc[1] + h // 2
            }
        
        return None
    except Exception as e:
        print(f"Error in template matching: {e}")
        return None

def find_by_text(image: np.ndarray, text: str, confidence: float, debug: bool) -> Optional[Dict[str, Any]]:
    """Find element by OCR text recognition."""
    try:
        # Convert image for OCR
        pil_image = Image.fromarray(image)
        
        # Perform OCR
        ocr_result = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Search for text
        boxes = []
        for i, word in enumerate(ocr_result['text']):
            # Skip empty results
            if not word.strip():
                continue
                
            # Check if this word matches our target text
            # Using lowercase and partial matching for more robust results
            if text.lower() in word.lower() or word.lower() in text.lower():
                confidence_score = float(ocr_result['conf'][i]) / 100.0
                
                # Only consider matches above confidence threshold
                if confidence_score >= confidence:
                    x = ocr_result['left'][i]
                    y = ocr_result['top'][i]
                    w = ocr_result['width'][i]
                    h = ocr_result['height'][i]
                    
                    boxes.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': confidence_score,
                        'method': 'ocr',
                        'text': word,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    })
        
        # Return the best match
        if boxes:
            # Sort by confidence score, highest first
            boxes.sort(key=lambda b: b['confidence'], reverse=True)
            return boxes[0]
        
        return None
    except Exception as e:
        print(f"Error in OCR text detection: {e}")
        return None

def find_by_element_type(image: np.ndarray, element_type: str, confidence: float, debug: bool) -> Optional[Dict[str, Any]]:
    """Find element by its type using heuristics and feature detection."""
    try:
        # Convert to grayscale for better processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Different detection methods based on element type
        if element_type.lower() == 'button':
            # Look for rectangles with certain aspect ratios
            # This is a simplified approach - real buttons have more complex features
            return find_button(gray, confidence, debug)
            
        elif element_type.lower() == 'checkbox':
            # Look for small squares
            return find_checkbox(gray, confidence, debug)
            
        elif element_type.lower() == 'input':
            # Look for rectangles with certain aspect ratios, often with text fields
            return find_input(gray, confidence, debug)
            
        else:
            print(f"Unsupported element type: {element_type}")
            return None
            
    except Exception as e:
        print(f"Error in element type detection: {e}")
        return None

def find_button(gray: np.ndarray, confidence: float, debug: bool) -> Optional[Dict[str, Any]]:
    """Find button elements in grayscale image."""
    # A simple implementation - detect rectangle shapes
    # Use edge detection and contour finding
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    buttons = []
    for contour in contours:
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
                # This is a heuristic and can be improved
                aspect_confidence = 1.0 - min(abs(aspect_ratio - 3.0) / 3.0, 1.0)
                size_confidence = min(w * h / 5000.0, 1.0)
                rect_confidence = aspect_confidence * 0.6 + size_confidence * 0.4
                
                if rect_confidence >= confidence:
                    buttons.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': rect_confidence,
                        'method': 'button_detection',
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    })
    
    # Return the best match
    if buttons:
        # Sort by confidence score, highest first
        buttons.sort(key=lambda b: b['confidence'], reverse=True)
        return buttons[0]
    
    return None

def find_checkbox(gray: np.ndarray, confidence: float, debug: bool) -> Optional[Dict[str, Any]]:
    """Find checkbox elements in grayscale image."""
    # Look for small squares
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    checkboxes = []
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
                    checkboxes.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': checkbox_confidence,
                        'method': 'checkbox_detection',
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    })
    
    # Return the best match
    if checkboxes:
        # Sort by confidence score, highest first
        checkboxes.sort(key=lambda b: b['confidence'], reverse=True)
        return checkboxes[0]
    
    return None

def find_input(gray: np.ndarray, confidence: float, debug: bool) -> Optional[Dict[str, Any]]:
    """Find input field elements in grayscale image."""
    # Look for rectangles with higher width than height
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    inputs = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If the polygon has 4 points, it might be a rectangle/input
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            
            # Input fields usually have a specific aspect ratio
            aspect_ratio = float(w) / h
            
            # Typical input fields are wider than tall
            if 3.0 <= aspect_ratio <= 20.0 and w >= 100 and 20 <= h <= 50:
                # Calculate confidence
                aspect_confidence = 1.0 - min(abs(aspect_ratio - 10.0) / 10.0, 1.0)
                size_confidence = min(w / 300.0, 1.0) if w < 300 else min(300.0 / w, 1.0)
                input_confidence = aspect_confidence * 0.5 + size_confidence * 0.5
                
                if input_confidence >= confidence:
                    inputs.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': input_confidence,
                        'method': 'input_detection',
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    })
    
    # Return the best match
    if inputs:
        # Sort by confidence score, highest first
        inputs.sort(key=lambda b: b['confidence'], reverse=True)
        return inputs[0]
    
    return None 