"""
Automation module for screen interaction capabilities.
Provides functions to click and input text on the screen.
"""

import time
import subprocess
import os
import sys
from typing import Tuple, Optional, Dict, Any, List, Union

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

# Check if we have our Swedish typing module
try:
    from . import swedish_typing
    HAS_SWEDISH_TYPING = True
except ImportError:
    HAS_SWEDISH_TYPING = False

print(f"DEBUG: automation.py: HAS_SWEDISH_TYPING = {HAS_SWEDISH_TYPING}")

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

# Global variable to store detection steps
detection_steps = []

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

def type_text(text: str, interval: float = 0.05, keyboard_layout: str = "auto", use_xdotool: bool = False) -> bool:
    """
    Type text at the current cursor position.
    
    Args:
        text: Text to type
        interval: Time interval between keystrokes
        keyboard_layout: Keyboard layout to use ("auto", "us", "sv", etc.)
        use_xdotool: Force using xdotool on Linux systems
        
    Returns:
        Success status
    """
    check_requirements()
    
    # Check if text has special characters
    has_special_chars = any(ord(c) > 127 for c in text) or any(c in 'åäöÅÄÖ;:[]{}@$€\\|~_' for c in text)
    is_linux = sys.platform.startswith('linux')
    is_swedish = keyboard_layout == "sv" or (keyboard_layout == "auto" and has_special_chars)
    
    # Use our specialized Swedish typing module if available
    if (has_special_chars or is_swedish) and is_linux and HAS_SWEDISH_TYPING:
        print(f"Text contains special characters or Swedish layout selected, using specialized typing module")
        if swedish_typing.type_special_text(text, interval):
            print(f"Successfully typed with specialized Swedish typing module")
            return True
        print("Specialized typing module failed, falling back to other methods")
    
    # Direct clipboard method for all characters - simple and reliable
    if is_linux and has_special_chars:
        try:
            print(f"Using clipboard method for text with special characters")
            # Create a temporary file with the text
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
                f.write(text)
                temp_path = f.name
            
            # Use xclip to copy text from file to clipboard
            try:
                subprocess.run(['xclip', '-selection', 'clipboard', '-i', temp_path], check=True)
                print("Text copied to clipboard using xclip")
                
                # Wait a moment for clipboard to update
                time.sleep(0.2)
                
                # Paste using keyboard shortcut
                pyautogui.hotkey('ctrl', 'v')
                print("Text pasted using Ctrl+V")
                
                # Remove the temporary file
                os.remove(temp_path)
                return True
            except Exception as clip_e:
                print(f"Error using xclip method: {clip_e}")
                # Safely remove temp_path if it exists and was defined
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError as e_remove:
                        print(f"Error removing temp_path during exception handling: {e_remove}")
                # No return here, fall through to other methods
        except Exception as e:
            print(f"Error using clipboard method: {e}")
            # Continue to other methods
    
    # Try xdotool with Unicode support for Linux as third fallback
    if is_linux and HAS_XDOTOOL and (use_xdotool or has_special_chars):
        try:
            print(f"Using xdotool with Unicode support")
            delay_ms = int(interval * 1000)
            
            # For each character, use Unicode typing
            for char_idx, char in enumerate(text):
                if ord(char) > 127:  # Non-ASCII character
                    # Use xdotool's Unicode typing capability
                    hex_code = hex(ord(char))[2:]
                    subprocess.run(['xdotool', 'key', f'U{hex_code}'], check=True)
                else:
                    # Type regular ASCII character
                    subprocess.run(['xdotool', 'type', char], check=True)
                
                # Apply delay
                if interval > 0 and char_idx < len(text) - 1:
                    time.sleep(interval)
        
            print(f"Typed text using xdotool with Unicode support")
            return True
        except Exception as e:
            print(f"Error using xdotool with Unicode: {e}")
            # Fall back to regular xdotool or PyAutoGUI
    
    # Try regular xdotool method as fourth fallback
    if is_linux and HAS_XDOTOOL and (use_xdotool or keyboard_layout != "us"):
        try:
            print(f"Using standard xdotool as fallback")
            delay_ms = int(interval * 1000)
            run_xdotool(['type', '--clearmodifiers', f'--delay={delay_ms}', text])
            print(f"Typed text using standard xdotool")
            return True
        except Exception as e:
            print(f"Error using standard xdotool: {e}")
            # Fall back to PyAutoGUI
    
    # Final fallback: Use PyAutoGUI character by character
    try:
        print(f"Using PyAutoGUI character by character as final fallback")
        for char in text:
            # Try to write the character
            pyautogui.write(char)
            if interval > 0:
                time.sleep(interval)
        
        print(f"Typed text using PyAutoGUI: '{text[:20]}{'...' if len(text) > 20 else ''}'")
        return True
    except Exception as e:
        print(f"Error typing text with PyAutoGUI: {e}")
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
    debug: bool = False,
    ocr_enabled: bool = True,
    detect_ui: bool = True,
    detect_text: bool = True,
    detect_images: bool = True,
    match_type: str = "contains",
    max_results: int = 1
) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
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
        ocr_enabled: Enable OCR for text detection
        detect_ui: Enable UI element detection
        detect_text: Enable text detection
        detect_images: Enable image detection
        match_type: Type of text matching (exact, contains, starts_with, ends_with, regex)
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with element information, list of dictionaries, or None if not found
    """
    check_requirements()
    
    if not HAS_CV:
        print("Computer vision dependencies not installed. Cannot perform element detection.")
        return None
    
    if not any([element_type, text, reference_image]):
        print("ERROR: At least one of element_type, text, or reference_image must be provided")
        return None
    
    # Store detection steps for debugging
    global detection_steps
    detection_steps = []
    
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
        
        # Print image dimensions for debugging
        print(f"Image dimensions: {image_rgb.shape}")
        
        # Apply search area if provided
        if search_area and len(search_area) == 4:
            x, y, w, h = search_area
            image_rgb = image_rgb[y:y+h, x:x+w]
            offset_x, offset_y = x, y
            print(f"Using search area: ({x}, {y}, {w}, {h})")
        else:
            offset_x, offset_y = 0, 0
        
        # Create a copy for visualization if debug is enabled
        if debug:
            debug_image = image_rgb.copy()
            print("Debug mode enabled, created debug image copy")
        
        # Store all potential elements
        all_elements = []
        
        # Method 1: Find by reference image (template matching)
        if reference_image and detect_images:
            print(f"Attempting to find element by template matching with reference image: {reference_image}")
            
            if debug:
                detection_steps.append({
                    "name": "Template Matching",
                    "description": f"Looking for reference image: {reference_image}",
                    "time_start": time.time()
                })
            
            result = find_by_template(image_rgb, reference_image, confidence, debug)
            
            if result:
                result['x'] += offset_x
                result['y'] += offset_y
                result['type'] = element_type or "image"
                all_elements.append(result)
                print(f"Found element by template matching at ({result['x']}, {result['y']}) with confidence {result['confidence']}")
                
                if debug:
                    detection_steps[-1]["detected_count"] = 1
                    detection_steps[-1]["time_end"] = time.time()
                    detection_steps[-1]["details"] = {"location": f"({result['x']}, {result['y']})", "confidence": result['confidence']}
            else:
                print("Template matching failed to find element")
                
                if debug:
                    detection_steps[-1]["detected_count"] = 0
                    detection_steps[-1]["time_end"] = time.time()
        
        # Method 2: Find by text using OCR
        if text and ocr_enabled and detect_text:
            print(f"Attempting to find element by text content: '{text}' with match type '{match_type}'")
            
            if debug:
                detection_steps.append({
                    "name": "OCR Text Detection",
                    "description": f"Looking for text: '{text}' using match type '{match_type}'",
                    "time_start": time.time()
                })
            
            # Use extended version that supports match_type
            text_elements = find_by_text_extended(image_rgb, text, confidence, debug, match_type)
            
            if text_elements:
                for elem in text_elements:
                    elem['x'] += offset_x
                    elem['y'] += offset_y
                    elem['type'] = element_type or "text"
                    all_elements.append(elem)
                
                print(f"Found {len(text_elements)} elements by text '{text}'")
                
                if debug:
                    detection_steps[-1]["detected_count"] = len(text_elements)
                    detection_steps[-1]["time_end"] = time.time()
                    detection_steps[-1]["details"] = {
                        "match_type": match_type,
                        "elements": [
                            {"location": f"({elem['x']}, {elem['y']})", "confidence": elem['confidence'], "text": elem.get('text', '')}
                            for elem in text_elements[:3]  # First 3 for brevity
                        ]
                    }
            else:
                print(f"OCR failed to find text '{text}'")
                
                if debug:
                    detection_steps[-1]["detected_count"] = 0
                    detection_steps[-1]["time_end"] = time.time()
        
        # Method 3: Find by element type (heuristics and feature detection)
        if element_type and detect_ui:
            print(f"Attempting to find element by type: '{element_type}'")
            
            if debug:
                detection_steps.append({
                    "name": "UI Element Detection",
                    "description": f"Looking for element of type: '{element_type}'",
                    "time_start": time.time()
                })
            
            ui_elements = find_by_element_type_extended(image_rgb, element_type, confidence, debug)
            
            if ui_elements:
                for elem in ui_elements:
                    elem['x'] += offset_x
                    elem['y'] += offset_y
                    elem['type'] = element_type
                    all_elements.append(elem)
                
                print(f"Found {len(ui_elements)} elements of type '{element_type}'")
                
                if debug:
                    detection_steps[-1]["detected_count"] = len(ui_elements)
                    detection_steps[-1]["time_end"] = time.time()
                    detection_steps[-1]["details"] = {
                        "elements": [
                            {"location": f"({elem['x']}, {elem['y']})", "confidence": elem['confidence']}
                            for elem in ui_elements[:3]  # First 3 for brevity
                        ]
                    }
            else:
                print(f"Failed to find element of type '{element_type}'")
        
                if debug:
                    detection_steps[-1]["detected_count"] = 0
                    detection_steps[-1]["time_end"] = time.time()
        
        # Sort all elements by confidence
        all_elements.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        if debug:
            detection_steps.append({
                "name": "Results Summary",
                "description": f"Found {len(all_elements)} potential elements in total",
                "detected_count": len(all_elements),
                "details": {"total_elements": len(all_elements)}
            })
        
        # Return the results
        if not all_elements:
            print("Element not found using any method")
            return None
        elif max_results == 1:
            # Return only the highest confidence element
            return all_elements[0]
        else:
            # Return up to max_results elements
            return all_elements[:max_results]
    
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

def find_by_text_extended(image: np.ndarray, text: str, confidence: float, debug: bool, match_type: str = "contains") -> List[Dict[str, Any]]:
    """Find elements by text content with support for different matching types."""
    if not HAS_TESSERACT:
        print("ERROR: Tesseract OCR is not installed. Cannot perform text detection.")
        return []
    
    try:
        # Create placeholder for results
        results = []
        
        # Convert image for OCR
        image_rgb = image
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Perform OCR on the image
        ocr_data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)
        
        # Process OCR results
        for i in range(len(ocr_data['text'])):
            ocr_text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            # Skip empty or low confidence results
            if not ocr_text or conf < 0:
                continue
                
            # Apply different matching strategies
            match_found = False
            
            if match_type == "exact":
                match_found = ocr_text.lower() == text.lower()
            elif match_type == "contains":
                match_found = text.lower() in ocr_text.lower()
            elif match_type == "starts_with":
                match_found = ocr_text.lower().startswith(text.lower())
            elif match_type == "ends_with":
                match_found = ocr_text.lower().endswith(text.lower())
            elif match_type == "regex":
                import re
                match_found = bool(re.search(text, ocr_text))
            
            if match_found:
                # Calculate match confidence (combination of OCR confidence and text match)
                match_conf = conf / 100.0
                
                # Create result
                x = ocr_data['left'][i]
                y = ocr_data['top'][i]
                w = ocr_data['width'][i]
                h = ocr_data['height'][i]
                    
                # Add padding for better clickability
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w += padding * 2
                h += padding * 2
                
                results.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                    'confidence': float(match_conf),
                        'method': 'ocr',
                    'text': ocr_text,
                        'center_x': x + w // 2,
                        'center_y': y + h // 2
                    })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    except Exception as e:
        print(f"Error in OCR detection: {e}")
        return []

def find_by_element_type_extended(image: np.ndarray, element_type: str, confidence: float, debug: bool) -> List[Dict[str, Any]]:
    """Find elements by type with support for multiple results."""
    try:
        # Convert to grayscale for better processing
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Different detection methods based on element type
        if element_type.lower() == 'button':
            # Look for rectangles with certain aspect ratios
            return find_buttons(gray, confidence, debug)
            
        elif element_type.lower() == 'checkbox':
            # Look for small squares
            return find_checkboxes(gray, confidence, debug)
            
        elif element_type.lower() == 'input':
            # Look for rectangles with certain aspect ratios, often with text fields
            return find_inputs(gray, confidence, debug)
            
        elif element_type.lower() == 'any':
            # Look for any UI element
            buttons = find_buttons(gray, confidence, debug)
            checkboxes = find_checkboxes(gray, confidence, debug)
            inputs = find_inputs(gray, confidence, debug)
            
            # Combine all results
            all_elements = buttons + checkboxes + inputs
            
            # Sort by confidence
            all_elements.sort(key=lambda x: x['confidence'], reverse=True)
            
            return all_elements
            
        else:
            print(f"Unsupported element type: {element_type}")
            return []
            
    except Exception as e:
        print(f"Error in element type detection: {e}")
        return []

# Placeholder for the multi-element detection functions
# These should be implemented based on the existing find_button, find_checkbox, etc.
# and modified to return lists of elements

def find_buttons(gray: np.ndarray, confidence: float, debug: bool) -> List[Dict[str, Any]]:
    """Find all button-like elements in the image."""
    # Enhanced button detection algorithm inspired by test_input_detection.py
    buttons = []
    
    # Create a color version for OCR
    if len(gray.shape) == 2:
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        color_img = gray.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try different button aspect ratios and sizes
    button_ranges = [
        (1.5, 5.0, 30, 15, 60),    # Standard rectangular buttons
        (0.8, 2.0, 30, 20, 100),   # Square-ish buttons
        (2.0, 8.0, 60, 15, 60)     # Wide buttons
    ]
    
    # Apply edge detection with different parameters for better button detection
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
                
                # Allow for slightly imperfect rectangles (4-6 points)
                if 4 <= len(approx) <= 6:
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Calculate aspect ratio
                    aspect_ratio = float(w) / h
                    
                    # Check against all our button criteria ranges
                    for aspect_min, aspect_max, min_width, min_height, max_height in button_ranges:
                        # Check if it matches our current criteria
                        if (aspect_min <= aspect_ratio <= aspect_max and 
                            w >= min_width and 
                            min_height <= h <= max_height):
                            
                            # Calculate confidence metrics
                            aspect_confidence = 1.0 - min(abs(aspect_ratio - (aspect_min + aspect_max) / 2) / (aspect_max - aspect_min), 1.0)
                            size_confidence = min(w * h / 2000.0, 1.0)  # Size factor
                            
                            # Add additional metrics for button shape regularity
                            rect_area = w * h
                            contour_area = cv2.contourArea(contour)
                            area_ratio = contour_area / rect_area if rect_area > 0 else 0
                            shape_confidence = area_ratio if area_ratio <= 1.0 else 1.0/area_ratio
                            
                            # Combine confidence metrics
                            button_confidence = (aspect_confidence * 0.4 + 
                                               size_confidence * 0.3 + 
                                               shape_confidence * 0.3)
                            
                            if button_confidence >= confidence:
                                # Check for duplicates with significant overlap
                                is_duplicate = False
                                for existing in buttons:
                                    ex, ey, ew, eh = existing['x'], existing['y'], existing['width'], existing['height']
                                    # Check for significant overlap (>50%)
                                    overlap_area = max(0, min(ex + ew, x + w) - max(ex, x)) * max(0, min(ey + eh, y + h) - max(ey, y))
                                    if overlap_area > 0.5 * min(ew * eh, w * h):
                                        is_duplicate = True
                                        # If new one has higher confidence, replace the old one
                                        if button_confidence > existing['confidence']:
                                            buttons.remove(existing)
                                            is_duplicate = False
                                        break
                                
                                if not is_duplicate:
                                    # Extract button region for OCR
                                    button_roi = color_img[max(0, y):min(color_img.shape[0], y+h), 
                                                          max(0, x):min(color_img.shape[1], x+w)]
                                    
                                    # Skip empty regions
                                    if button_roi.size == 0:
                                        continue
                                        
                                    # Get text inside button using OCR
                                    button_text = ""
                                    text_confidence = 0.0
                                    
                                    try:
                                        # Convert to RGB for Tesseract
                                        button_roi_rgb = cv2.cvtColor(button_roi, cv2.COLOR_BGR2RGB)
                                        
                                        # Use Tesseract to extract text
                                        ocr_data = pytesseract.image_to_data(button_roi_rgb, output_type=pytesseract.Output.DICT)
                                        
                                        # Process OCR results
                                        ocr_text_segments = []
                                        ocr_confidences = []
                                        
                                        for i in range(len(ocr_data['text'])):
                                            ocr_text = ocr_data['text'][i].strip()
                                            conf = int(ocr_data['conf'][i])
                                            
                                            # Skip empty or low confidence results
                                            if ocr_text and conf > 0:
                                                ocr_text_segments.append(ocr_text)
                                                ocr_confidences.append(conf)
                                        
                                        if ocr_text_segments:
                                            button_text = " ".join(ocr_text_segments)
                                            text_confidence = sum(ocr_confidences) / len(ocr_confidences) / 100.0
                                    except Exception as e:
                                        print(f"OCR error in button text detection: {e}")
                                    
                                    buttons.append({
                                        'x': x,
                                        'y': y,
                                        'width': w,
                                        'height': h,
                                        'confidence': button_confidence,
                                        'method': 'enhanced_button_detection',
                                        'center_x': x + w // 2,
                                        'center_y': y + h // 2,
                                        'aspect_ratio': aspect_ratio,
                                        'text': button_text,
                                        'text_confidence': text_confidence
                                    })
                            break
    
    # Sort by confidence
    buttons.sort(key=lambda b: b['confidence'], reverse=True)
    
    # Limit to top 5 results
    return buttons[:5]

def find_checkboxes(gray: np.ndarray, confidence: float, debug: bool) -> List[Dict[str, Any]]:
    """Find all checkbox-like elements in the image."""
    # Call the existing function for backward compatibility
    checkbox = find_checkbox(gray, confidence, debug)
    if checkbox:
        return [checkbox]
    return []

def find_inputs(gray: np.ndarray, confidence: float, debug: bool) -> List[Dict[str, Any]]:
    """Find all input-like elements in the image."""
    # Call the existing function for backward compatibility
    input_field = find_input(gray, confidence, debug)
    if input_field:
        return [input_field]
    return []

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