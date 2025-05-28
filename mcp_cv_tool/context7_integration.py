"""
Context7-like integration for the screenshot functionality.
This provides a standardized interface similar to Context7 for screenshot capture and analysis.
"""

import json
import os
import sys
import time
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Union

from .screenshot import get_screenshot_with_analysis, analyze_image, get_available_monitors
from .automation import mouse_click, type_text, press_key, get_screen_position, highlight_area, find_element
from . import cv_debug  # Import the new CV debug module

def mcp_screenshot_capture(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to capture a screenshot and analyze it.
    
    Parameters:
    - monitor: Monitor number to capture (default: 0)
    - output_file: Path to save the screenshot (optional)
    - debug: Enable debug mode (default: False)
    - analyze: Perform analysis on the image (default: True)
    - screenshots_dir: Directory to save screenshots (optional)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - screenshot_path: Path to the saved screenshot
      - resolution: Image resolution
      - analysis: Analysis results if requested
    """
    monitor = params.get("monitor", 0)
    output_file = params.get("output_file", None)
    debug = params.get("debug", False)
    analyze_flag = params.get("analyze", True)
    screenshots_dir = params.get("screenshots_dir", None)
    
    # Capture screenshot
    screenshot_path, analysis = get_screenshot_with_analysis(
        monitor=monitor,
        output_file=output_file,
        debug=debug,
        screenshots_dir=screenshots_dir
    )
    
    # Prepare response
    response = {
        "success": screenshot_path is not None,
        "screenshot_path": screenshot_path,
        "resolution": analysis.get("basic_info", {}).get("size", None) if analysis else None,
    }
    
    # Include analysis if requested
    if analyze_flag and analysis:
        response["analysis"] = analysis
    
    return response

def mcp_screenshot_analyze_image(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to analyze an existing image.
    
    Parameters:
    - image_path: Path to the image file to analyze
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - analysis: Analysis results
    """
    image_path = params.get("image_path")
    
    if not image_path or not os.path.exists(image_path):
        return {
            "success": False,
            "error": f"Image file not found: {image_path}",
        }
    
    # Analyze image
    analysis = analyze_image(image_path)
    
    return {
        "success": analysis is not None,
        "analysis": analysis,
    }

def mcp_screenshot_list_monitors(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to list available monitors.
    
    Parameters:
    - None required
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - monitors: List of monitor information
      - primary: Primary monitor index
    """
    try:
        # Use the internal function directly
        monitors_info = get_available_monitors()
        
        if not monitors_info:
            return {
                "success": False,
                "error": "Failed to retrieve monitor information",
            }
        
        return {
            "success": True,
            "monitors": monitors_info.get("monitors", []),
            "primary": monitors_info.get("primary", 0),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error retrieving monitor information: {str(e)}",
        }

def mcp_test_function(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple test function that doesn't use any screenshot functionality.
    
    Parameters:
    - message: A test message (optional)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - message: Echo of the input message or default message
    """
    message = params.get("message", "Hello from MCP test function!")
    
    return {
        "success": True,
        "message": message,
    }

def mcp_mouse_click(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to click at a specific screen position.
    
    Parameters:
    - x: X coordinate (required)
    - y: Y coordinate (required)
    - button: Mouse button ('left', 'right', 'middle') (default: 'left')
    - clicks: Number of clicks (default: 1)
    - monitor: Monitor ID (1-based index) (default: None - uses primary monitor)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - position: [x, y] coordinates where clicked
      - button: Button used for clicking
      - clicks: Number of clicks performed
      - monitor: Monitor that was used
    """
    x = params.get("x")
    y = params.get("y")
    
    if x is None or y is None:
        return {
            "success": False,
            "error": "Missing required parameters: x and y coordinates are required"
        }
    
    button = params.get("button", "left")
    clicks = params.get("clicks", 1)
    monitor = params.get("monitor", None)
    
    try:
        # Try to import the improved mouse control module
        import sys
        import os
        
        # Add the parent directory to the path to find the improved_mouse_control module
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        # Try to import the improved mouse control module
        try:
            from improved_mouse_control import improved_mouse_click
            # Use the improved function
            success = improved_mouse_click(x=x, y=y, button=button, clicks=clicks, monitor=monitor)
            print(f"Used improved_mouse_click at ({x}, {y}) on monitor {monitor}")
        except ImportError:
            # Fall back to the original function
            print("Could not import improved_mouse_click, using fallback")
            success = mouse_click(x=x, y=y, button=button, clicks=clicks)
    except Exception as e:
        # If there's any error, fall back to the original function
        print(f"Error trying to use improved_mouse_click: {e}")
        success = mouse_click(x=x, y=y, button=button, clicks=clicks)
    
    return {
        "success": success,
        "position": [x, y],
        "button": button,
        "clicks": clicks,
        "monitor": monitor
    }

def mcp_type_text(params: Dict[str, Any]) -> Dict[str, Any]:
    """Type text at the current cursor position."""
    result = {"success": False}
    
    # Extract parameters
    text = params.get("text")
    delay = params.get("delay", 0.05)  # Default to 50ms
    keyboard_layout = params.get("keyboard_layout", "auto")  # Auto-detect or user-specified
    use_xdotool = params.get("use_xdotool", False)  # Whether to force using xdotool
    
    # Validate parameters
    if not isinstance(text, str):
        result["error"] = "Invalid or missing 'text' parameter. Expected a string."
        return result
    
    # Convert delay from milliseconds to seconds if needed
    if isinstance(delay, int) and delay > 0:
        delay = delay / 1000.0
    
    if not isinstance(delay, (int, float)) or delay < 0:
        result["error"] = "Invalid 'delay' parameter. Expected a non-negative number."
        return result
    
    try:
        # Check for special characters
        has_special_chars = any(c in 'åäöÅÄÖ;:[]{}@$€\\|~' for c in text)
        
        # Auto-enable xdotool on Linux for special characters
        if has_special_chars and sys.platform.startswith('linux') and not use_xdotool:
            use_xdotool = True
            print("Auto-enabling xdotool for special characters")
        
        # Log the request with more detailed info
        print(f"Typing text with {delay}s interval (Keyboard: {keyboard_layout}, xdotool: {use_xdotool})")
        if has_special_chars:
            print(f"Text contains special characters that may require special handling")
        
        # Type the text using automation module
        success = type_text(
            text=text, 
            interval=delay, 
            keyboard_layout=keyboard_layout,
            use_xdotool=use_xdotool
        )
        
        if success:
            result["success"] = True
            result["message"] = f"Successfully typed text with {delay}s interval using layout: {keyboard_layout}"
            if has_special_chars:
                result["message"] += " (special characters detected)"
        else:
            result["error"] = "Failed to type text. Check console logs for details."
    except Exception as e:
        result["error"] = f"Error typing text: {str(e)}"
        import traceback
        traceback.print_exc()
    
    return result

def mcp_press_key(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to press a specific key.
    
    Parameters:
    - key: Key to press (e.g., 'enter', 'tab', 'esc') (required)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - key: Key that was pressed
    """
    key = params.get("key")
    
    if key is None:
        return {
            "success": False,
            "error": "Missing required parameter: key is required"
        }
    
    success = press_key(key=key)
    
    return {
        "success": success,
        "key": key
    }

def mcp_get_mouse_position(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to get the current mouse position.
    
    Parameters:
    - None required
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - position: [x, y] coordinates of the mouse
    """
    try:
        position = get_screen_position()
        
        # Return in a format that matches our MCPCallResponse model
        return {
            "success": True,
            "position": list(position),
            "message": f"Current position: {position[0]}, {position[1]}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def mcp_highlight_area(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to highlight an area on the screen.
    
    Parameters:
    - x: X coordinate of top-left corner (required)
    - y: Y coordinate of top-left corner (required)
    - width: Width of area (required)
    - height: Height of area (required)
    - duration: Duration of highlight in seconds (default: 1.0)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - area: Details of the highlighted area
    """
    x = params.get("x")
    y = params.get("y")
    width = params.get("width")
    height = params.get("height")
    
    if None in (x, y, width, height):
        return {
            "success": False,
            "error": "Missing required parameters: x, y, width, and height are required"
        }
    
    duration = params.get("duration", 1.0)
    
    success = highlight_area(x=x, y=y, width=width, height=height, duration=duration)
    
    return {
        "success": success,
        "area": {
            "x": x,
            "y": y,
            "width": width,
            "height": height
        },
        "duration": duration
    }

def mcp_find_element(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context7-style function to find UI elements on the screen using computer vision.
    
    Parameters:
    - image_path: Path to screenshot image (optional, will capture a new one if not provided)
    - element_type: Type of element to find (button, input, checkbox) (optional)
    - search_text: Text content to match (optional)
    - match_type: Type of text matching (exact, contains, starts_with, ends_with) (default: contains)
    - reference_image: Path to reference image of element (optional)
    - search_area: Dictionary with x, y, width, height to limit search area (optional)
    - confidence: Confidence threshold (default: 0.7)
    - max_results: Maximum number of results to return (default: 1)
    - ocr_enabled: Enable OCR for text detection (default: True)
    - detect_ui: Enable UI element detection (default: True)
    - detect_text: Enable text detection (default: True)
    - detect_images: Enable image detection (default: True)
    - debug: Enable debug mode (default: False)
    - cv_debug: Enable computer vision debug images (default: False)
    - debug_level: Level of debug information (1-3) (default: 1)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - element: Element information (position, size, confidence)
      - all_elements: List of all detected elements if max_results > 1
      - error: Error message if unsuccessful
      - debug_info: Debug information if debug is enabled
    """
    # Validate parameters
    image_path = params.get("image_path") or params.get("screenshot_path")
    element_type = params.get("element_type")
    search_text = params.get("search_text") or params.get("text")
    match_type = params.get("match_type", "contains")
    reference_image = params.get("reference_image")
    search_area = params.get("search_area")
    confidence = params.get("confidence", 0.7)
    max_results = params.get("max_results", 1)
    debug = params.get("debug", False)
    
    # Advanced options
    ocr_enabled = params.get("ocr_enabled", True)
    detect_ui = params.get("detect_ui", True)
    detect_text = params.get("detect_text", True)
    detect_images = params.get("detect_images", True)
    
    # CV debug options
    cv_debug_enabled = params.get("cv_debug", False)
    debug_level = params.get("debug_level", 1)
    
    # Initialize debug info
    debug_info = {} if debug else None
    
    # Check if at least one search method is provided
    if not any([element_type, search_text, reference_image]):
        return {
            "success": False,
            "error": "At least one of element_type, search_text, or reference_image must be provided"
        }
    
    # Start timer for performance tracking
    start_time = time.time()
    
    # Capture a screenshot if needed
    if not image_path:
        screenshot_result = mcp_screenshot_capture({"debug": debug})
        if not screenshot_result.get("success"):
            return {
                "success": False,
                "error": "Failed to capture screenshot"
            }
        image_path = screenshot_result.get("screenshot_path")
    
    # Call the find_element function with additional parameters
    try:
        # Convert search_area from dictionary to list if provided
        area_list = None
        if search_area:
            if isinstance(search_area, dict):
                area_list = [
                    search_area.get("x", 0),
                    search_area.get("y", 0),
                    search_area.get("width", 0),
                    search_area.get("height", 0)
                ]
            elif isinstance(search_area, list):
                area_list = search_area
        
        # Load the image for CV debugging
        if cv_debug_enabled:
            import cv2
            import numpy as np
            debug_image = cv2.imread(image_path)
            
            # Initialize detection steps list for detailed debugging
            detection_steps = []
        
        # Additional parameters for the advanced element finder
        additional_params = {
            "ocr_enabled": ocr_enabled,
            "detect_ui": detect_ui,
            "detect_text": detect_text,
            "detect_images": detect_images,
            "match_type": match_type,
            "max_results": max_results
        }
        
        # Call the find_element function
        finder_result = find_element(
            screenshot_path=image_path,
            element_type=element_type,
            text=search_text,
            reference_image=reference_image,
            search_area=area_list,
            confidence=confidence,
            debug=debug,
            **additional_params
        )
        
        # Process the result
        if finder_result:
            # Check if it's a single element or a list
            all_elements = []
            main_element = None
            
            if isinstance(finder_result, list):
                all_elements = finder_result
                main_element = finder_result[0] if finder_result else None
            else:
                main_element = finder_result
                all_elements = [finder_result]
            
            if not main_element:
                return {
                    "success": False,
                    "error": "Element not found"
                }
            
            # Process all elements for serialization
            serialized_elements = []
            for element in all_elements:
                # Create a serializable copy of the element dictionary
                element_data = {}
                
                # Check if the element already has a position dict
                if "position" in element:
                    element_data["position"] = element["position"]
                else:
                    # Create position dict from x, y, width, height
                    element_data["position"] = {
                        "x": int(element.get("x", 0)),
                        "y": int(element.get("y", 0)),
                        "width": int(element.get("width", 0)),
                        "height": int(element.get("height", 0))
                    }
                
                # Calculate center coordinates if not present
                if "center_x" not in element_data["position"] and "x" in element_data["position"] and "width" in element_data["position"]:
                    element_data["position"]["center_x"] = element_data["position"]["x"] + element_data["position"]["width"] // 2
                if "center_y" not in element_data["position"] and "y" in element_data["position"] and "height" in element_data["position"]:
                    element_data["position"]["center_y"] = element_data["position"]["y"] + element_data["position"]["height"] // 2
                
                # Copy other properties
                element_data["confidence"] = float(element.get("confidence", 0.0))
                element_data["method"] = str(element.get("method", "unknown"))
                element_data["type"] = str(element.get("type", "unknown"))
                
                # Add text if it's available (for OCR results)
                if "text" in element:
                    element_data["text"] = str(element.get("text", ""))
                
                serialized_elements.append(element_data)
            
            # Calculate elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Prepare the response
            response = {
                "success": True,
                "element": serialized_elements[0],
                "message": f"Found element using method '{serialized_elements[0].get('method')}' with confidence {serialized_elements[0].get('confidence'):.2f}"
            }
            
            # Add all elements if requested
            if max_results > 1:
                response["all_elements"] = serialized_elements
            
            # Generate debug information if requested
            if debug:
                debug_info = {
                    "stats": {
                        "total_time": elapsed_time,
                        "ocr_time": 0,  # Placeholder, will be updated if available
                        "ui_detection_time": 0  # Placeholder, will be updated if available
                    },
                    "detection_steps": []  # Will be populated with detection steps if available
                }
                
                # Add detection steps if available
                if hasattr(automation, "detection_steps"):
                    debug_info["detection_steps"] = automation.detection_steps
                
                response["debug_info"] = debug_info
            
            # Generate CV debug images if requested
            if cv_debug_enabled and debug_image is not None:
                try:
                    print(f"Generating CV debug images with debug_level={debug_level}")
                    
                    # Generate CV debug images
                    cv_debug_images = cv_debug.capture_cv_debug(
                        debug_image, 
                        debug_level=debug_level,
                        elements=serialized_elements
                    )
                    
                    # Add CV debug images to debug info
                    if debug_info is None:
                        debug_info = {}
                    
                    # Log what we got back
                    for category, images in cv_debug_images.items():
                        print(f"CV debug category '{category}': {len(images)} images")
                        
                    debug_info["cv_debug_images"] = cv_debug_images
                    response["debug_info"] = debug_info
                    print("CV debug images successfully added to response")
                except Exception as e:
                    print(f"Error generating CV debug images: {e}")
                    import traceback
                    traceback.print_exc()
            
            return response
        else:
            return {
                "success": False,
                "error": "Element not found"
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": f"Error finding element: {str(e)}"
        }

# Function registry mapping function names to their implementations
FUNCTION_REGISTRY = {
    "mcp_screenshot_capture": mcp_screenshot_capture,
    "mcp_screenshot_analyze_image": mcp_screenshot_analyze_image,
    "mcp_screenshot_list_monitors": mcp_screenshot_list_monitors,
    "mcp_test_function": mcp_test_function,
    "mcp_mouse_click": mcp_mouse_click,
    "mcp_type_text": mcp_type_text,
    "mcp_press_key": mcp_press_key,
    "mcp_get_mouse_position": mcp_get_mouse_position,
    "mcp_highlight_area": mcp_highlight_area,
    "mcp_find_element": mcp_find_element,
}

def call_function(function_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call a function by name with the given parameters.
    
    Args:
        function_name: Name of the function to call
        params: Parameters to pass to the function
        
    Returns:
        Result of the function call
    """
    if function_name not in FUNCTION_REGISTRY:
        return {
            "success": False,
            "error": f"Unknown function: {function_name}",
            "available_functions": list(FUNCTION_REGISTRY.keys()),
        }
    
    try:
        return FUNCTION_REGISTRY[function_name](params)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "function": function_name,
        } 