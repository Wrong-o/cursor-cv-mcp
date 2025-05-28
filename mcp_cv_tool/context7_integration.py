"""
Context7-like integration for the screenshot functionality.
This provides a standardized interface similar to Context7 for screenshot capture and analysis.
"""

import json
import os
from typing import Dict, Any, Optional, List, Tuple

from .screenshot import get_screenshot_with_analysis, analyze_image, get_available_monitors
from .automation import mouse_click, type_text, press_key, get_screen_position, highlight_area, find_element

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
    """
    Context7-style function to type text at the current cursor position.
    
    Parameters:
    - text: Text to type (required)
    - interval: Time interval between keystrokes in seconds (default: 0.05)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - text: Text that was typed
      - interval: Interval used
    """
    text = params.get("text")
    
    if text is None:
        return {
            "success": False,
            "error": "Missing required parameter: text is required"
        }
    
    interval = params.get("interval", 0.05)
    
    success = type_text(text=text, interval=interval)
    
    return {
        "success": success,
        "text": text,
        "interval": interval
    }

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
    - screenshot_path: Path to screenshot image (optional, will capture a new one if not provided)
    - element_type: Type of element to find (button, input, checkbox) (optional)
    - text: Text content to match (optional)
    - reference_image: Path to reference image of element (optional)
    - search_area: [x, y, width, height] to limit search area (optional)
    - confidence: Confidence threshold (default: 0.7)
    - debug: Enable debug mode (default: false)
    
    Returns:
    - Dictionary containing:
      - success: Boolean indicating success
      - element: Element information (position, size, confidence)
      - error: Error message if unsuccessful
    """
    # Validate parameters
    screenshot_path = params.get("screenshot_path")
    element_type = params.get("element_type")
    text = params.get("text")
    reference_image = params.get("reference_image")
    search_area = params.get("search_area")
    confidence = params.get("confidence", 0.7)
    debug = params.get("debug", False)
    
    # Check if at least one search method is provided
    if not any([element_type, text, reference_image]):
        return {
            "success": False,
            "error": "At least one of element_type, text, or reference_image must be provided"
        }
    
    # Capture a screenshot if needed
    if not screenshot_path:
        screenshot_result = mcp_screenshot_capture({"debug": debug})
        if not screenshot_result.get("success"):
            return {
                "success": False,
                "error": "Failed to capture screenshot"
            }
        screenshot_path = screenshot_result.get("screenshot_path")
    
    # Call the find_element function
    try:
        element = find_element(
            screenshot_path=screenshot_path,
            element_type=element_type,
            text=text,
            reference_image=reference_image,
            search_area=search_area,
            confidence=confidence,
            debug=debug
        )
        
        if not element:
            return {
                "success": False,
                "error": "Element not found"
            }
        
        # Make sure element is properly serialized
        print(f"Found element details: {element}")
        
        # Create a serializable copy of the element dictionary
        element_data = {
            "x": int(element.get("x", 0)),
            "y": int(element.get("y", 0)),
            "width": int(element.get("width", 0)),
            "height": int(element.get("height", 0)),
            "confidence": float(element.get("confidence", 0.0)),
            "method": str(element.get("method", "unknown")),
            "center_x": int(element.get("center_x", 0)),
            "center_y": int(element.get("center_y", 0))
        }
        
        # Add text if it's available (for OCR results)
        if "text" in element:
            element_data["text"] = str(element.get("text", ""))
        
        return {
            "success": True,
            "element": element_data,
            "message": f"Found element using method '{element.get('method')}' with confidence {element.get('confidence'):.2f}"
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