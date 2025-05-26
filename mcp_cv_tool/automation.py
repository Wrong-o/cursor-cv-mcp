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