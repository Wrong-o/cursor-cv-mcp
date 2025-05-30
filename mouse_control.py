import pyautogui
import mss
from cv_and_screenshots import get_available_monitors
import time

def mouse_move(x, y, monitor=None):
    """
    Move mouse to a specific position on a specific monitor.
    
    Args:
        x: X coordinate relative to the monitor
        y: Y coordinate relative to the monitor
        monitor: Monitor ID (1-based index) or None for primary
    
    Returns:
        Success status
    """
    try:
        # Get monitor information
        monitors_info = get_available_monitors()
        
        if not monitors_info or not monitors_info.get("monitors"):
            print("Error: Could not get monitor information")
            return False
            
        # Get current position
        current_x, current_y = pyautogui.position()
        print(f"Current mouse position before move: ({current_x}, {current_y})")
        
        # Calculate absolute coordinates based on monitor
        abs_x = x
        abs_y = y
        
        if monitor is not None and monitor > 0:
            # Find the specified monitor in the list
            target_monitor = None
            for mon in monitors_info["monitors"]:
                if mon["id"] == monitor:
                    target_monitor = mon
                    break
            
            if target_monitor:
                # Adjust coordinates based on monitor position
                abs_x = target_monitor["left"] + x
                abs_y = target_monitor["top"] + y
                print(f"Adjusting coordinates for monitor {monitor}: ({x}, {y}) -> ({abs_x}, {abs_y})")
            else:
                print(f"Warning: Monitor {monitor} not found, using raw coordinates")
        else:
            # Use primary monitor if none specified
            primary_idx = monitors_info.get("primary", 1)
            for mon in monitors_info["monitors"]:
                if mon["id"] == primary_idx:
                    abs_x = mon["left"] + x
                    abs_y = mon["top"] + y
                    print(f"Using primary monitor {primary_idx}: ({x}, {y}) -> ({abs_x}, {abs_y})")
                    break
        
        # Move to position with a slight delay for better accuracy
        print(f"Moving mouse to absolute position ({abs_x}, {abs_y})...")
        pyautogui.moveTo(abs_x, abs_y, duration=0.2)
        
        # Verify position after move
        after_x, after_y = pyautogui.position()
        print(f"Mouse position after move: ({after_x}, {after_y})")
        
        return True
    except Exception as e:
        print(f"Error moving to position ({x}, {y}) on monitor {monitor}: {e}")
        import traceback
        traceback.print_exc()
        return False

def mouse_click(x, y, button="left", clicks=1, monitor=None):
    """
    Click at a specific position on a specific monitor.
    
    Args:
        x: X coordinate relative to the monitor
        y: Y coordinate relative to the monitor
        button: Mouse button ('left', 'right', 'middle')
        clicks: Number of clicks
        monitor: Monitor ID (1-based index) or None for primary
    
    Returns:
        Success status
    """
    try:
        print(f"Attempting to click at ({x}, {y}) on monitor {monitor}")
        
        # First move the mouse to the position on the specified monitor
        if not mouse_move(x, y, monitor):
            print("Failed to move mouse before clicking")
            return False
            
        # Get current position (after move)
        current_x, current_y = pyautogui.position()
        
        # Add a small delay before clicking to ensure the mouse has settled
        time.sleep(0.1)
        
        # Click at the current position
        print(f"Clicking at current position ({current_x}, {current_y}) with button={button}, clicks={clicks}")
        pyautogui.click(button=button, clicks=clicks)
        
        # Verify the click position
        after_x, after_y = pyautogui.position()
        print(f"Position after click: ({after_x}, {after_y})")
        
        return True
    except Exception as e:
        print(f"Error clicking at position ({x}, {y}) on monitor {monitor}: {e}")
        import traceback
        traceback.print_exc()
        return False