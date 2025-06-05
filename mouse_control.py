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
        # Get monitor information with detailed debug output
        monitors_info = get_available_monitors()
        
        if not monitors_info or not monitors_info.get("monitors"):
            print("Error: Could not get monitor information")
            return False
        
        # Print debug information about all monitors
        print("\n=== DEBUG: MONITOR INFORMATION ===")
        print(f"Total monitors detected: {len(monitors_info['monitors'])}")
        print(f"Primary monitor index: {monitors_info.get('primary', 'Unknown')}")
        
        for mon in monitors_info["monitors"]:
            print(f"Monitor {mon['id']}: {mon['width']}x{mon['height']} at position ({mon['left']}, {mon['top']})")
        print("=== END MONITOR INFORMATION ===\n")
            
        # Get current position
        current_x, current_y = pyautogui.position()
        print(f"Current mouse position before move: ({current_x}, {current_y})")
        
        # Default to coordinates as provided
        abs_x = x
        abs_y = y
        target_monitor = None
        
        # Determine which monitor to use
        if monitor is not None and monitor > 0:
            # Find the specified monitor in the list
            for mon in monitors_info["monitors"]:
                if mon["id"] == monitor:
                    target_monitor = mon
                    break
            
            if not target_monitor:
                print(f"WARNING: Requested monitor {monitor} not found! Falling back to primary monitor.")
                # Fall back to primary monitor
                monitor = monitors_info.get("primary", 1)
                for mon in monitors_info["monitors"]:
                    if mon["id"] == monitor:
                        target_monitor = mon
                        break
        else:
            # Use primary monitor if none specified
            primary_idx = monitors_info.get("primary", 1)
            for mon in monitors_info["monitors"]:
                if mon["id"] == primary_idx:
                    target_monitor = mon
                    monitor = primary_idx
                    break
        
        # Final safety check - if we still don't have a target monitor, use the first one
        if not target_monitor and monitors_info["monitors"]:
            target_monitor = monitors_info["monitors"][0]
            monitor = target_monitor["id"]
            print(f"WARNING: Falling back to first available monitor: {monitor}")
        
        # Calculate absolute coordinates based on monitor position
        if target_monitor:
            # Check if coordinates are within monitor bounds
            if x < 0 or x >= target_monitor["width"] or y < 0 or y >= target_monitor["height"]:
                print(f"WARNING: Coordinates ({x}, {y}) are outside monitor {monitor} bounds: "
                      f"0-{target_monitor['width']-1} x 0-{target_monitor['height']-1}")
                
                # Clamp coordinates to monitor bounds
                x = max(0, min(x, target_monitor["width"] - 1))
                y = max(0, min(y, target_monitor["height"] - 1))
                print(f"Clamped coordinates to: ({x}, {y})")
            
            # Adjust coordinates based on monitor position
            abs_x = target_monitor["left"] + x
            abs_y = target_monitor["top"] + y
            print(f"Target monitor {monitor}: {target_monitor['width']}x{target_monitor['height']} at ({target_monitor['left']}, {target_monitor['top']})")
            print(f"Converting relative coordinates ({x}, {y}) to absolute: ({abs_x}, {abs_y})")
        else:
            print("ERROR: Could not determine target monitor. Using raw coordinates.")
        
        # Sanity check the final coordinates against all monitor bounds to ensure we're on screen
        is_on_screen = False
        for mon in monitors_info["monitors"]:
            mon_right = mon["left"] + mon["width"]
            mon_bottom = mon["top"] + mon["height"]
            if (mon["left"] <= abs_x < mon_right and 
                mon["top"] <= abs_y < mon_bottom):
                is_on_screen = True
                containing_monitor = mon["id"]
                break
        
        if not is_on_screen:
            print(f"WARNING: Calculated position ({abs_x}, {abs_y}) is not on any screen!")
            return False
        
        if containing_monitor != monitor:
            print(f"ERROR: Calculated position ({abs_x}, {abs_y}) is on monitor {containing_monitor}, not requested monitor {monitor}!")
            return False
        
        # Move to position with a slower duration for better accuracy
        print(f"Moving mouse to absolute position ({abs_x}, {abs_y})...")
        
        # Use a slower movement and a more precise approach by using two movements
        # First move most of the way quickly
        pyautogui.moveTo(abs_x, abs_y, duration=0.3)
        
        # Get position after initial move
        interim_x, interim_y = pyautogui.position()
        
        # If not exactly on target, make a second slower, more precise movement
        if interim_x != abs_x or interim_y != abs_y:
            # Calculate the difference
            diff_x = abs_x - interim_x
            diff_y = abs_y - interim_y
            
            # Make the final adjustment with more precision
            pyautogui.moveRel(diff_x, diff_y, duration=0.2)
        
        # Add a small pause to let the OS and hardware catch up
        time.sleep(0.2)
        
        # Verify position after move
        after_x, after_y = pyautogui.position()
        print(f"Mouse position after move: ({after_x}, {after_y})")
        
        # Return whether we got to the exact position
        position_ok = abs(after_x - abs_x) <= 1 and abs(after_y - abs_y) <= 1
        
        if not position_ok:
            print(f"WARNING: Final position ({after_x}, {after_y}) differs from target ({abs_x}, {abs_y})")
        
        return position_ok
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
        
        # Try up to 3 times to ensure accurate positioning
        max_attempts = 3
        for attempt in range(max_attempts):
            # First move the mouse to the position on the specified monitor
            if mouse_move(x, y, monitor):
                # Get current position (after move)
                current_x, current_y = pyautogui.position()
                
                # Add a more substantial delay before clicking to ensure the mouse has settled
                time.sleep(0.3)
                
                # Click at the current position
                print(f"Clicking at current position ({current_x}, {current_y}) with button={button}, clicks={clicks}")
                
                # Use separate single clicks for more reliability when multiple clicks needed
                for _ in range(clicks):
                    pyautogui.click(button=button)
                    time.sleep(0.1)  # Small delay between clicks
                
                # Verify the click position
                after_x, after_y = pyautogui.position()
                print(f"Position after click: ({after_x}, {after_y})")
                
                # Check if we stayed at the desired position
                if abs(after_x - current_x) <= 1 and abs(after_y - current_y) <= 1:
                    return True
                else:
                    print(f"Warning: Mouse position changed during click attempt {attempt+1}")
            
            if attempt < max_attempts - 1:
                print(f"Retrying mouse click, attempt {attempt+2}/{max_attempts}")
                time.sleep(0.5)  # Pause before retry
        
        print("Failed to accurately click after multiple attempts")
        return False
    except Exception as e:
        print(f"Error clicking at position ({x}, {y}) on monitor {monitor}: {e}")
        import traceback
        traceback.print_exc()
        return False