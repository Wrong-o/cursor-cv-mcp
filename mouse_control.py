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
        
        # IMPORTANT: Validate explicitly that monitor is valid and exists
        if monitor is None:
            # Use primary monitor if none specified
            monitor = monitors_info.get("primary", 1)
            print(f"No monitor specified, using primary monitor: {monitor}")
        
        # Check if the requested monitor exists
        monitor_exists = False
        available_monitor_ids = []
        for mon in monitors_info["monitors"]:
            available_monitor_ids.append(mon["id"])
            if mon["id"] == monitor:
                target_monitor = mon
                monitor_exists = True
                break
        
        if not monitor_exists:
            print(f"ERROR: Requested monitor {monitor} not found! Available monitors: {available_monitor_ids}")
            print(f"Falling back to primary monitor {monitors_info.get('primary', 1)}")
            
            # Fall back to primary monitor
            monitor = monitors_info.get("primary", 1)
            for mon in monitors_info["monitors"]:
                if mon["id"] == monitor:
                    target_monitor = mon
                    break
        
        # Final safety check - if we still don't have a target monitor, use the first one
        if not target_monitor and monitors_info["monitors"]:
            target_monitor = monitors_info["monitors"][0]
            monitor = target_monitor["id"]
            print(f"WARNING: Falling back to first available monitor: {monitor}")
        
        print(f"\n=== USING MONITOR {monitor} ===")
        print(f"Monitor details: {target_monitor}")
        print(f"=== END MONITOR DETAILS ===\n")
        
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
            return False
        
        # Sanity check the final coordinates against all monitor bounds to ensure we're on screen
        is_on_screen = False
        containing_monitor = None
        for mon in monitors_info["monitors"]:
            mon_right = mon["left"] + mon["width"]
            mon_bottom = mon["top"] + mon["height"]
            if (mon["left"] <= abs_x < mon_right and 
                mon["top"] <= abs_y < mon_bottom):
                is_on_screen = True
                containing_monitor = mon["id"]
                break
        
        if not is_on_screen:
            print(f"ERROR: Calculated position ({abs_x}, {abs_y}) is not on any screen!")
            return False
        
        if containing_monitor != monitor:
            print(f"ERROR: Calculated position ({abs_x}, {abs_y}) is on monitor {containing_monitor}, not requested monitor {monitor}!")
            print(f"This indicates a calculation error in coordinate translation. Aborting to prevent clicking on wrong monitor.")
            return False
        
        # Move to position with a slower duration for better accuracy
        print(f"Moving mouse to absolute position ({abs_x}, {abs_y}) on monitor {monitor}...")
        
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
        
        # Check which monitor we ended up on
        final_monitor = None
        for mon in monitors_info["monitors"]:
            mon_right = mon["left"] + mon["width"]
            mon_bottom = mon["top"] + mon["height"]
            if (mon["left"] <= after_x < mon_right and 
                mon["top"] <= after_y < mon_bottom):
                final_monitor = mon["id"]
                break
        
        print(f"Final mouse position is on monitor: {final_monitor}")
        
        # Verify we're on the correct monitor
        if final_monitor != monitor:
            print(f"ERROR: Mouse ended up on monitor {final_monitor}, not the requested monitor {monitor}!")
            return False
        
        # Return whether we got to the exact position
        position_ok = abs(after_x - abs_x) <= 1 and abs(after_y - abs_y) <= 1
        
        if not position_ok:
            print(f"WARNING: Final position ({after_x}, {after_y}) differs from target ({abs_x}, {abs_y})")
        
        return position_ok and final_monitor == monitor
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
                
                # Verify we're on the correct monitor before clicking
                monitors_info = get_available_monitors()
                current_monitor = None
                
                for mon in monitors_info["monitors"]:
                    mon_right = mon["left"] + mon["width"]
                    mon_bottom = mon["top"] + mon["height"]
                    if (mon["left"] <= current_x < mon_right and 
                        mon["top"] <= current_y < mon_bottom):
                        current_monitor = mon["id"]
                        break
                
                if current_monitor != monitor:
                    print(f"ERROR: About to click on monitor {current_monitor}, not requested monitor {monitor}!")
                    print("Aborting click to prevent action on wrong monitor")
                    return False
                
                print(f"Verified mouse is on correct monitor {monitor} before clicking")
                
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

def click_window_position(window_id, rel_x, rel_y, button="left", clicks=1):
    """
    Click at a position relative to a specific window.
    
    Args:
        window_id: ID of the window to click on
        rel_x: X coordinate relative to the window
        rel_y: Y coordinate relative to the window
        button: Mouse button ('left', 'right', 'middle')
        clicks: Number of clicks
    
    Returns:
        Success status
    """
    try:
        # First, we need to get window information
        # This requires a function to get window info by ID
        import subprocess
        import json
        
        # Run a command to get window information (using xdotool or equivalent)
        cmd = ["xdotool", "getwindowgeometry", "--shell", window_id]
        try:
            result = subprocess.check_output(cmd, text=True)
            window_info = {}
            for line in result.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    window_info[key.lower()] = int(value) if value.isdigit() else value
            
            # Calculate absolute position
            abs_x = window_info.get("x", 0) + rel_x
            abs_y = window_info.get("y", 0) + rel_y
            
            # Determine which monitor this window is on
            monitors_info = get_available_monitors()
            window_monitor = None
            
            for mon in monitors_info["monitors"]:
                mon_right = mon["left"] + mon["width"]
                mon_bottom = mon["top"] + mon["height"]
                if (mon["left"] <= window_info.get("x", 0) < mon_right and 
                    mon["top"] <= window_info.get("y", 0) < mon_bottom):
                    window_monitor = mon["id"]
                    break
            
            if window_monitor is None:
                print(f"Could not determine which monitor window {window_id} is on")
                window_monitor = monitors_info.get("primary", 1)
            
            print(f"Window {window_id} is on monitor {window_monitor}")
            print(f"Clicking at window-relative position ({rel_x}, {rel_y})")
            print(f"This translates to absolute position ({abs_x}, {abs_y})")
            
            # First, activate the window to bring it to front
            activate_cmd = ["xdotool", "windowactivate", "--sync", window_id]
            subprocess.run(activate_cmd, check=True)
            time.sleep(0.5)  # Give the window time to activate
            
            # Move to the absolute position
            pyautogui.moveTo(abs_x, abs_y, duration=0.3)
            time.sleep(0.2)
            
            # Click at that position
            for _ in range(clicks):
                pyautogui.click(button=button)
                time.sleep(0.1)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error getting window geometry: {e}")
            return False
    except Exception as e:
        print(f"Error clicking in window {window_id}: {e}")
        import traceback
        traceback.print_exc()
        return False

def click_in_minecraft_launcher(rel_x=None, rel_y=None, button="left", clicks=1):
    """
    Special function to click in the Minecraft launcher.
    This can be used to find and click the Play button.
    
    Args:
        rel_x: X coordinate relative to the launcher window (default: bottom-right where Play button usually is)
        rel_y: Y coordinate relative to the launcher window
        button: Mouse button ('left', 'right', 'middle')
        clicks: Number of clicks
    
    Returns:
        Success status
    """
    try:
        # Find the Minecraft launcher window
        import subprocess
        import json
        
        # Look for windows with "launcher" in their title (case insensitive)
        cmd = ["xdotool", "search", "--name", "launcher"]
        try:
            result = subprocess.check_output(cmd, text=True)
            launcher_windows = result.strip().split("\n")
            
            if not launcher_windows or not launcher_windows[0]:
                print("Could not find Minecraft launcher window")
                return False
            
            launcher_id = launcher_windows[0]  # Use the first matching window
            
            # Get window geometry
            geo_cmd = ["xdotool", "getwindowgeometry", "--shell", launcher_id]
            geo_result = subprocess.check_output(geo_cmd, text=True)
            
            window_info = {}
            for line in geo_result.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    window_info[key.lower()] = int(value) if value.isdigit() else value
            
            # If rel_x or rel_y not provided, use defaults for Play button
            if rel_x is None or rel_y is None:
                width = window_info.get("width", 1000)
                height = window_info.get("height", 600)
                
                # Play button is typically in the bottom right
                if rel_x is None:
                    rel_x = int(width * 0.85)  # 85% from the left
                
                if rel_y is None:
                    rel_y = int(height * 0.85)  # 85% from the top
            
            print(f"Minecraft launcher window found: {launcher_id}")
            print(f"Window size: {window_info.get('width', 'unknown')}x{window_info.get('height', 'unknown')}")
            print(f"Clicking at position ({rel_x}, {rel_y}) relative to window")
            
            # Use the window-specific click function
            return click_window_position(launcher_id, rel_x, rel_y, button, clicks)
            
        except subprocess.CalledProcessError as e:
            print(f"Error finding Minecraft launcher window: {e}")
            return False
    except Exception as e:
        print(f"Error clicking in Minecraft launcher: {e}")
        import traceback
        traceback.print_exc()
        return False