#!/usr/bin/env python
"""
Improved mouse control functions for multi-monitor setups.
This script fixes the coordinate translation issues between monitors.
"""

import os
import sys
import time
import subprocess
import pyautogui

def get_monitor_info():
    """Get information about all connected monitors."""
    try:
        # Use xrandr on Linux to get monitor information
        if sys.platform.startswith('linux'):
            result = subprocess.run(['xrandr', '--current'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error running xrandr: {result.stderr}")
                return None
            
            output = result.stdout
            monitors = []
            current_monitor = {}
            
            for line in output.splitlines():
                if ' connected ' in line:
                    parts = line.split()
                    name = parts[0]
                    
                    # Extract resolution and position
                    geometry = None
                    for part in parts:
                        if '+' in part and 'x' in part and part[0].isdigit():
                            geometry = part
                            break
                    
                    if geometry:
                        # Format: WIDTHxHEIGHT+X+Y
                        dims, pos = geometry.split('+', 1)
                        width, height = map(int, dims.split('x'))
                        x, y = map(int, pos.split('+'))
                        
                        monitors.append({
                            'id': len(monitors) + 1,  # 1-based index
                            'name': name,
                            'width': width,
                            'height': height,
                            'left': x,
                            'top': y,
                            'is_primary': 'primary' in line
                        })
            
            # Identify primary monitor
            primary_index = next((i for i, m in enumerate(monitors) if m.get('is_primary')), 0)
            if monitors:
                monitors[primary_index]['is_primary'] = True
            
            return {
                'monitors': monitors,
                'primary': primary_index + 1  # 1-based index
            }
        else:
            # Fallback to pyautogui's screen size
            width, height = pyautogui.size()
            return {
                'monitors': [{
                    'id': 1,
                    'name': 'default',
                    'width': width,
                    'height': height,
                    'left': 0,
                    'top': 0,
                    'is_primary': True
                }],
                'primary': 1
            }
    except Exception as e:
        print(f"Error getting monitor info: {e}")
        import traceback
        traceback.print_exc()
        return None

def translate_coordinates(x, y, monitor=None):
    """
    Translate coordinates relative to a specific monitor to absolute screen coordinates.
    
    Args:
        x: X coordinate relative to the monitor
        y: Y coordinate relative to the monitor
        monitor: Monitor ID (1-based index) or None for primary
    
    Returns:
        (absolute_x, absolute_y) coordinates
    """
    monitor_info = get_monitor_info()
    if not monitor_info or not monitor_info['monitors']:
        return x, y  # Can't translate, return as-is
    
    # Default to primary monitor if none specified
    if monitor is None:
        monitor = monitor_info['primary']
    
    # Convert to 0-based index
    monitor_index = monitor - 1
    
    # Validate monitor index
    if monitor_index < 0 or monitor_index >= len(monitor_info['monitors']):
        print(f"Warning: Monitor {monitor} not found. Using primary monitor.")
        monitor_index = monitor_info['primary'] - 1
    
    # Get monitor offset
    monitor_data = monitor_info['monitors'][monitor_index]
    offset_x = monitor_data['left']
    offset_y = monitor_data['top']
    
    # Apply offset
    absolute_x = offset_x + x
    absolute_y = offset_y + y
    
    print(f"Translated coordinates: ({x}, {y}) on monitor {monitor} → ({absolute_x}, {absolute_y}) absolute")
    return absolute_x, absolute_y

def improved_mouse_click(x, y, button="left", clicks=1, monitor=None):
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
    # Translate coordinates to absolute screen position
    abs_x, abs_y = translate_coordinates(x, y, monitor)
    
    try:
        # Get screen size for validation
        screen_width, screen_height = pyautogui.size()
        print(f"Screen size: {screen_width}x{screen_height}")
        
        # Get current position
        current_x, current_y = pyautogui.position()
        print(f"Current mouse position before move: ({current_x}, {current_y})")
        
        # Move to position with a slight delay for better accuracy
        print(f"Moving mouse to absolute position ({abs_x}, {abs_y})...")
        pyautogui.moveTo(abs_x, abs_y, duration=0.2)
        
        # Verify position after move
        after_x, after_y = pyautogui.position()
        print(f"Mouse position after move: ({after_x}, {after_y})")
        
        # Click
        print(f"Clicking with {button} button, {clicks} times...")
        pyautogui.click(x=abs_x, y=abs_y, button=button, clicks=clicks)
        
        print(f"Clicked at position ({abs_x}, {abs_y}) with {button} button")
        return True
    except Exception as e:
        print(f"Error clicking at position ({abs_x}, {abs_y}): {e}")
        import traceback
        traceback.print_exc()
        return False

def improved_mouse_move(x, y, monitor=None):
    """
    Move mouse to a specific position on a specific monitor.
    
    Args:
        x: X coordinate relative to the monitor
        y: Y coordinate relative to the monitor
        monitor: Monitor ID (1-based index) or None for primary
    
    Returns:
        Success status
    """
    # Translate coordinates to absolute screen position
    abs_x, abs_y = translate_coordinates(x, y, monitor)
    
    try:
        # Get current position
        current_x, current_y = pyautogui.position()
        print(f"Current mouse position before move: ({current_x}, {current_y})")
        
        # Move to position with a slight delay for better accuracy
        print(f"Moving mouse to absolute position ({abs_x}, {abs_y})...")
        pyautogui.moveTo(abs_x, abs_y, duration=0.2)
        
        # Verify position after move
        after_x, after_y = pyautogui.position()
        print(f"Mouse position after move: ({after_x}, {after_y})")
        
        return True
    except Exception as e:
        print(f"Error moving to position ({abs_x}, {abs_y}): {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the improved mouse control
    print("Testing improved mouse control...")
    
    # Print monitor information
    monitors = get_monitor_info()
    if monitors:
        print(f"Detected {len(monitors['monitors'])} monitors:")
        for monitor in monitors['monitors']:
            print(f"  Monitor {monitor['id']}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
        print(f"Primary monitor: {monitors['primary']}")
    
    # Test with arguments if provided
    if len(sys.argv) >= 4:
        x = int(sys.argv[1])
        y = int(sys.argv[2])
        monitor = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        
        print(f"Testing click at ({x}, {y}) on monitor {monitor}...")
        improved_mouse_click(x, y, monitor=monitor)
    else:
        # Just print monitor information if no coordinates provided
        print("To test, provide coordinates: python improved_mouse_control.py X Y [MONITOR]") 