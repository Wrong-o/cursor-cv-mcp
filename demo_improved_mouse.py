#!/usr/bin/env python
"""
Demonstration of improved mouse control without relying on screenshot functionality.
This script moves the mouse to multiple points on different monitors with high accuracy.
"""

import os
import sys
import time
from improved_mouse_control import improved_mouse_click, improved_mouse_move, get_monitor_info

def demonstrate_mouse_control():
    """Demonstrate accurate mouse control across multiple monitors."""
    # Get monitor information
    monitors = get_monitor_info()
    if not monitors:
        print("Failed to get monitor information")
        return False
    
    print(f"Detected {len(monitors['monitors'])} monitors:")
    for monitor in monitors['monitors']:
        print(f"  Monitor {monitor['id']}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
    print(f"Primary monitor: {monitors['primary']}")
    
    # Sequence of points to click on monitor 2
    points_monitor_2 = [
        (100, 100), (300, 100), (500, 100), (700, 100), (900, 100),
        (100, 300), (300, 300), (500, 300), (700, 300), (900, 300),
        (100, 500), (300, 500), (500, 500), (700, 500), (900, 500),
    ]
    
    # Sequence of points to click on monitor 1
    points_monitor_1 = [
        (100, 100), (300, 100), (500, 100), (700, 100), (900, 100),
        (100, 300), (300, 300), (500, 300), (700, 300), (900, 300),
        (100, 500), (300, 500), (500, 500), (700, 500), (900, 500),
    ]
    
    print("\nStarting mouse movement demonstration...")
    print("\nMoving to points on Monitor 2:")
    for i, (x, y) in enumerate(points_monitor_2):
        print(f"\nPoint {i+1}: Moving to ({x}, {y}) on Monitor 2")
        improved_mouse_move(x, y, monitor=2)
        time.sleep(0.5)  # Pause to see the movement
    
    print("\nMoving to points on Monitor 1:")
    for i, (x, y) in enumerate(points_monitor_1):
        print(f"\nPoint {i+1}: Moving to ({x}, {y}) on Monitor 1")
        improved_mouse_move(x, y, monitor=1)
        time.sleep(0.5)  # Pause to see the movement
    
    # Return to center of primary monitor
    primary_monitor = monitors["primary"]
    primary_data = monitors["monitors"][primary_monitor - 1]
    center_x = primary_data["width"] // 2
    center_y = primary_data["height"] // 2
    
    print(f"\nReturning to center of primary monitor ({center_x}, {center_y})")
    improved_mouse_move(center_x, center_y, monitor=primary_monitor)
    
    print("\nDemonstration complete!")
    return True

def main():
    """Main function."""
    print("==== Improved Mouse Control Demonstration ====")
    
    # If an argument is provided, use it as the delay before starting
    delay = 0
    if len(sys.argv) > 1:
        try:
            delay = int(sys.argv[1])
            print(f"Waiting {delay} seconds before starting...")
            time.sleep(delay)
        except ValueError:
            print("Invalid delay argument, starting immediately")
    
    success = demonstrate_mouse_control()
    
    if success:
        print("Demonstration completed successfully!")
        return 0
    else:
        print("Demonstration failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 