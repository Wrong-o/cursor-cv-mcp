#!/usr/bin/env python3
"""
Test script to verify PyAutoGUI functionality.
"""

import pyautogui
import time

# Print current screen size
print(f"Screen size: {pyautogui.size()}")

# Print current mouse position
current_x, current_y = pyautogui.position()
print(f"Current mouse position: ({current_x}, {current_y})")

# Move mouse to position (500, 500)
print("Moving mouse to (500, 500)...")
pyautogui.moveTo(500, 500, duration=1)
time.sleep(0.5)

# Get new position
new_x, new_y = pyautogui.position()
print(f"New mouse position: ({new_x}, {new_y})")

# Move back to original position
print(f"Moving back to original position ({current_x}, {current_y})...")
pyautogui.moveTo(current_x, current_y, duration=1)

print("Test completed.") 