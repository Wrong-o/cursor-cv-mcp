#!/usr/bin/env python
"""
Script to interact with chat interfaces like LM Studio using improved mouse control.
"""

import os
import sys
import time
import requests
import json
from improved_mouse_control import improved_mouse_click, improved_mouse_move, get_monitor_info

# Base URL for the screenshot server
SCREENSHOT_SERVER_URL = "http://localhost:8001"

def take_screenshot(monitor_id=1):
    """Take a screenshot of a specific monitor."""
    try:
        # Call the MCP function endpoint to capture screenshot
        response = requests.post(
            f"{SCREENSHOT_SERVER_URL}/mcp/call_function",
            json={
                "function_name": "mcp_screenshot_capture",
                "params": {"monitor": monitor_id}
            }
        )
        
        if response.status_code != 200:
            print(f"Error taking screenshot: {response.status_code}")
            return None
        
        data = response.json()
        if not data.get("success"):
            print(f"Screenshot capture failed: {data.get('error')}")
            return None
        
        # Return the path to the screenshot
        return data.get("screenshot_path")
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return None

def find_chat_input(screenshot_path, monitor_id=1):
    """Find the chat input area in the screenshot."""
    try:
        # Call the MCP function endpoint to analyze the image
        with open(screenshot_path, 'rb') as img_file:
            # First save the image file path
            files = {'image': img_file}
            response = requests.post(
                f"{SCREENSHOT_SERVER_URL}/mcp/call_function",
                json={
                    "function_name": "mcp_screenshot_analyze_image",
                    "params": {"image_path": screenshot_path, "detect_inputs": True, "confidence": 0.4}
                }
            )
        
        if response.status_code != 200:
            print(f"Error analyzing screenshot: {response.status_code}")
            return None
        
        data = response.json()
        if not data.get("success"):
            print(f"Image analysis failed: {data.get('error')}")
            return None
        
        # Get the analysis results
        analysis = data.get("analysis", {})
        inputs = analysis.get("elements", {}).get("inputs", [])
        
        if not inputs:
            print("No input elements found")
            return None
        
        # Sort inputs by y-coordinate (bottom to top)
        inputs.sort(key=lambda x: -x.get("center_y", 0))
        
        # Get screen dimensions
        monitor_info = get_monitor_info()
        if not monitor_info:
            print("Could not get monitor info")
            return None
        
        monitor_index = monitor_id - 1
        if monitor_index < 0 or monitor_index >= len(monitor_info["monitors"]):
            print(f"Monitor {monitor_id} not found")
            return None
            
        monitor_height = monitor_info["monitors"][monitor_index]["height"]
        
        # Find inputs in the bottom third of the screen
        bottom_third = monitor_height * 2 / 3
        bottom_inputs = [i for i in inputs if i.get("center_y", 0) > bottom_third]
        
        if bottom_inputs:
            # Use the bottommost input
            chat_input = bottom_inputs[0]
            print(f"Found potential chat input: {chat_input}")
            return chat_input
        else:
            # If no inputs in bottom third, use the bottommost input
            print(f"No inputs found in bottom third, using bottommost input: {inputs[0]}")
            return inputs[0]
    except Exception as e:
        print(f"Error finding chat input: {e}")
        import traceback
        traceback.print_exc()
        return None

def send_message(message, monitor_id=1):
    """Send a message to the chat interface."""
    print(f"Attempting to send message: '{message}'")
    
    # Take a screenshot
    screenshot_path = take_screenshot(monitor_id)
    if not screenshot_path:
        print("Failed to take screenshot")
        return False
    
    print(f"Screenshot taken: {screenshot_path}")
    
    # Find the chat input
    chat_input = find_chat_input(screenshot_path, monitor_id)
    if not chat_input:
        print("Failed to find chat input")
        return False
    
    # Click on the chat input
    x = chat_input.get("center_x")
    y = chat_input.get("center_y")
    
    print(f"Clicking on chat input at ({x}, {y})")
    success = improved_mouse_click(x, y, monitor=monitor_id)
    if not success:
        print("Failed to click on chat input")
        return False
    
    # Wait a moment for the click to register
    time.sleep(0.5)
    
    # Type the message
    import pyautogui
    pyautogui.typewrite(message, interval=0.05)
    print(f"Typed message: '{message}'")
    
    # Press Enter to send
    time.sleep(0.5)
    pyautogui.press('enter')
    print("Pressed Enter to send message")
    
    return True

def main():
    """Main function."""
    # Print monitor information
    monitors = get_monitor_info()
    if monitors:
        print(f"Detected {len(monitors['monitors'])} monitors:")
        for monitor in monitors['monitors']:
            print(f"  Monitor {monitor['id']}: {monitor['width']}x{monitor['height']} at ({monitor['left']}, {monitor['top']})")
        print(f"Primary monitor: {monitors['primary']}")
    
    # Default message
    message = "Hello from Claude! Using improved mouse control for accurate positioning."
    
    # Use monitor ID and message from command line if provided
    monitor_id = 1  # Default to monitor 1
    if len(sys.argv) > 1:
        monitor_id = int(sys.argv[1])
    if len(sys.argv) > 2:
        message = sys.argv[2]
    
    print(f"Using monitor {monitor_id}")
    success = send_message(message, monitor_id)
    
    if success:
        print("Message sent successfully!")
        return 0
    else:
        print("Failed to send message")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 