#!/usr/bin/env python
"""
Script to fill out the test HTML form using improved mouse control.
This demonstrates accurate mouse movements and clicks in a multi-monitor setup.
"""

import os
import sys
import time
import webbrowser
from improved_mouse_control import improved_mouse_click, improved_mouse_move, translate_coordinates

def open_test_page():
    """Open the test HTML page in the default browser."""
    # Get the absolute path of the test page
    test_page_path = os.path.abspath("examples/test_page.html")
    
    # Create the page if it doesn't exist
    if not os.path.exists(test_page_path):
        print(f"Error: Test page not found at {test_page_path}")
        return False
    
    test_page_url = f"file://{test_page_path}"
    
    print(f"Opening test page: {test_page_url}")
    
    # Open the test page in the default browser
    webbrowser.open(test_page_url)
    
    # Give the browser some time to open
    print("Waiting for the browser to open...")
    time.sleep(2)
    return True

def fill_form(monitor_id=2):
    """Fill out the test form with accurate mouse movements."""
    print(f"Filling out form on monitor {monitor_id}...")
    
    # Wait to ensure the page is loaded
    time.sleep(2)
    
    # Step 1: Click on the name field (coordinate values are relative to the target monitor)
    print("\nStep 1: Fill in the name field")
    improved_mouse_click(600, 280, monitor=monitor_id)
    time.sleep(0.5)
    
    # Type the name using pyautogui directly
    import pyautogui
    pyautogui.typewrite("John Doe", interval=0.1)
    time.sleep(0.5)
    
    # Step 2: Click on the email field
    print("\nStep 2: Fill in the email field")
    improved_mouse_click(600, 350, monitor=monitor_id)
    time.sleep(0.5)
    pyautogui.typewrite("john.doe@example.com", interval=0.1)
    time.sleep(0.5)
    
    # Step 3: Click on the message textarea
    print("\nStep 3: Fill in the message field")
    improved_mouse_click(600, 450, monitor=monitor_id)
    time.sleep(0.5)
    pyautogui.typewrite("This is a test message.", interval=0.1)
    time.sleep(0.5)
    
    # Step 4: Click on the first checkbox
    print("\nStep 4: Check the first checkbox")
    improved_mouse_click(460, 540, monitor=monitor_id)
    time.sleep(0.5)
    
    # Step 5: Click on the second checkbox
    print("\nStep 5: Check the second checkbox")
    improved_mouse_click(460, 570, monitor=monitor_id)
    time.sleep(0.5)
    
    # Step 6: Click the submit button
    print("\nStep 6: Click the submit button")
    improved_mouse_click(550, 620, monitor=monitor_id)
    time.sleep(0.5)
    
    print("\nForm submission complete!")
    return True

if __name__ == "__main__":
    # Use monitor ID from command line if provided
    monitor_id = 2  # Default to monitor 2
    if len(sys.argv) > 1:
        monitor_id = int(sys.argv[1])
    
    # Open the test page
    if open_test_page():
        # Fill out the form
        fill_form(monitor_id)
    else:
        print("Failed to open test page.") 