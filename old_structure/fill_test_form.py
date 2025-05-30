import time
import pyautogui

def fill_form(monitor_id=2):
    """Fill out the test form with accurate mouse movements."""
    print(f"Filling out form on monitor {monitor_id}...")
    
    # Wait to ensure the page is loaded
    time.sleep(2)
    
    # Step 1: Click on the name field (coordinate values are relative to the target monitor)
    # Adjusted Y coordinates to click lower on each element
    print("\nStep 1: Fill in the name field")
    improved_mouse_click(600, 310, monitor=monitor_id)  # Y adjusted from 280 to 310
    time.sleep(0.5)
    
    # Type the name using pyautogui directly
    pyautogui.typewrite("John Doe", interval=0.1)
    time.sleep(0.5)
    
    # Step 2: Click on the email field
    print("\nStep 2: Fill in the email field")
    improved_mouse_click(600, 380, monitor=monitor_id)  # Y adjusted from 350 to 380
    time.sleep(0.5)
    pyautogui.typewrite("john.doe@example.com", interval=0.1)
    time.sleep(0.5)
    
    # Step 3: Click on the message textarea
    print("\nStep 3: Fill in the message field")
    improved_mouse_click(600, 480, monitor=monitor_id)  # Y adjusted from 450 to 480
    time.sleep(0.5)
    pyautogui.typewrite("This is a test message.", interval=0.1)
    time.sleep(0.5)
    
    # Step 4: Click on the first checkbox
    print("\nStep 4: Check the first checkbox")
    improved_mouse_click(460, 570, monitor=monitor_id)  # Y adjusted from 540 to 570
    time.sleep(0.5)
    
    # Step 5: Click on the second checkbox
    print("\nStep 5: Check the second checkbox")
    improved_mouse_click(460, 600, monitor=monitor_id)  # Y adjusted from 570 to 600
    time.sleep(0.5)
    
    # Step 6: Click the submit button
    print("\nStep 6: Click the submit button")
    improved_mouse_click(550, 650, monitor=monitor_id)  # Y adjusted from 620 to 650
    time.sleep(0.5)
    
    print("\nForm submission complete!")
    return True 