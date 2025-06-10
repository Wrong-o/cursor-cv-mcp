import pyautogui
import pyperclip
import time
import platform

def keyboard_layout_info():
    """
    Get information about the current keyboard layout.
    """
    # PyAutoGUI doesn't have a KEYBOARD_LAYOUT attribute
    # Return a basic keyboard info object with system info instead
    class KeyboardInfo:
        def __init__(self):
            self.system = platform.system()
            self.name = f"Default {self.system} keyboard"
    
    return KeyboardInfo()

def keyboard_type_text(text: str) -> bool:
    pyperclip.copy(text)
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'v')
    return True
def keyboard_press_keys(keys: list[str]) -> bool:
    """
    Press a combination of keys, adapting to the current keyboard layout.
    
    Args:
        keys: List of key names to press simultaneously.
              Use "win" for the Windows/Super key.
        
    Returns:
        Success status
    """
    try:
        # Handle special keys that might be layout-dependent
        layout_adapted_keys = []
        for key in keys:
            # You could add more layout-specific adaptations here
            # For now, just use the key as is
            layout_adapted_keys.append(key)
        
        # Press all keys in sequence
        for key in layout_adapted_keys:
            pyautogui.keyDown(key)
            
        # Release in reverse order to ensure proper key combination behavior
        for key in reversed(layout_adapted_keys):
            pyautogui.keyUp(key)
            
        return True
    except Exception as e:
        print(f"Error pressing keys {keys}: {str(e)}")
        # Try to release any keys that might be stuck
        for key in keys:
            try:
                pyautogui.keyUp(key)
            except:
                pass
        return False

if __name__ == "__main__":
    keyboard_type_text("Hello, world!")