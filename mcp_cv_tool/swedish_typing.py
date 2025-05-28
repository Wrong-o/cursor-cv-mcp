"""
Specialized module for handling Swedish and special character typing.
Prioritizes clipboard paste via xclip for the most reliable character input.
"""

import subprocess
import time
import os
import tempfile
import sys # Added for platform check, though xclip/xdotool are Linux-centric

# --- Configuration for Fallback Methods --- 
# These are used if clipboard paste fails.

# Method 2: X11 Key Names (for xdotool key [name])
# Problematic if system keyboard layout doesn't match Swedish when typing.
X11_KEYS = {
    'ä': 'apostrophe',   # Physical key for 'ä' on Swedish layout
    'ö': 'semicolon',    # Physical key for 'ö' on Swedish layout
    'å': 'bracketleft',  # Physical key for 'å' on Swedish layout
    'Ä': 'apostrophe',   # Shift + apostrophe
    'Ö': 'semicolon',    # Shift + semicolon
    'Å': 'bracketleft',  # Shift + bracketleft
    '_': 'minus',        # Shift + minus
    ':': 'period',       # Shift + period
    '!': '1',            # Shift + 1
    '"': '2',           # Shift + 2 (often needs escape or specific handling)
    '#': '3',            # Shift + 3
    '(': '8',            # Shift + 8
    ')': '9',            # Shift + 9
    '/': '7',            # Shift + 7
    '=': '0',            # Shift + 0
    '+': 'plus',         # plus key (usually without shift)
    '?': 'slash',        # Shift + slash on US, but / is shift+7 on SE. Use 'slash' for xdotool
                         # For '?' on Swedish layout it's Shift + plus. This mapping might be tricky.
    '\\': 'backslash',    # backslash key
    '|': 'bar',          # bar often Shift + backslash or specific key
    # AltGr characters are more complex and xdotool key might not directly support them easily.
    # Clipboard or Unicode is better for these.
    '@': 'at',           # xdotool often has 'at' for @
    # '€': 'EuroSign',   # xdotool might have EuroSign
}

# Method 3: Unicode Input (for xdotool key UXXXX)
# Relies on application support for Unicode input via xdotool.
UNICODE_CHAR_MAP = {
    'å': 'U00e5', 'ä': 'U00e4', 'ö': 'U00f6',
    'Å': 'U00c5', 'Ä': 'U00c4', 'Ö': 'U00d6',
    '€': 'U20ac',
    # Add other common special characters if needed
}

# Method 4: Old SWEDISH_KEY_MAP (Physical key simulation with modifiers)
# This is similar to X11_KEYS but explicitly defines shift/altgr.
# Kept as a last resort if other xdotool methods fail.
SWEDISH_KEY_MAP = {
    'å': 'bracketleft', 'ä': 'apostrophe', 'ö': 'semicolon',
    'Å': 'shift+bracketleft', 'Ä': 'shift+apostrophe', 'Ö': 'shift+semicolon',
    '_': 'shift+minus', ':': 'shift+period', '!': 'shift+1', '"': 'shift+2',
    '#': 'shift+3', '%': 'shift+5', '&': 'shift+6', '/': 'shift+7',
    '(': 'shift+8', ')': 'shift+9', '=': 'shift+0',
    '+': 'plus', # Note: 'plus' key is usually unshifted. Shift+plus for '?' on SE.
    '?': 'shift+plus', # Correct for SE layout '?'
    '\\': 'backslash', # Assuming a dedicated backslash key
    '|': 'shift+less', # Often shift + key left of '1' or specific key for '|' on SE.
    '@': 'Alt_R+2',      # AltGr+2 on SE
    '£': 'Alt_R+3',      # AltGr+3 on SE
    '$': 'Alt_R+4',      # AltGr+4 on SE
    '€': 'Alt_R+e',      # AltGr+e on SE
    '{': 'Alt_R+7',      # AltGr+7 on SE
    '[': 'Alt_R+8',      # AltGr+8 on SE
    ']': 'Alt_R+9',      # AltGr+9 on SE
    '}': 'Alt_R+0',      # AltGr+0 on SE
    '~': 'Alt_R+asciitilde', # AltGr + key for ¨ / ^, then space
}

def has_special_chars(text):
    """Check if text contains characters that might need special handling."""
    # Broaden the definition of special characters to include anything not plain ASCII
    # or anything explicitly mapped for xdotool key names.
    if any(ord(c) > 127 for c in text):
        return True
    if any(c in X11_KEYS for c in text):
        return True # Characters we have specific xdotool 'key' fallbacks for
    # Add other characters that might be problematic for direct pyautogui.typewrite() or xdotool type
    if any(c in '"\'`$()[]{}|\\' for c in text): # Corrected line with escaped backslash
        return True
    return False


def _run_subprocess(command_args, description):
    """Helper to run subprocess and catch/print errors."""
    try:
        # Using text=True for Python 3.7+
        process = subprocess.run(command_args, check=True, capture_output=True, text=True, encoding='utf-8', timeout=5)
        # print(f"Subprocess '{description}' successful. STDOUT: {process.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Subprocess '{description}' FAILED. Return code: {e.returncode}")
        print(f"STDERR: {e.stderr.strip()}")
        print(f"STDOUT: {e.stdout.strip()}")
        return False
    except subprocess.TimeoutExpired as e:
        print(f"Subprocess '{description}' TIMED OUT after 5 seconds.")
        if e.stderr:
            print(f"STDERR (on timeout): {e.stderr.strip()}")
        if e.stdout:
            print(f"STDOUT (on timeout): {e.stdout.strip()}")
        return False
    except FileNotFoundError:
        print(f"Subprocess '{description}' FAILED. Command not found (e.g., xclip or xdotool missing?).")
        return False
    except Exception as e:
        print(f"Subprocess '{description}' FAILED with unexpected error: {e}")
        return False


# --- Typing Methods (Order of Preference) ---

def type_with_clipboard(text, interval=0.05): # interval often not strictly needed here
    """Primary Method: Type text using clipboard (xclip + xdotool paste)."""
    print(f"Attempting Method 1: Clipboard Paste for: '{text}'")
    temp_path = ""
    try:
        # Create a temporary file with the text, ensuring UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            f.write(text)
            temp_path = f.name
        print(f"Text written to temporary file: {temp_path}")

        print(f"DEBUG: swedish_typing.py: About to call _run_subprocess for xclip with temp_file: {temp_path}")

        # Use xclip to copy text from file to clipboard
        if not _run_subprocess(['xclip', '-selection', 'clipboard', '-i', temp_path], "xclip copy"):
            print("xclip copy failed. Is xclip installed and in PATH?")
            return False
        print("xclip copy successful.")

        # Wait a moment for clipboard to update
        time.sleep(0.2) 

        # Paste using xdotool keyboard shortcut (Ctrl+V)
        if not _run_subprocess(['xdotool', 'key', '--clearmodifiers', 'ctrl+v'], "xdotool paste"):
            print("xdotool paste (ctrl+v) failed.")
            return False
        print("xdotool paste successful.")
        
        print(f"Clipboard method appears successful for '{text}'.")
        return True

    except Exception as e:
        print(f"Error during clipboard method: {e}")
        return False
    finally:
        # Remove the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Temporary file {temp_path} removed.")
            except OSError as e:
                print(f"Error removing temporary file {temp_path}: {e}")


def type_with_unicode_input(text, interval=0.05):
    """Fallback Method 2: Type text using xdotool's Unicode input (UXXXX)."""
    print(f"Attempting Method 2: Unicode Input for: '{text}'")
    try:
        for char_index, char in enumerate(text):
            if char in UNICODE_CHAR_MAP:
                key_code = UNICODE_CHAR_MAP[char]
                print(f"Typing char '{char}' using mapped Unicode: {key_code}")
            elif ord(char) > 127: # Any non-ASCII character
                key_code = f"U{ord(char):04X}" # Format as UXXXX (hex)
                print(f"Typing char '{char}' using direct Unicode: {key_code}")
            else: # Plain ASCII
                print(f"Typing char '{char}' using xdotool type (ASCII)")
                if not _run_subprocess(['xdotool', 'type', '--delay', str(int(interval*1000)), char], f"xdotool type '{char}'"):
                    # If typing even plain ASCII fails, something is very wrong with xdotool
                    return False
                if interval > 0 and char_index < len(text) -1: time.sleep(interval)
                continue # Skip common key press for ASCII typed this way

            # For Unicode mapped or generated characters
            if not _run_subprocess(['xdotool', 'key', '--clearmodifiers', key_code], f"xdotool key {key_code}"):
                return False # Abort if any char fails
            
            if interval > 0 and char_index < len(text) - 1: # Don't sleep after the last character
                time.sleep(interval)
        print(f"Unicode input method successful for '{text}'.")
        return True
    except Exception as e:
        print(f"Error during Unicode input method: {e}")
        return False


def type_with_x11_key_names(text, interval=0.05):
    """Fallback Method 3: Type text using xdotool key names (e.g., 'apostrophe')."""
    print(f"Attempting Method 3: X11 Key Names for: '{text}'")
    try:
        for char_index, char in enumerate(text):
            key_sequence = None
            is_shifted = False

            if char.isupper() and char.lower() in X11_KEYS:
                key_sequence = X11_KEYS[char.lower()]
                is_shifted = True
                print(f"Typing '{char}' (shifted '{char.lower()}') using X11 key name: {key_sequence}")
            elif char in X11_KEYS:
                key_sequence = X11_KEYS[char]
                print(f"Typing '{char}' using X11 key name: {key_sequence}")
                # Check if this char inherently requires shift from its mapping in X11_KEYS (e.g. '!')
                # This is a simplification; a better map would distinguish base keys from shifted keys.
                if char in "_:!\"#()=?" : # Add other chars that are shifted on standard US for these key names
                    is_shifted = True 
            
            if key_sequence:
                if is_shifted:
                    if not _run_subprocess(['xdotool', 'keydown', 'shift'], "xdotool keydown shift"):
                         return False
                
                if not _run_subprocess(['xdotool', 'key', '--clearmodifiers', key_sequence], f"xdotool key {key_sequence}"):
                    if is_shifted: _run_subprocess(['xdotool', 'keyup', 'shift'], "xdotool keyup shift") # Attempt cleanup
                    return False

                if is_shifted:
                    if not _run_subprocess(['xdotool', 'keyup', 'shift'], "xdotool keyup shift"):
                        return False # Or at least log it
            else:
                # Fallback to simple type for characters not in X11_KEYS (could be plain ASCII)
                print(f"Typing char '{char}' using xdotool type (not in X11_KEYS map)")
                if not _run_subprocess(['xdotool', 'type', char], f"xdotool type '{char}'"):
                    return False

            if interval > 0 and char_index < len(text) - 1:
                time.sleep(interval)
        print(f"X11 key names method successful for '{text}'.")
        return True
    except Exception as e:
        print(f"Error during X11 key names method: {e}")
        return False


def type_with_physical_key_simulation(text, interval=0.05):
    """Fallback Method 4: Simulate physical keystrokes using SWEDISH_KEY_MAP (explicit modifiers)."""
    print(f"Attempting Method 4: Physical Key Simulation (SWEDISH_KEY_MAP) for: '{text}'")
    try:
        # Optional: Check current keyboard layout (for debugging only, doesn't change behavior)
        # try:
        #     layout_info = subprocess.run(['setxkbmap', '-query'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #     print(f"DEBUG: Current keyboard layout (setxkbmap -query):\n{layout_info.stdout}")
        # except Exception as e:
        #     print(f"DEBUG: Could not get keyboard layout info: {e}")
            
        for char_index, char in enumerate(text):
            if char in SWEDISH_KEY_MAP:
                key_sequence = SWEDISH_KEY_MAP[char]
                print(f"Typing '{char}' using defined SWEDISH_KEY_MAP sequence: {key_sequence}")
                
                parts = key_sequence.split('+')
                modifiers_down = []

                for part in parts[:-1]: # Handle all modifier keys
                    modifier_key_name = part.lower() # e.g. 'shift', 'alt_r'
                    if not _run_subprocess(['xdotool', 'keydown', modifier_key_name], f"xdotool keydown {modifier_key_name}"):
                        # Attempt to release any pressed modifiers before returning
                        for mod_up in reversed(modifiers_down):
                            _run_subprocess(['xdotool', 'keyup', mod_up], f"xdotool keyup {mod_up} (cleanup)")
                        return False
                    modifiers_down.append(modifier_key_name)
                    time.sleep(0.03) # Small delay after modifier down

                # Press the main key
                main_key = parts[-1]
                if not _run_subprocess(['xdotool', 'key', '--clearmodifiers', main_key], f"xdotool key {main_key}"):
                    for mod_up in reversed(modifiers_down):
                        _run_subprocess(['xdotool', 'keyup', mod_up], f"xdotool keyup {mod_up} (cleanup)")
                    return False
                time.sleep(0.03) # Small delay after main key press

                # Release modifiers in reverse order
                for modifier_key_name in reversed(modifiers_down):
                    if not _run_subprocess(['xdotool', 'keyup', modifier_key_name], f"xdotool keyup {modifier_key_name}"):
                        # Log this but don't necessarily fail the whole sequence
                        print(f"Warning: Failed to keyup modifier {modifier_key_name}")
                    time.sleep(0.02)
            
            elif ord(char) > 127: # Non-ASCII, not in SWEDISH_KEY_MAP
                # Fallback to Unicode input for characters not in SWEDISH_KEY_MAP
                print(f"Char '{char}' not in SWEDISH_KEY_MAP, trying direct Unicode for it.")
                key_code = f"U{ord(char):04X}"
                if not _run_subprocess(['xdotool', 'key', '--clearmodifiers', key_code], f"xdotool key {key_code} (fallback within physical sim)"):
                    return False # Abort if this sub-part fails
            else:
                # Plain ASCII, not in SWEDISH_KEY_MAP
                print(f"Typing char '{char}' using xdotool type (ASCII, not in SWEDISH_KEY_MAP)")
                if not _run_subprocess(['xdotool', 'type', char], f"xdotool type '{char}'"):
                    return False
            
            if interval > 0 and char_index < len(text) - 1:
                time.sleep(interval)
                
        print(f"Physical key simulation method successful for '{text}'.")
        return True
    except Exception as e:
        print(f"Error during physical key simulation method: {e}")
        return False


# --- Main Entry Point --- 

def type_special_text(text, interval=0.05):
    """
    Main function to type text, especially with special/Swedish characters.
    Prioritizes Unicode input via xdotool.
    """
    print(f"--- Typing special text: '{text}' (Interval: {interval*1000}ms) ---")
    if not sys.platform.startswith('linux'):
        print("Warning: This script is optimized for Linux with xdotool. Behavior on other OS may vary.")
        # For non-Linux, you might want a very basic pyautogui.typewrite or similar if pyautogui is a dependency.
        # For now, it will just fail for non-Linux if Unicode input isn't applicable.

    # Directly use Unicode Input method
    if type_with_unicode_input(text, interval):
        print("--- Unicode Input SUCCEEDED. ---")
        return True
    
    print(f"--- Unicode Input FAILED for '{text}'. Consider checking xdotool installation and target application compatibility. ---")
    return False


# Example usage for direct testing of this script:
if __name__ == '__main__':
    print("Running swedish_typing.py directly for testing.")
    test_string_problematic = "_:2183u70äöööööööööööööööööööööööööööööööööööäp"
    test_string_simple_swedish = "åäöÅÄÖ"
    test_string_mixed = "Hello & Välkommen! What is your name? Åke's price is €25."
    test_string_symbols = "!\"#€%&/()=?+-_.:,;<>|@[]{}~`*^'\""

    print("\n--- Test 1: Problematic String --- ")
    type_special_text(test_string_problematic, interval=0.05)

    print("\n--- Test 2: Simple Swedish --- ")
    type_special_text(test_string_simple_swedish, interval=0.05)

    print("\n--- Test 3: Mixed Content --- ")
    type_special_text(test_string_mixed, interval=0.05)
    
    print("\n--- Test 4: Symbols String --- ")
    type_special_text(test_string_symbols, interval=0.05)

    print("\n--- Test 5: ASCII only (should still work) --- ")
    type_special_text("Plain ASCII text 123.", interval=0.05)

    print("\nTesting complete. Check terminal output above and where text was typed.") 