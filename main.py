import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi_mcp import FastApiMCP
from cv_and_screenshots import get_available_monitors, get_screenshot, caption_window, get_screenshot_with_analysis, find_text_in_image as cv_find_text_in_image, analyze_window, get_screenshot_at_mouse, extract_dropdown_options
from mouse_control import mouse_move as move_mouse_function, mouse_click as click_mouse_function, click_window_element
from keyboard_control import keyboard_type_text, keyboard_press_keys, keyboard_layout_info
from window_control import get_open_windows, activate_window, launch_application
from file_operations import open_downloads_folder, get_folder_contents
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from microphone import listen_to_microphone
import os
from fastapi.responses import FileResponse
from tts_service import tts_service
from os_info import get_os_info
import platform
import subprocess
import time
import pyautogui

class PressKeysInput(BaseModel):
    """Input for the press_keys endpoint"""
    keys: list[str]

class TypeTextRequest(BaseModel):
    text: str

class FindTextInImageInput(BaseModel):
    """Input for the find_text_in_image endpoint"""
    image: bytes
    target: str

app = FastAPI(title="pinoMCP")

@app.get("/hello", operation_id="say_hello")
async def hello():
    """A simple greeting endpoint"""
    return {"message": "Hello World"}

@app.get("/system_info", operation_id="system_info")
async def system_info():
    """MUST BE CALLED BEFORE ANY OTHER ENDPOINT.Get information about the system, including monitors, operating system, and keyboard layout"""
    monitors = get_available_monitors()
    return {"monitors": monitors, "os": get_os_info(), "keyboard_layout": keyboard_layout_info(), "keyboard_layout_name": keyboard_layout_info().name}

@app.get("/screenshot_with_analysis", operation_id="screenshot_with_analysis")
async def screenshot_with_analysis(monitor_id: int, target_string: Optional[str] = None):
    """Screenshots entire monitor.
    This function should not be used if only a certain window is of interest, use analyze_window instead.
    ALWAYS MAKE SURE THAT THE RIGHT WINDOW IS VISIBLE BY USING LIST WINDOWS AND ACTIVATE WINDOW.
    Args:
        monitor_id: ID of the monitor to capture and analyze
        target_string: Optional string to find in the image

    Returns:
        Dictionary containing analysis of the screenshot, buttons, checkboxes, text matches, and UI regions
    """
    try:
        # Print debug info
        print(f"Requested screenshot with monitor_id={monitor_id}, target_string={target_string}")
        
        # Get screenshot with analysis
        screenshot_data, analysis, ui_buttons, ui_checkboxes, ui_regions = get_screenshot_with_analysis(monitor_id)
        if screenshot_data is None:
            return {"error": "Failed to capture screenshot"}

        text_matches = None

        if target_string is not None:
            try:
                text_matches = cv_find_text_in_image(image_data=screenshot_data, target=target_string)
                # Convert to a more user-friendly format
                if text_matches:
                    formatted_matches = []
                    for word, bounds in text_matches:
                        x, y, w, h = bounds
                        formatted_matches.append({
                            "text": word,
                            "position": {
                                "x": x,
                                "y": y,
                                "width": w,
                                "height": h,
                                "center_x": x + w//2,
                                "center_y": y + h//2
                            }
                        })
                    text_matches = formatted_matches
            except Exception as text_error:
                print(f"Error finding text in image: {str(text_error)}")
                import traceback
                traceback.print_exc()
                text_matches = {"error": str(text_error)}

        # Format checkbox results to be more user-friendly
        formatted_checkboxes = []
        if ui_checkboxes:
            # Handle checkboxes which are already in the correct format with 'type' and 'position' keys
            formatted_checkboxes = ui_checkboxes

        # Process UI regions to remove binary data
        processed_regions = []
        if ui_regions:
            for region in ui_regions:
                # Make a copy of the region without the binary image data
                processed_region = region.copy()
                if "image_data" in processed_region:
                    # Remove the binary image data
                    del processed_region["image_data"]
                processed_regions.append(processed_region)
                
        # ADDED: Debug output for buttons
        print(f"BEFORE FILTERING: Found {len(ui_buttons) if ui_buttons else 0} buttons")
        if ui_buttons:
            for i, button in enumerate(ui_buttons):
                print(f"  Button {i+1}: text='{button.get('text', 'No text')}', position={button.get('position', {})}")
        
        # Filter out buttons with text "Button"
        filtered_buttons = []
        if ui_buttons:
            for button in ui_buttons:
                button_text = button.get("text", "")
                if button_text != "Button":
                    filtered_buttons.append(button)
                else:
                    print(f"  FILTERED OUT: Button with generic 'Button' text at position {button.get('position', {})}")
            
            print(f"AFTER FILTERING: Kept {len(filtered_buttons)} buttons, removed {len(ui_buttons) - len(filtered_buttons)}")
        else:
            filtered_buttons = []

        return {
            "analysis": analysis,
            "buttons": filtered_buttons,
            "checkboxes": formatted_checkboxes,
            "text_matches": text_matches,
            "ui_regions": processed_regions
        }
    except Exception as e:
        # Detailed error logging
        import traceback
        print(f"ERROR in screenshot_with_analysis: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "details": traceback.format_exc()}

@app.get("/mouse_move", operation_id="mouse_move")
async def mouse_move(x: int, y: int, monitor: int):
    """Move the mouse to a specific position on a specific monitor
    Double check that we are on the correct monitor before moving the mouse"""
    result = move_mouse_function(x, y, monitor)
    if not result:
        # Get monitor information to provide helpful error response
        monitors_info = get_available_monitors()
        available_monitors = [mon["id"] for mon in monitors_info.get("monitors", [])]
        primary_monitor = monitors_info.get("primary", None)
        
        return {
            "success": False, 
            "error": f"Failed to move mouse to ({x}, {y}) on monitor {monitor}",
            "available_monitors": available_monitors,
            "primary_monitor": primary_monitor,
            "tip": "Make sure to specify a valid monitor ID. Check logs for detailed error information."
        }
    return {"success": result}

@app.get("/mouse_click", operation_id="mouse_click")
async def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1, monitor: int = 0):
    """Click the mouse at a specific position on a specific monitor
    Double check that we are on the correct monitor before moving the mouse"""
    result = click_mouse_function(x, y, button, clicks, monitor)
    if not result:
        # Get monitor information to provide helpful error response
        monitors_info = get_available_monitors()
        available_monitors = [mon["id"] for mon in monitors_info.get("monitors", [])]
        primary_monitor = monitors_info.get("primary", None)
        
        return {
            "success": False, 
            "error": f"Failed to click at ({x}, {y}) on monitor {monitor}",
            "available_monitors": available_monitors,
            "primary_monitor": primary_monitor,
            "tip": "Make sure to specify a valid monitor ID and coordinates within that monitor's bounds."
        }
    return {"success": result}

@app.get("/type_text", operation_id="type_text")
async def type_text(text: str):
    """Type text at a specific position on a specific monitor
    For button combinations, use the press_keys endpoint. This can only produce text, not button combinations."""
    return {"success": keyboard_type_text(text)}

@app.post("/press_keys", operation_id="press_keys")
async def press_keys(data: PressKeysInput):
    """Press keys on the keyboard, keys is a list of strings. All keys are pressed at the same time.
    Example: press_keys["ctrl", "tab"]
    Use "win" for the Windows/Super key."""
    return {"success": keyboard_press_keys(data.keys)}

@app.post("/find_text_in_image", operation_id="find_text_in_image")
async def find_text_in_image(data: FindTextInImageInput):
    """Find text in an image"""
    return {"success": cv_find_text_in_image(data.image, data.target)}

@app.post("/speech_input", operation_id="speech_input")
async def speech_input():
    """Listen to the microphone and convert speech to text."""
    try:
        result = listen_to_microphone()
        return {"success": True, "text": result}
    except Exception as e:
        print(f"Error in speech input: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/list_windows", operation_id="list_windows")
async def list_windows():
    """List all open windows across platforms.
    
    Returns information about all open windows including title, position, and size.
    This function works on Windows, macOS, and Linux.
    """
    try:
        windows = get_open_windows()
        return {"success": True, "windows": windows}
    except Exception as e:
        print(f"Error listing windows: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/activate_window", operation_id="activate_window")
async def api_activate_window(window_id: Optional[str] = None, window_title: Optional[str] = None):
    """Activate (bring to foreground) a specific window.
    
    Args:
        window_id: ID of the window to activate (platform-specific)
        window_title: Title of the window to activate (if window_id not provided)
        
    Returns:
        Success status
    """
    if not window_id and not window_title:
        return {"success": False, "error": "Either window_id or window_title must be provided"}
    
    try:
        result = activate_window(window_id, window_title)
        return {"success": result}
    except Exception as e:
        print(f"Error activating window: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/launch_application", operation_id="launch_application")
async def api_launch_application(app_name: str, app_path: Optional[str] = None, launch_method: str = "GUI"):
    """Launches an application. After running this function, you should use list_windows to check if the application is open.
    The 2 diffrent methods are:
    1. GUI: This is the default method and will use the GUI to launch the application.
    2. Terminal: This will use the terminal to launch the application. This may be needed if GUI method doesn't work properly.
    
    Args:
        app_name: Name of the application to launch
        app_path: Full path to the application executable (optional)
        launch_method: Method to launch the application - "Terminal" (default) or "GUI"
        
    Returns:
        Success status
    """
    try:
        if launch_method.lower() == "gui":
            # GUI method: Press Win key, type app name, press Enter
            keyboard_press_keys(["win"])
            time.sleep(0.5)
            keyboard_type_text(app_name)
            time.sleep(0.5)
            keyboard_press_keys(["enter"])
            return {"success": True}
        else:
            # Terminal method: Use the original launch_application function
            result = launch_application(app_name, app_path)
            return {"success": result}
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/read_with_voice", operation_id="read_with_voice")
async def read_with_voice(text: str, speaker_id: Optional[int] = 0):
    """Generate speech from text and play it
    
    Args:
        text: The text to convert to speech
        speaker_id: Optional speaker ID for voice selection (default: 0, not used with gTTS)
        
    Returns:
        Success status and audio file path
    """
    try:
        # Generate speech
        audio_path = tts_service.generate_speech(text, speaker_id)
        
        if not audio_path:
            raise HTTPException(status_code=400, detail="Failed to generate speech: Empty text")
            
        # Return the audio file for streaming
        return FileResponse(
            path=audio_path,
            media_type="audio/mpeg",
            filename="speech.mp3",
            headers={"Content-Disposition": "inline"}
        )
    except Exception as e:
        import traceback
        print(f"Error in read_with_voice: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

@app.get("/mcp_read_with_voice", operation_id="mcp_read_with_voice")
async def mcp_read_with_voice(text: str, speaker_id: Optional[int] = 0):
    """Generate speech from text and play it (MCP-compatible version)
    
    This endpoint is specifically designed to be compatible with the MCP interface.
    It generates the speech file and plays it directly on the server, then returns
    a JSON response instead of the binary audio data.
    
    Args:
        text: The text to convert to speech
        speaker_id: Optional speaker ID for voice selection (default: 0, not used with gTTS)
        
    Returns:
        JSON with success status and text that was spoken
    """
    try:
        # Generate speech
        audio_path = tts_service.generate_speech(text, speaker_id)
        
        if not audio_path:
            return {"success": False, "error": "Failed to generate speech: Empty text"}
            
        # Play the audio on the server
        try:
            # Try different players in order of preference
            players = [
                ['cvlc', '--play-and-exit', '--no-video'],
                ['vlc', '--play-and-exit', '--no-video'],
                ['mpv', '--no-video'],
                ['mplayer']
            ]
            
            import subprocess
            success = False
            for player_cmd in players:
                try:
                    cmd = player_cmd + [audio_path]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    success = True
                    break
                except FileNotFoundError:
                    continue
            
            if not success:
                print("Warning: No compatible media player found to play the speech")
            
            # Return JSON response (MCP-compatible)
            return {
                "success": True, 
                "text": text,
                "message": "Text was successfully converted to speech and played"
            }
            
        except Exception as play_error:
            print(f"Error playing speech: {str(play_error)}")
            # Even if playing failed, we return partial success
            return {
                "success": False,
                "text": text,
                "error": f"Generated speech but failed to play: {str(play_error)}"
            }
            
    except Exception as e:
        import traceback
        print(f"Error in mcp_read_with_voice: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": f"Failed to generate speech: {str(e)}"}

@app.get("/open_folder_gui", operation_id="open_folder_gui")
async def api_open_folder_gui(folder_path: str):
    """Open the user's Downloads folder based on the operating system.
    
    Works on Windows, macOS, and Linux.
    
    Returns:
        Success status
    """
    try:
        result = open_downloads_folder()
        return {"success": result}
    except Exception as e:
        print(f"Error opening Downloads folder: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/list_folder_contents", operation_id="list_folder_contents")
async def api_list_folder_contents(folder_path: str):
    """Get the contents of a folder
    Use this function to get the contents of a folder instead of screenshotting the folder.
    
    Args:
        folder_path: Path to the folder to get the contents of
        
    Returns:
        List of files and directories in the folder
    """
    try:
        result = get_folder_contents(folder_path)
        return result
    except Exception as e:
        print(f"Error getting folder contents: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/analyze_window", operation_id="analyze_window")
async def api_analyze_window(window_id: str, window_title: Optional[str] = None):
    """Analyze a window and return a list of UI elements
    If you want to caption a window, use the caption_window endpoint.
    
    Detects UI elements like buttons and checkboxes 
    
    Args:
        window_id: ID of the window to analyze
        window_title: Title of the window to analyze (optional, used if window_id not found)
        
    Returns:
        Analysis results UI elements, and window metadata
    """
    try:
        print(f"Analyzing window with ID {window_id} and title {window_title}")
        
        # First ensure that required tools are available on Linux
        if platform.system() == "Linux":
            try:
                subprocess.run(['which', 'xdotool', 'import', 'convert'], check=False, capture_output=True)
                # Check if tools are available
                result = subprocess.run(['command', '-v', 'xdotool'], shell=True, capture_output=True)
                if result.returncode != 0:
                    print("xdotool not found, recommending installation")
                    return {
                        "success": False,
                        "error": "Required tool 'xdotool' not found",
                        "message": "Please install xdotool with: sudo apt-get install xdotool"
                    }
                result = subprocess.run(['command', '-v', 'convert'], shell=True, capture_output=True)
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": "Required tool 'ImageMagick' not found",
                        "message": "Please install ImageMagick with: sudo apt-get install imagemagick"
                    }
            except Exception as e:
                print(f"Error checking for required tools: {e}")
        
        # Import the backend function - now using analyze_window instead of caption_window
        from cv_and_screenshots import analyze_window
        
        # Activate the window first to bring it to front
        activate_result = activate_window(window_id=window_id, window_title=window_title)
        if not activate_result:
            print(f"Warning: Failed to activate window {window_id} / {window_title}")
            # Continue anyway as the screenshot might still work
        
        # Call the backend function to analyze the window and extract UI elements
        result = analyze_window(window_title=window_title, window_id=window_id)
        
        if result is None:
            return {
                "success": False,
                "error": "Window analysis returned None",
                "message": "Failed to capture or analyze window. Check if the window exists and is visible."
            }
            
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "message": "Error analyzing window content."
            }
        
        # Create a comprehensive response with all the UI elements
        response = {
            "success": True,
            "caption": result.get("caption", ""),
            "text_elements": result.get("text_elements", []),
            "buttons": result.get("buttons", []),
            "checkboxes": result.get("checkboxes", []),
            "ui_regions": result.get("ui_regions", []),
            "window_info": result.get("window_info", {})
        }
        
        return response
    except Exception as e:
        import traceback
        print(f"Error analyzing window: {str(e)}")
        traceback.print_exc()
        error_details = traceback.format_exc()
        return {
            "success": False, 
            "error": str(e), 
            "details": error_details,
            "message": "Failed to analyze window. Check if the window exists and if necessary screenshot tools are installed."
        }

@app.get("/caption_window", operation_id="caption_window")
async def api_caption_window(window_id: str, window_title: Optional[str] = None):
    """Get only a caption description of a window using BLIP analysis.
    
    This lightweight endpoint captures a screenshot of the specified window and uses BLIP
    to generate a textual description of its contents without performing other analysis.
    
    Args:
        window_id: ID of the window to caption
        window_title: Title of the window to caption (optional, used if window_id not found)
        
    Returns:
        Caption description of the window contents
    """
    try:
        print(f"Captioning window with ID {window_id} and title {window_title}")
        
        # First ensure that required tools are available on Linux
        if platform.system() == "Linux":
            try:
                subprocess.run(['which', 'xdotool', 'import', 'convert'], check=False, capture_output=True)
                # Try to install if not available
                result = subprocess.run(['command', '-v', 'xdotool'], shell=True, capture_output=True)
                if result.returncode != 0:
                    print("xdotool not found, recommending installation")
                    return {
                        "success": False,
                        "error": "Required tool 'xdotool' not found",
                        "message": "Please install xdotool with: sudo apt-get install xdotool"
                    }
                result = subprocess.run(['command', '-v', 'convert'], shell=True, capture_output=True)
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": "Required tool 'ImageMagick' not found",
                        "message": "Please install ImageMagick with: sudo apt-get install imagemagick"
                    }
            except Exception as e:
                print(f"Error checking for required tools: {e}")
        
        # Import the backend function to avoid name conflicts
        from cv_and_screenshots import caption_window as backend_caption_window
        
        # Activate the window first to bring it to front
        activate_result = activate_window(window_id=window_id, window_title=window_title)
        if not activate_result:
            print(f"Warning: Failed to activate window {window_id} / {window_title}")
            # Continue anyway as the screenshot might still work
        
        # Call the backend function to analyze the window
        result = backend_caption_window(window_title=window_title, window_id=window_id)
        
        if result is None:
            return {
                "success": False,
                "error": "Window analysis returned None",
                "message": "Failed to capture or analyze window. Check if the window exists and is visible."
            }
            
        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "message": "Error analyzing window content."
            }
        
        # Create a simplified response with just the caption
        response = {
            "success": True,
            "window_id": window_id,
            "caption": result.get("caption", "")
        }
        
        return response
    except Exception as e:
        import traceback
        print(f"Error captioning window: {str(e)}")
        traceback.print_exc()
        error_details = traceback.format_exc()
        return {
            "success": False,
            "error": str(e),
            "details": error_details,
            "message": "Failed to caption window. Check logs for details."
        }

@app.get("/click_window_element", operation_id="click_window_element")
async def api_click_window_element(window_id: str, element_x: int, element_y: int, 
                                  element_width: Optional[int] = None, element_height: Optional[int] = None,
                                  button: str = "left", clicks: int = 1):
    """Click on a UI element in a window using coordinates relative to the window.
    
    This endpoint correctly translates window-relative coordinates to screen coordinates
    before clicking, solving the coordinate translation problem between window analysis and clicking.
    
    Args:
        window_id: ID of the window
        element_x: X coordinate of the UI element relative to the window
        element_y: Y coordinate of the UI element relative to the window
        element_width: Width of the UI element (optional)
        element_height: Height of the UI element (optional)
        button: Mouse button ('left', 'right', 'middle')
        clicks: Number of clicks
    
    Returns:
        Success status
    """
    try:
        # First, get window information
        windows = get_open_windows()
        target_window = None
        
        for window in windows:
            if "id" in window and str(window["id"]) == str(window_id):
                target_window = window
                break
                
        if not target_window:
            return {
                "success": False, 
                "error": f"Window with ID {window_id} not found"
            }
            
        # Create element position object
        element_position = {
            "x": element_x,
            "y": element_y
        }
        
        # Add width and height if provided
        if element_width is not None:
            element_position["width"] = element_width
        if element_height is not None:
            element_position["height"] = element_height
            
        # Calculate center_x and center_y if width and height are available
        if element_width is not None and element_height is not None:
            element_position["center_x"] = element_x + element_width // 2
            element_position["center_y"] = element_y + element_height // 2
            
        # Click on the element
        result = click_window_element(target_window, element_position, button, clicks)
        
        if not result:
            return {
                "success": False,
                "error": "Failed to click window element. See logs for details."
            }
            
        return {"success": True}
    except Exception as e:
        import traceback
        print(f"Error in click_window_element: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/click_ui_element", operation_id="click_ui_element")
async def api_click_ui_element(window_id: str, element_index: int, element_type: str = "button", button: str = "left", clicks: int = 1):
    """Click on a UI element that was detected by the analyze_window function.
    
    This endpoint first gets the window analysis results, then clicks on the specified UI element
    using its coordinates. This solves the coordinate translation problem by handling both
    analysis and clicking in one operation.
    
    Args:
        window_id: ID of the window
        element_index: Index of the UI element in the analysis results (0-based)
        element_type: Type of UI element ('button', 'checkbox', 'text', 'region')
        button: Mouse button ('left', 'right', 'middle')
        clicks: Number of clicks
    
    Returns:
        Success status
    """
    try:
        # First, analyze the window to get UI elements
        window_analysis = cv_analyze_window(window_id=window_id)
        
        if not window_analysis or "error" in window_analysis:
            error_msg = window_analysis.get("error", "Unknown error") if window_analysis else "Window analysis failed"
            return {
                "success": False, 
                "error": f"Failed to analyze window: {error_msg}"
            }
            
        # Get the UI elements list based on element_type
        ui_elements = []
        if element_type == "button":
            ui_elements = window_analysis.get("buttons", [])
        elif element_type == "checkbox":
            ui_elements = window_analysis.get("checkboxes", [])
        elif element_type == "text":
            ui_elements = window_analysis.get("extracted_text", [])
        elif element_type == "region":
            ui_elements = window_analysis.get("ui_regions", [])
        else:
            return {
                "success": False,
                "error": f"Unknown element type: {element_type}"
            }
            
        # Check if the element index is valid
        if not ui_elements or element_index < 0 or element_index >= len(ui_elements):
            return {
                "success": False,
                "error": f"Element index {element_index} out of range (0-{len(ui_elements)-1 if ui_elements else 0})"
            }
            
        # Get the element
        element = ui_elements[element_index]
        
        # Check if the element has position information
        if "position" not in element:
            return {
                "success": False,
                "error": f"Element has no position information: {element}"
            }
            
        # Get window info
        window_info = window_analysis.get("window_info", {})
        
        # Click on the element
        result = click_window_element(window_info, element["position"], button, clicks)
        
        if not result:
            return {
                "success": False,
                "error": "Failed to click UI element. See logs for details."
            }
            
        return {
            "success": True,
            "element": element,
            "element_type": element_type,
            "index": element_index
        }
    except Exception as e:
        import traceback
        print(f"Error in click_ui_element: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/screenshot_at_mouse", operation_id="screenshot_at_mouse")
async def screenshot_at_mouse(width: int = 300, height: int = 300):
    """Capture a screenshot of a rectangular area extending down and to the right from the current mouse position.
    Useful for capturing dropdown menus after a right-click.
    
    Args:
        width: Width of the area to capture in pixels (default: 300)
        height: Height of the area to capture in pixels (default: 300)
        
    Returns:
        JSON with base64 encoded image data and status
    """
    try:
        screenshot_data = get_screenshot_at_mouse(width, height)
        
        if screenshot_data is None:
            return {"success": False, "error": "Failed to capture screenshot at mouse position"}
        
        # Convert to base64 for API response
        base64_image = base64.b64encode(screenshot_data).decode('utf-8')
        
        return {
            "success": True, 
            "image_data": base64_image,
            "format": "png",
            "width": width,
            "height": height
        }
    except Exception as e:
        import traceback
        print(f"Error in screenshot_at_mouse: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/get_dropdown_at_mouse", operation_id="get_dropdown_at_mouse")
async def get_dropdown_at_mouse(width: int = 300, height: int = 300, include_image: bool = False):
    """Capture a screenshot at the current mouse position and extract dropdown menu options.
    This is especially useful after a right-click to capture context menus.
    
    Args:
        width: Width of the area to capture in pixels (default: 300)
        height: Height of the area to capture in pixels (default: 300)
        include_image: Whether to include the base64 encoded screenshot in the response (default: False)
        
    Returns:
        JSON with menu lines including text and absolute position information
    """
    try:
        # First capture the screenshot
        screenshot_data = get_screenshot_at_mouse(width, height)
        
        if screenshot_data is None:
            return {"success": False, "error": "Failed to capture screenshot at mouse position"}
        
        # Extract menu options from the screenshot
        menu_lines = extract_dropdown_options(screenshot_data)
        
        # Get current mouse position for reference
        mouse_x, mouse_y = pyautogui.position()
        
        response = {
            "success": True,
            "menu_lines": menu_lines,
            "count": len(menu_lines),
            "mouse_position": {
                "x": mouse_x,
                "y": mouse_y
            }
        }
        
        # Include the image data if requested
        if include_image:
            base64_image = base64.b64encode(screenshot_data).decode('utf-8')
            response["image_data"] = base64_image
            response["format"] = "png"
        
        return response
    except Exception as e:
        import traceback
        print(f"Error in get_dropdown_at_mouse: {str(e)}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Expose MCP server
mcp = FastApiMCP(app, name="pinoMCP")
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
