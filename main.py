import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi_mcp import FastApiMCP
from cv_and_screenshots import get_available_monitors, get_screenshot, analyze_image, get_screenshot_with_analysis, find_text_in_image as cv_find_text_in_image, analyze_window as cv_analyze_window
from mouse_control import mouse_move as move_mouse_function, mouse_click as click_mouse_function
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
    """Get information about the system, including monitors, operating system, and keyboard layout"""
    monitors = get_available_monitors()
    return {"monitors": monitors, "os": get_os_info(), "keyboard_layout": keyboard_layout_info(), "keyboard_layout_name": keyboard_layout_info().name}

@app.get("/screenshot_with_analysis", operation_id="screenshot_with_analysis")
async def screenshot_with_analysis(monitor_id: int, target_string: Optional[str] = None):
    """Analyze screenshot of monitor and find buttons
    target_string is an optional parameter to find a specific text in the image and return the position of the text"""
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

        return {
            "analysis": analysis,
            "buttons": ui_buttons,
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

@app.get("/click_ui_element", operation_id="click_ui_element")
async def click_ui_element(
    monitor_id: int, 
    element_type: str, 
    search_text: Optional[str] = None,
    index: int = 0, 
    button: str = "left", 
    clicks: int = 1
):
    """Click on a UI element identified by type and optional text.

    Args:
        monitor_id: ID of the monitor to capture and click on
        element_type: Type of element to click ('text', 'button', 'checkbox', 'region')
        search_text: Text to search for (if element_type is 'text' or 'region')
        index: Index of the element to click if multiple matches are found
        button: Mouse button to click with ('left', 'right', 'middle')
        clicks: Number of clicks
        
    Returns:
        Success status and information about the clicked element
    """
    try:
        # Validate monitor_id first
        monitors_info = get_available_monitors()
        available_monitors = [mon["id"] for mon in monitors_info.get("monitors", [])]
        primary_monitor = monitors_info.get("primary", None)
        
        if monitor_id not in available_monitors:
            return {
                "success": False,
                "error": f"Invalid monitor ID: {monitor_id}",
                "available_monitors": available_monitors,
                "primary_monitor": primary_monitor,
                "tip": "Please specify a valid monitor ID from the available monitors list."
            }
        
        # Get screenshot with analysis
        screenshot_data, analysis, ui_buttons, ui_checkboxes, ui_regions = get_screenshot_with_analysis(monitor_id)
        if screenshot_data is None:
            return {
                "success": False, 
                "error": "Failed to capture screenshot",
                "monitor_info": {
                    "requested_monitor": monitor_id,
                    "available_monitors": available_monitors,
                    "primary_monitor": primary_monitor
                }
            }
            
        # Default values
        element_position = None
        element_info = None
        
        # Process the element based on its type
        if element_type == "text" and search_text:
            # Find text in the image
            text_matches = cv_find_text_in_image(image_data=screenshot_data, target=search_text)
            
            # Check if any matches were found
            if not text_matches or len(text_matches) <= index:
                return {
                    "success": False,
                    "error": f"Text '{search_text}' not found or index {index} out of bounds",
                    "matches_found": len(text_matches) if text_matches else 0
                }
            
            # Get the specified match
            matched_word, bounds = text_matches[index]
            x, y, w, h = bounds
            element_position = {"x": x + w//2, "y": y + h//2}
            element_info = {"text": matched_word, "bounds": bounds}
            
        elif element_type == "button":
            if not ui_buttons or len(ui_buttons) <= index:
                return {
                    "success": False,
                    "error": f"No button found at index {index}",
                    "buttons_found": len(ui_buttons) if ui_buttons else 0
                }
                
            # Handle the new button format which is a dict with 'text' and 'position'
            button = ui_buttons[index]
            button_text = button["text"] if "text" in button else "Unknown"
            position = button["position"]
            x, y = position["center_x"], position["center_y"]
            
            # Move and click at the button position
            result = click_mouse_function(x, y, button=button_text, clicks=clicks, monitor=monitor_id)
            
            return {
                "success": result,
                "clicked_element": {
                    "type": "button",
                    "text": button_text,
                    "position": position,
                    "index": index
                }
            }
            
        elif element_type == "checkbox":
            # Find checkbox elements
            if not ui_checkboxes or len(ui_checkboxes) <= index:
                return {
                    "success": False,
                    "error": f"No checkboxes found or index {index} out of bounds",
                    "checkboxes_found": len(ui_checkboxes) if ui_checkboxes else 0
                }
                
            # Get the specified checkbox
            checkbox = ui_checkboxes[index]
            element_position = {"x": checkbox["position"]["center_x"], "y": checkbox["position"]["center_y"]}
            element_info = checkbox
            
        elif element_type == "region":
            # Find image regions that match the description
            if not ui_regions:
                return {
                    "success": False,
                    "error": "No image regions detected",
                    "regions_found": 0
                }
                
            # If search text is provided, find regions that match the description
            matching_regions = []
            if search_text:
                for region in ui_regions:
                    if search_text.lower() in region["caption"].lower():
                        matching_regions.append(region)
            else:
                matching_regions = ui_regions
                
            if not matching_regions or len(matching_regions) <= index:
                return {
                    "success": False,
                    "error": f"No matching regions found for '{search_text}' or index {index} out of bounds",
                    "matching_regions_found": len(matching_regions),
                    "total_regions_found": len(ui_regions)
                }
                
            # Get the specified region
            region = matching_regions[index]
            element_position = {"x": region["position"]["center_x"], "y": region["position"]["center_y"]}
            element_info = {"caption": region["caption"], "position": region["position"]}
        else:
            return {
                "success": False,
                "error": f"Invalid element_type: {element_type}",
                "valid_types": ["text", "button", "checkbox", "region"],
                "tip": "Please specify a valid element type from the list."
            }
            
        # Check if element was found
        if not element_position or "x" not in element_position or "y" not in element_position:
            return {
                "success": False,
                "error": f"Failed to get position for {element_type}",
                "element_info": element_info
            }
            
        # Click the element
        x = element_position["x"]
        y = element_position["y"]
        
        # Move mouse to the element
        mouse_move_result = mouse_move(x, y, monitor_id)
        if not mouse_move_result.get("success", False):
            return {
                "success": False,
                "error": f"Failed to move mouse to position ({x}, {y})",
                "mouse_move_result": mouse_move_result
            }
            
        # Perform the click
        mouse_click_result = mouse_click(x, y, button, clicks, monitor_id)
        if not mouse_click_result.get("success", False):
            return {
                "success": False,
                "error": f"Failed to click at position ({x}, {y})",
                "mouse_click_result": mouse_click_result,
                "element_info": element_info
            }
            
        # Return success result
        return {
            "success": True,
            "element_type": element_type,
            "position": element_position,
            "button": button,
            "clicks": clicks,
            "monitor_id": monitor_id,
            "element_info": element_info
        }
    except Exception as e:
        import traceback
        print(f"Error clicking UI element: {str(e)}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "details": traceback.format_exc()
        }

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
async def api_launch_application(app_name: str, app_path: Optional[str] = None):
    """Launches an application. If this fails, you can try to launch the application like a user, eg superkey + search on debian via other functions.
    After running this function, you should use list_windows to check if the application is open.
    
    Args:
        app_name: Name of the application to launch
        app_path: Full path to the application executable (optional)
        
    Returns:
        Success status
    """
    try:
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

@app.get("/open_downloads_folder", operation_id="open_downloads_folder")
async def api_open_downloads_folder():
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

@app.get("/folder_contents", operation_id="folder_contents")
async def api_folder_contents(folder_path: str):
    """Get the contents of a folder"""
    try:
        result = get_folder_contents(folder_path)
        return result
    except Exception as e:
        print(f"Error getting folder contents: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.get("/analyze_window", operation_id="analyze_window")
async def analyze_window(window_id: str, window_title: Optional[str] = None):
    """Analyze a window and return information about it
    
    Uses computer vision and BLIP to analyze the contents of a window.
    Detects UI elements like buttons and checkboxes, and provides a caption
    describing what's in the window.
    
    Args:
        window_id: ID of the window to analyze
        window_title: Title of the window to analyze (optional, used if window_id not found)
        
    Returns:
        Analysis results including caption, UI elements, and window metadata
    """
    try:
        # Try to ensure xdotool/screenshot tools are available on Linux
        if platform.system() == "Linux":
            try:
                # Check for xdotool
                subprocess.run(['which', 'xdotool'], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("xdotool not installed, attempting to install...")
                try:
                    subprocess.run(['sudo', 'apt-get', 'update'], check=False)
                    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'xdotool'], check=False)
                except Exception as e:
                    print(f"Failed to install xdotool: {e}")
            
            # Check for imagemagick (for import command)
            try:
                subprocess.run(['which', 'import'], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print("imagemagick not installed, attempting to install...")
                try:
                    subprocess.run(['sudo', 'apt-get', 'update'], check=False)
                    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'imagemagick'], check=False)
                except Exception as e:
                    print(f"Failed to install imagemagick: {e}")
        
        print(f"Analyzing window with ID {window_id} and title {window_title}")
        # Call the backend function to analyze the window
        result = cv_analyze_window(window_title=window_title, window_id=window_id)
        
        if result is None:
            return {"success": False, "error": "Window analysis returned None"}
            
        if "error" in result:
            return {"success": False, "error": result["error"]}
            
        # Format checkbox results to be more user-friendly
        formatted_checkboxes = []
        if "ui_checkboxes" in result and result["ui_checkboxes"]:
            # The checkboxes are already in the correct format with 'type' and 'position' keys
            formatted_checkboxes = result["ui_checkboxes"]
        
        # Process UI regions to remove binary data or encode it as base64
        processed_regions = []
        if "ui_regions" in result and result["ui_regions"]:
            for region in result["ui_regions"]:
                # Make a copy of the region without the binary image data
                processed_region = region.copy()
                if "image_data" in processed_region:
                    # Either remove the image data
                    del processed_region["image_data"]
                    # Or encode it as base64 (uncomment below if you want to keep the images)
                    # processed_region["image_data_base64"] = base64.b64encode(processed_region["image_data"]).decode('utf-8')
                processed_regions.append(processed_region)
        
        # Create response with formatted data
        response = {
            "success": True,
            "window_id": window_id,
            "window_title": window_title,
            "caption": result.get("caption", ""),
            "buttons": result.get("buttons", []),
            "checkboxes": formatted_checkboxes,
            "ui_regions": processed_regions,
            "window_info": result.get("window_info", {})
        }
        
        return response
    except Exception as e:
        import traceback
        print(f"Error analyzing window: {str(e)}")
        traceback.print_exc()
        error_traceback = traceback.format_exc()
        return {
            "success": False, 
            "error": str(e), 
            "details": error_traceback,
            "message": "Failed to analyze window. Check if the window exists and if necessary screenshot tools are installed."
        }

# Expose MCP server
mcp = FastApiMCP(app, name="pinoMCP")
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
