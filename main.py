import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi_mcp import FastApiMCP
from cv_and_screenshots import get_available_monitors, get_screenshot, analyze_image, get_screenshot_with_analysis, find_text_in_image as cv_find_text_in_image
from mouse_control import mouse_move as move_mouse_function, mouse_click as click_mouse_function
from keyboard_control import keyboard_type_text, keyboard_press_keys
from window_control import get_open_windows, activate_window, launch_application
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from microphone import listen_to_microphone
import os
from fastapi.responses import FileResponse
from tts_service import tts_service

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

@app.get("/list_monitors", operation_id="list_monitors")
async def list_monitors():
    """List all available monitors and their dimensions"""
    monitors = get_available_monitors()
    return {"monitors": monitors}

@app.get("/screenshot_with_analysis", operation_id="screenshot_with_analysis")
async def screenshot_with_analysis(monitor_id: int, target_string: Optional[str] = None):
    """Analyze screenshot of monitor and find buttons
    target_string is an optional parameter to find a specific text in the image and return the position of the text"""
    try:
        # Print debug info
        print(f"Requested screenshot with monitor_id={monitor_id}, target_string={target_string}")
        
        # Get screenshot with analysis
        screenshot_data, analysis, ui_buttons, ui_checkboxes = get_screenshot_with_analysis(monitor_id)
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
            for checkbox_type, bounds in ui_checkboxes:
                x, y, w, h = bounds
                formatted_checkboxes.append({
                    "type": checkbox_type,
                    "position": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "center_x": x + w//2,
                        "center_y": y + h//2
                    }
                })

        return {
            "analysis": analysis,
            "buttons": ui_buttons,
            "checkboxes": formatted_checkboxes,
            "text_matches": text_matches
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
    return {"success": move_mouse_function(x, y, monitor)}

@app.get("/mouse_click", operation_id="mouse_click")
async def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1, monitor: int = 0):
    """Click the mouse at a specific position on a specific monitor
    Double check that we are on the correct monitor before moving the mouse"""
    return {"success": click_mouse_function(x, y, button, clicks, monitor)}

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
        element_type: Type of element to click ('text', 'button', 'checkbox')
        search_text: Text to search for (if element_type is 'text')
        index: Index of the element to click if multiple matches are found
        button: Mouse button to click with ('left', 'right', 'middle')
        clicks: Number of clicks
        
    Returns:
        Success status and information about the clicked element
    """
    try:
        # Get screenshot with analysis
        screenshot_data, analysis, ui_buttons, ui_checkboxes = get_screenshot_with_analysis(monitor_id)
        if screenshot_data is None:
            raise HTTPException(status_code=500, detail="Failed to capture screenshot")
        
        target_element = None
        element_list = []
        
        # Find the requested element
        if element_type == "text" and search_text:
            # Find text in image
            text_matches = cv_find_text_in_image(image_data=screenshot_data, target=search_text)
            if text_matches:
                element_list = []
                for word, bounds in text_matches:
                    x, y, w, h = bounds
                    element_list.append({
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
        
        elif element_type == "button":
            if ui_buttons:
                element_list = []
                for text, bounds in ui_buttons:
                    # If search_text is provided, filter by text
                    if search_text and search_text.lower() not in text.lower():
                        continue
                        
                    x, y, w, h = bounds
                    element_list.append({
                        "text": text,
                        "position": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "center_x": x + w//2,
                            "center_y": y + h//2
                        }
                    })
                    
        elif element_type == "checkbox":
            if ui_checkboxes:
                element_list = []
                for checkbox_type, bounds in ui_checkboxes:
                    # If search_text is provided, filter by checkbox type
                    if search_text and search_text.lower() != checkbox_type.lower():
                        continue
                        
                    x, y, w, h = bounds
                    element_list.append({
                        "type": checkbox_type,
                        "position": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "center_x": x + w//2,
                            "center_y": y + h//2
                        }
                    })
        
        # Check if we found any elements
        if not element_list:
            return {
                "success": False,
                "error": f"No {element_type} elements found" + (f" matching '{search_text}'" if search_text else "")
            }
        
        # Get the element at the specified index
        if index < 0 or index >= len(element_list):
            return {
                "success": False,
                "error": f"Index {index} out of range (0-{len(element_list)-1})"
            }
            
        target_element = element_list[index]
        
        # Get center coordinates for clicking
        center_x = target_element["position"]["center_x"]
        center_y = target_element["position"]["center_y"]
        
        # Click at the center of the element
        click_success = click_mouse_function(
            x=center_x, 
            y=center_y, 
            button=button, 
            clicks=clicks, 
            monitor=monitor_id
        )
        
        return {
            "success": click_success,
            "element": target_element,
            "clicked_at": {
                "x": center_x,
                "y": center_y,
                "monitor": monitor_id
            }
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error clicking UI element: {str(e)}")
        print(error_details)
        return {
            "success": False,
            "error": str(e),
            "details": error_details
        }

@app.get("/type_text", operation_id="type_text")
async def type_text(text: str):
    """Type text at a specific position on a specific monitor
    For button combinations, use the press_keys endpoint. This can only produce text, not button combinations."""
    return {"success": keyboard_type_text(text)}

@app.post("/press_keys", operation_id="press_keys")
async def press_keys(data: PressKeysInput):
    """Press keys on the keyboard, keys is a list of strings. All keys are pressed at the same time.
    Example: press_keys["ctrl", "tab"]"""
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
    """Launch an application.
    
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

# Expose MCP server
mcp = FastApiMCP(app, name="pinoMCP")
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
