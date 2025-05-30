import base64
from fastapi import FastAPI, Body, HTTPException
from fastapi_mcp import FastApiMCP
from cv_and_screenshots import get_available_monitors, get_screenshot, analyze_image, get_screenshot_with_analysis, find_text_in_image as cv_find_text_in_image
from mouse_control import mouse_move as move_mouse_function, mouse_click as click_mouse_function
from keyboard_control import keyboard_type_text, keyboard_press_keys
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union

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


# Expose MCP server
mcp = FastApiMCP(app, name="pinoMCP")
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
