"""
SSE server implementation for the Cursor CV MCP server.
Provides real-time screen capture data through Server-Sent Events.
"""

import asyncio
import json
import logging
import uvicorn
import sys
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from sse_starlette.sse import EventSourceResponse

from mcp_cv_tool.core.screen_capture import ScreenCapture, ScreenCaptureError

# Check if we have pytesseract for OCR
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Check if we have pyautogui for mouse control
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ScreenData(BaseModel):
    """Screen capture data model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    platform: str
    resolution: tuple[int, int]
    timestamp: datetime
    image: bytes | None = None
    error: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    app.state.screen_capture = ScreenCapture()
    yield
    app.state.screen_capture.sct.close()

app = FastAPI(title="Cursor CV MCP Server", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error("unhandled_exception", extra={"error": str(exc), "path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}

# Define request models using Pydantic
class ScreenshotRequest(BaseModel):
    monitor: int = 1
    output_file: Optional[str] = None
    debug: bool = False
    analyze: bool = True

class FindElementRequest(BaseModel):
    screenshot_path: str
    element_type: str
    text: Optional[str] = None
    confidence: float = 0.7

class MouseClickRequest(BaseModel):
    x: int
    y: int
    button: str = "left"

class KeyboardInputRequest(BaseModel):
    text: str
    interval: float = 0.1  # Time between keypresses in seconds

# MCP handler functions

@app.post("/mcp_screenshot_capture")
@app.post("/mcp_cv_tool/mcp_screenshot_capture")
@app.post("/mcp-api/mcp_screenshot_capture")
async def mcp_screenshot_capture(request: ScreenshotRequest) -> Dict[str, Any]:
    """
    MCP handler for capturing screenshots.
    """
    try:
        if request.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
        
        logger.info(f"Capturing screenshot from monitor {request.monitor}")
        screen = app.state.screen_capture.capture_screen(request.monitor)
        
        # Generate a temporary file if output_file is not provided
        output_file = request.output_file
        if not output_file:
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(temp_dir, f"cursor_screenshot_{timestamp}.jpg")
        
        # Save the screenshot
        cv2.imwrite(output_file, np.array(screen.rgb))
        logger.info(f"Screenshot saved to {output_file}")
        
        # Perform OCR analysis if requested
        text_content = ""
        if request.analyze:
            if HAS_TESSERACT:
                try:
                    # Convert to grayscale for better OCR
                    img_array = np.array(screen.rgb)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    
                    # Apply thresholding to improve OCR
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    
                    # Perform OCR
                    text_content = pytesseract.image_to_string(binary)
                    
                    logger.info(f"OCR analysis completed successfully")
                except Exception as e:
                    logger.error(f"OCR analysis failed: {e}")
                    text_content = f"OCR analysis failed: {str(e)}"
            else:
                text_content = "OCR analysis not available. Install pytesseract for text extraction."
                logger.warning("OCR analysis requested but pytesseract is not installed")
        
        return {
            "success": True,
            "screenshot_path": output_file,
            "resolution": screen.size,
            "timestamp": datetime.now().isoformat(),
            "text_content": text_content if request.analyze else None,
        }
    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }

@app.post("/mcp_find_element")
@app.post("/mcp_cv_tool/mcp_find_element")
@app.post("/mcp-api/mcp_find_element")
async def mcp_find_element(request: FindElementRequest) -> Dict[str, Any]:
    """
    MCP handler for finding UI elements.
    """
    try:
        logger.info(f"Finding {request.element_type} elements in {request.screenshot_path}")
        
        # Check if screenshot exists
        if not os.path.exists(request.screenshot_path):
            return {
                "success": False,
                "error": f"Screenshot not found: {request.screenshot_path}",
            }
        
        # In a real implementation, you would use CV techniques to find elements
        # This is a placeholder implementation
        return {
            "success": True,
            "elements": [
                {
                    "type": request.element_type,
                    "text": request.text,
                    "confidence": request.confidence,
                    "position": {
                        "x": 100,
                        "y": 100,
                        "width": 200,
                        "height": 50,
                    },
                }
            ],
        }
    except Exception as e:
        logger.error(f"Find element failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }

@app.post("/mcp_mouse_click")
@app.post("/mcp_cv_tool/mcp_mouse_click")
@app.post("/mcp-api/mcp_mouse_click")
async def mcp_mouse_click(request: MouseClickRequest) -> Dict[str, Any]:
    """
    MCP handler for mouse clicks.
    """
    try:
        logger.info(f"Clicking at ({request.x}, {request.y}) with {request.button} button")
        
        if not HAS_PYAUTOGUI:
            logger.warning("PyAutoGUI not installed. Cannot perform actual mouse click.")
            return {
                "success": False,
                "error": "PyAutoGUI not installed. Install with: pip install pyautogui",
                "message": f"Would click at ({request.x}, {request.y}) with {request.button} button"
            }
        
        # Move mouse to the specified position
        pyautogui.moveTo(request.x, request.y, duration=0.25)
        
        # Perform the click with the specified button
        if request.button.lower() == "right":
            pyautogui.rightClick()
        elif request.button.lower() == "middle":
            pyautogui.middleClick()
        else:  # Default to left click
            pyautogui.click()
            
        return {
            "success": True,
            "message": f"Clicked at ({request.x}, {request.y}) with {request.button} button",
        }
    except Exception as e:
        logger.error(f"Mouse click failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }

@app.post("/mcp_keyboard_input")
@app.post("/mcp_cv_tool/mcp_keyboard_input")
@app.post("/mcp-api/mcp_keyboard_input")
async def mcp_keyboard_input(request: KeyboardInputRequest) -> Dict[str, Any]:
    """
    MCP handler for keyboard input.
    """
    try:
        logger.info(f"Typing text: {request.text}")
        
        if not HAS_PYAUTOGUI:
            logger.warning("PyAutoGUI not installed. Cannot perform keyboard input.")
            return {
                "success": False,
                "error": "PyAutoGUI not installed. Install with: pip install pyautogui",
                "message": f"Would type: {request.text}"
            }
        
        # Handle special keys
        special_keys = {
            "\\n": "enter",
            "\\t": "tab",
            "\\b": "backspace",
            "\\e": "escape",
            "\\s": "space",
            "\\u": "up",
            "\\d": "down",
            "\\l": "left",
            "\\r": "right"
        }
        
        # Check if this is a special key sequence
        if request.text in special_keys:
            pyautogui.press(special_keys[request.text])
            logger.info(f"Pressed special key: {special_keys[request.text]}")
        else:
            # Type the text with the specified interval
            pyautogui.write(request.text, interval=request.interval)
        
        return {
            "success": True,
            "message": f"Typed text: {request.text}",
        }
    except Exception as e:
        logger.error(f"Keyboard input failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }

async def screen_capture_generator() -> AsyncGenerator[Dict[str, Any], None]:
    """Generate screen capture data."""
    while True:
        try:
            screen = app.state.screen_capture.capture_screen()
            data = ScreenData(
                platform=app.state.screen_capture.platform,
                resolution=screen.size,
                timestamp=datetime.now(),
                image=cv2.imencode(".jpg", np.array(screen.rgb))[1].tobytes(),
            )
            yield {"data": data.model_dump_json()}
            await asyncio.sleep(0.1)  # Adjust rate as needed
        except ScreenCaptureError as e:
            yield {"data": ScreenData(
                platform=app.state.screen_capture.platform,
                resolution=(0, 0),
                timestamp=datetime.now(),
                error=str(e),
            ).model_dump_json()}
            await asyncio.sleep(1)  # Wait longer on error
        except Exception as e:
            logger.error("screen_capture_error", extra={"error": str(e)})
            yield {"data": ScreenData(
                platform=app.state.screen_capture.platform,
                resolution=(0, 0),
                timestamp=datetime.now(),
                error=str(e),
            ).model_dump_json()}
            await asyncio.sleep(1)

@app.get("/sse")
async def sse_endpoint() -> EventSourceResponse:
    """SSE endpoint for real-time screen capture data."""
    return EventSourceResponse(screen_capture_generator())

def run_server():
    """Run the server. This is the entry point called from the command line."""
    port = 8001
    host = "127.0.0.1"
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    logger.info(f"Starting Cursor CV MCP Server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server() 