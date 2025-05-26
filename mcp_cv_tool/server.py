"""
SSE server implementation for the Cursor CV MCP server.
Provides real-time screen capture data through Server-Sent Events.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from sse_starlette.sse import EventSourceResponse
from structlog import get_logger

from .core.screen_capture import CaptureConfig, ScreenCapture, ScreenCaptureError
from .context7_integration import FUNCTION_REGISTRY, call_function

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup
    logger.info("server_starting")
    app.state.capture_config = CaptureConfig()
    try:
        app.state.screen_capture = ScreenCapture(app.state.capture_config)
        logger.info("screen_capture_initialized")
    except Exception as e:
        logger.error("screen_capture_init_failed", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("server_shutting_down")
    if hasattr(app.state, "screen_capture"):
        app.state.screen_capture.sct.close()

app = FastAPI(
    title="Cursor CV MCP Server",
    description="""
    Server for screenshot capture and analysis with MCP (Modular Command Protocol) support.
    
    ## MCP Standard
    
    This server implements the MCP standard with two main endpoints:
    
    - **GET /mcp/list_functions**: Lists all available MCP functions and their documentation
    - **POST /mcp/call_function**: Calls any MCP function with parameters
    
    ## Available Functions
    
    The following MCP functions are available:
    
    - **mcp_screenshot_capture**: Captures a screenshot from a specified monitor
    - **mcp_screenshot_analyze_image**: Analyzes an existing image
    - **mcp_screenshot_list_monitors**: Lists available monitors
    - **mcp_test_function**: Simple test function
    
    ## Usage Example
    
    ```python
    import requests
    
    # List available functions
    functions = requests.get("http://localhost:8001/mcp/list_functions").json()
    
    # Capture a screenshot
    result = requests.post(
        "http://localhost:8001/mcp/call_function",
        json={
            "function_name": "mcp_screenshot_capture",
            "params": {"monitor": 1}
        }
    ).json()
    ```
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

class ScreenData(BaseModel):
    """Pydantic model for screen capture data validation."""
    platform: str
    resolution: tuple[int, int]
    timestamp: float
    error: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "platform": "Linux",
                "resolution": (1920, 1080),
                "timestamp": 1645564800.0,
                "error": None
            }
        }
    )

@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/monitors")
async def get_monitors() -> dict:
    """Get information about available monitors."""
    try:
        # Get monitor information from screen capture
        monitors_info = app.state.screen_capture.get_monitors_info()
        return {
            "monitors": monitors_info,
            "primary": app.state.capture_config.monitor_number
        }
    except Exception as e:
        logger.error("monitor_info_error", error=str(e))
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get monitor information: {str(e)}"}
        )

@app.get("/sse")
async def sse_endpoint(
    request: Request, 
    monitor: int = Query(None, description="Monitor number to capture")
) -> EventSourceResponse:
    """SSE endpoint for real-time screen capture data.
    
    Args:
        request: FastAPI request object
        monitor: Monitor number to capture (optional)
    
    Returns:
        EventSourceResponse: SSE response with screen capture data
    """
    # Override monitor number if provided
    capture_config = None
    if monitor is not None:
        # Create a copy of the global config with the specified monitor
        capture_config = CaptureConfig(
            monitor_number=monitor,
            capture_interval=app.state.capture_config.capture_interval,
            max_retries=app.state.capture_config.max_retries,
            compression_quality=app.state.capture_config.compression_quality
        )
    
    async def event_generator() -> AsyncGenerator[dict, None]:
        """Generate SSE events with screen capture data."""
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("client_disconnected")
                    break

                try:
                    # Use specific config if provided
                    if capture_config:
                        capture_data = await app.state.screen_capture.capture_screen(capture_config)
                    else:
                        capture_data = await app.state.screen_capture.capture_screen()
                    
                    # Convert numpy array to JPEG bytes for efficient transmission
                    _, img_encoded = cv2.imencode(
                        '.jpg', 
                        capture_data["image"], 
                        [cv2.IMWRITE_JPEG_QUALITY, app.state.capture_config.compression_quality]
                    )
                    
                    # Create validated response data
                    screen_data = ScreenData(
                        platform=capture_data["platform"],
                        resolution=capture_data["resolution"],
                        timestamp=capture_data["timestamp"]
                    )

                    yield {
                        "data": {
                            "image": img_encoded.tobytes().hex(),  # Convert to hex for safe transmission
                            **screen_data.model_dump()
                        }
                    }

                except ScreenCaptureError as e:
                    logger.error("capture_error", error=str(e))
                    yield {
                        "data": ScreenData(
                            platform=app.state.screen_capture.platform,
                            resolution=(0, 0),
                            timestamp=0.0,
                            error=str(e)
                        ).model_dump()
                    }

                await asyncio.sleep(app.state.capture_config.capture_interval)

        except Exception as e:
            logger.error("sse_stream_error", error=str(e))
            yield {
                "data": {"error": f"Stream error: {str(e)}"}
            }

    return EventSourceResponse(event_generator())

class MCPFunction(BaseModel):
    name: str
    doc: Optional[str] = None

class MCPListResponse(BaseModel):
    success: bool
    functions: List[MCPFunction]

class MCPCallRequest(BaseModel):
    function_name: str
    params: Dict[str, Any] = {}

class MCPCallResponse(BaseModel):
    """Response model for MCP function calls."""
    success: bool
    error: Optional[str] = None
    message: Optional[str] = None
    screenshot_path: Optional[str] = None
    resolution: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    monitors: Optional[List[Dict[str, Any]]] = None
    primary: Optional[int] = None
    position: Optional[List[int]] = None
    text: Optional[str] = None
    button: Optional[str] = None
    clicks: Optional[int] = None
    key: Optional[str] = None
    area: Optional[Dict[str, Any]] = None
    element: Optional[Dict[str, Any]] = None  # Added for element detection

@app.get(
    "/mcp/list_functions", 
    response_model=MCPListResponse,
    tags=["MCP Standard"],
    summary="List available MCP functions",
    description="Returns a list of all available MCP functions with their documentation"
)
async def mcp_list_functions() -> dict:
    """List available MCP functions and their docstrings."""
    return {
        "success": True,
        "functions": [
            {"name": fn, "doc": FUNCTION_REGISTRY[fn].__doc__} for fn in FUNCTION_REGISTRY
        ]
    }

@app.post("/mcp/call_function", response_model=MCPCallResponse)
async def mcp_call_function(request: MCPCallRequest):
    """Call an MCP function."""
    logger.info("mcp_call_function", function=request.function_name)
    
    if request.function_name not in FUNCTION_REGISTRY:
        return {
            "success": False,
            "error": f"Unknown function: {request.function_name}",
            "available_functions": list(FUNCTION_REGISTRY.keys()),
        }
    
    try:
        result = FUNCTION_REGISTRY[request.function_name](request.params)
        # Ensure result is a valid response model
        for key in result:
            if key not in MCPCallResponse.__annotations__:
                logger.warning(f"Unexpected key in response: {key}")
                
        return result
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error("Error calling function", 
                    function=request.function_name,
                    error=str(e),
                    traceback=error_trace)
        
        return {
            "success": False,
            "error": f"Error calling function {request.function_name}: {str(e)}",
        }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for all unhandled exceptions."""
    logger.error("unhandled_exception", 
                 error=str(exc),
                 path=str(request.url.path))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    ) 