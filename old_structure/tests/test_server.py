"""Tests for the SSE server implementation."""

import json
import pytest
from unittest.mock import MagicMock, patch
import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp_cv_tool.core.screen_capture import ScreenCaptureError
from mcp_cv_tool.server import app

@pytest.fixture
def mock_screen_capture():
    """Mock ScreenCapture class."""
    with patch("mcp_cv_tool.server.ScreenCapture") as mock:
        mock_instance = MagicMock()
        mock_instance.platform = "test_platform"
        mock_instance.capture_screen.return_value = MagicMock(
            size=(1920, 1080),
            rgb=b"test_image_data"
        )
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def test_client(mock_screen_capture):
    """Test client fixture."""
    return TestClient(app)

def test_health_check(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_sse_endpoint_connection(test_client, mock_screen_capture):
    """Test SSE endpoint connection and initial data stream."""
    response = test_client.get("/sse", stream=True)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"].lower()

    event_processed = False
    async for line in response.aiter_lines():
        if line and line.startswith("data:"):
            event_data = json.loads(line.replace("data:", "").strip())
            assert "platform" in event_data
            assert "resolution" in event_data
            assert "timestamp" in event_data
            assert "image" in event_data
            event_processed = True
            break
    await response.aclose()
    assert event_processed, "No event was processed from SSE stream"

@pytest.mark.asyncio
async def test_sse_endpoint_error_handling(test_client, mock_screen_capture):
    """Test SSE endpoint error handling."""
    mock_instance = mock_screen_capture.return_value
    mock_instance.capture_screen.side_effect = ScreenCaptureError("Test error")

    response = test_client.get("/sse", stream=True)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"].lower()

    event_processed = False
    async for line in response.aiter_lines():
        if line and line.startswith("data:"):
            event_data = json.loads(line.replace("data:", "").strip())
            assert event_data.get("error") == "Test error"
            event_processed = True
            break
    await response.aclose()
    assert event_processed, "No error event was processed from SSE stream"

@pytest.mark.asyncio
async def test_sse_endpoint_disconnection(test_client, mock_screen_capture):
    """Test SSE endpoint client disconnection handling."""
    response = test_client.get("/sse", stream=True)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"].lower()

    event_processed = False
    async for line in response.aiter_lines(): # Ensure stream is active by reading one event
        if line and line.startswith("data:"):
            event_processed = True
            break
    assert event_processed, "Stream did not start before disconnection test"
    await response.aclose() # Simulate client disconnecting
    # Further checks would require inspecting server logs or state if applicable

def test_startup_event(test_client, mock_screen_capture):
    """Test application startup event."""
    assert hasattr(app.state, "screen_capture")
    mock_screen_capture.assert_called_once()

def test_shutdown_event(test_client, mock_screen_capture):
    """Test application shutdown event."""
    # This test assumes the lifespan's shutdown will be triggered by TestClient context ending.
    # For more direct testing, one might need to manage app lifecycle explicitly.
    # We check if the .close() method on the mock sct object was called.
    # Note: The lifespan should ensure sct exists before trying to close.
    with TestClient(app) as client: # Re-instantiate client to trigger lifespan
        pass # Trigger startup/shutdown
    
    # Depending on exact timing and fixture scope, direct assertion on app.state might be tricky
    # after the app context is fully closed. Mock assertion is more reliable here.
    app.state.screen_capture.sct.close.assert_called_once()

@pytest.mark.asyncio
async def test_global_exception_handler(test_client):
    """Test global exception handler."""
    route_path = "/test-error-global-async-final" 
    # Ensure route is not already defined from a previous failed run
    # This is a simple way, more robust would be to reset app.routes for full isolation
    app.routes = [route for route in app.routes if route.path != route_path]

    @app.get(route_path)
    async def test_error_route_final():
        raise ValueError("Global Test Error Final")

    response = test_client.get(route_path)
    assert response.status_code == 500
    assert response.json() == {"error": "Global Test Error Final"} 