"""Tests for the screen capture module."""

import platform
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mss.exception import ScreenShotError

from mcp_cv_tool.core.screen_capture import (
    CaptureConfig,
    PlatformNotSupportedError,
    ScreenCapture,
    ScreenCaptureError,
)

@pytest.fixture
def mock_mss():
    """Create a mock MSS instance."""
    with patch("mcp_cv_tool.core.screen_capture.mss") as mock:
        mock_instance = MagicMock()
        mock_instance.monitors = [{"top": 0, "left": 0, "width": 1920, "height": 1080}]
        
        # Mock screenshot
        screenshot = MagicMock()
        screenshot.rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        screenshot.size = (1920, 1080)
        screenshot.width = 1920
        screenshot.height = 1080
        screenshot.timestamp = 123456.789
        
        mock_instance.grab.return_value = screenshot
        mock.return_value = mock_instance
        yield mock

@pytest.fixture
def screen_capture(mock_mss):
    """Create a ScreenCapture instance with mocked MSS."""
    return ScreenCapture()

@pytest.mark.asyncio
async def test_capture_screen_success(screen_capture):
    """Test successful screen capture."""
    result = await screen_capture.capture_screen()
    
    assert isinstance(result, dict)
    assert "image" in result
    assert "platform" in result
    assert "resolution" in result
    assert "timestamp" in result
    
    assert isinstance(result["image"], np.ndarray)
    assert result["platform"] == platform.system()
    assert result["resolution"] == (1920, 1080)
    assert isinstance(result["timestamp"], float)

@pytest.mark.asyncio
async def test_capture_screen_error(mock_mss):
    """Test screen capture error handling."""
    mock_mss.return_value.grab.side_effect = ScreenShotError("Test error")
    
    screen_capture = ScreenCapture()
    
    with pytest.raises(ScreenCaptureError) as exc_info:
        await screen_capture.capture_screen()
    
    assert "Failed to capture screen" in str(exc_info.value)

def test_unsupported_platform():
    """Test initialization with unsupported platform."""
    with patch("platform.system", return_value="Unsupported"):
        with pytest.raises(PlatformNotSupportedError) as exc_info:
            ScreenCapture()
        
        assert "Platform Unsupported not supported" in str(exc_info.value)

def test_custom_config():
    """Test initialization with custom configuration."""
    config = CaptureConfig(
        monitor_number=1,
        capture_interval=0.2,
        compression_quality=90
    )
    
    with patch("mcp_cv_tool.core.screen_capture.mss"):
        capture = ScreenCapture(config)
        assert capture.config.monitor_number == 1
        assert capture.config.capture_interval == 0.2
        assert capture.config.compression_quality == 90

def test_context_manager():
    """Test context manager functionality."""
    with patch("mcp_cv_tool.core.screen_capture.mss") as mock_mss:
        mock_instance = MagicMock()
        mock_mss.return_value = mock_instance
        
        with ScreenCapture() as capture:
            assert isinstance(capture, ScreenCapture)
        
        mock_instance.close.assert_called_once() 