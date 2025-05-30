"""
Screen capture module for the Cursor CV MCP server.
Provides functionality to capture screenshots across different platforms.
"""

import platform
import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import mss
import numpy as np

logger = logging.getLogger(__name__)

class ScreenCaptureError(Exception):
    """Exception raised for screen capture errors."""
    pass

@dataclass
class ScreenData:
    """Screen capture data."""
    size: Tuple[int, int]
    rgb: np.ndarray
    raw: Any

class ScreenCapture:
    """Screen capture utility with cross-platform support."""
    
    def __init__(self):
        """Initialize screen capture utility."""
        self.platform = platform.system()
        try:
            self.sct = mss.mss()
            self.monitors = self.sct.monitors
            logger.info(f"Initialized screen capture on {self.platform}, {len(self.monitors)} monitors detected")
        except Exception as e:
            logger.error(f"Failed to initialize screen capture: {e}")
            raise ScreenCaptureError(f"Failed to initialize screen capture: {e}")
    
    def capture_screen(self, monitor_num: int = 1) -> ScreenData:
        """
        Capture screen from specified monitor.
        
        Args:
            monitor_num: Monitor number (1-based index)
            
        Returns:
            ScreenData with captured screen information
            
        Raises:
            ScreenCaptureError: If screen capture fails
        """
        try:
            # Check if monitor exists
            if monitor_num >= len(self.monitors):
                raise ScreenCaptureError(f"Monitor {monitor_num} not found. Available monitors: {len(self.monitors)}")
            
            # Capture screen
            monitor = self.monitors[monitor_num]
            screenshot = self.sct.grab(monitor)
            
            # Convert to RGB format
            img = np.array(screenshot)
            
            return ScreenData(
                size=(screenshot.width, screenshot.height),
                rgb=img,
                raw=screenshot
            )
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            raise ScreenCaptureError(f"Screen capture failed: {e}")
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'sct'):
            self.sct.close() 