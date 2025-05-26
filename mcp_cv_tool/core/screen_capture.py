"""
Screen capture module for cross-platform screen capture functionality.
Supports MacOS, Windows 11, and Ubuntu 24.04.
"""

import platform
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from mss import mss
from mss.base import MSSBase
from mss.screenshot import ScreenShot
from structlog import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

logger = get_logger(__name__)

@dataclass
class CaptureConfig:
    """Configuration for screen capture."""
    monitor_number: int = 0
    capture_interval: float = 0.1  # seconds
    max_retries: int = 3
    compression_quality: int = 80  # 0-100, higher is better quality

class ScreenCaptureError(Exception):
    """Base exception for screen capture errors."""
    pass

class PlatformNotSupportedError(ScreenCaptureError):
    """Exception raised when platform is not supported."""
    pass

class ScreenCapture:
    """Cross-platform screen capture implementation using mss."""
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        """Initialize screen capture with optional configuration.
        
        Args:
            config: Optional capture configuration.
        
        Raises:
            PlatformNotSupportedError: If current platform is not supported.
        """
        self.config = config or CaptureConfig()
        self.platform = platform.system()
        
        if self.platform not in {"Darwin", "Windows", "Linux"}:
            raise PlatformNotSupportedError(f"Platform {self.platform} not supported")
        
        try:
            self.sct: MSSBase = mss()
            logger.info("screen_capture_initialized", 
                       platform=self.platform,
                       monitors=len(self.sct.monitors))
        except Exception as e:
            logger.error("screen_capture_init_failed", error=str(e))
            raise ScreenCaptureError(f"Failed to initialize screen capture: {e}")

    def get_monitors_info(self) -> List[Dict[str, Any]]:
        """Get information about available monitors.
        
        Returns:
            List of dictionaries containing monitor information
        """
        monitors_info = []
        for idx, monitor in enumerate(self.sct.monitors):
            # Skip the first monitor which is usually the "all monitors" virtual monitor
            if idx == 0 and len(self.sct.monitors) > 1:
                continue
                
            # Copy monitor info but skip some internal mss values
            monitor_data = {
                "width": monitor.get("width"),
                "height": monitor.get("height"),
                "left": monitor.get("left"),
                "top": monitor.get("top"),
                "name": f"Monitor {idx}"
            }
            
            # Detect primary monitor (usually at 0,0 position)
            if monitor.get("left") == 0 and monitor.get("top") == 0:
                monitor_data["is_primary"] = True
                
            monitors_info.append(monitor_data)
            
        return monitors_info

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def capture_screen(self, config: Optional[CaptureConfig] = None) -> Dict[str, Union[np.ndarray, str, Tuple[int, int]]]:
        """Capture screen content with retry mechanism.
        
        Args:
            config: Optional capture configuration to use for this capture
                   (overrides the instance config)
        
        Returns:
            Dict containing:
                - image: numpy array of screen content
                - platform: current platform
                - resolution: tuple of width and height
                - timestamp: current time in seconds since epoch
        
        Raises:
            ScreenCaptureError: If capture fails after retries
        """
        # Use provided config or fall back to instance config
        capture_config = config or self.config
        
        try:
            # Validate monitor number
            if capture_config.monitor_number >= len(self.sct.monitors):
                raise ScreenCaptureError(f"Invalid monitor number: {capture_config.monitor_number}. Available monitors: 0-{len(self.sct.monitors)-1}")
                
            monitor = self.sct.monitors[capture_config.monitor_number]
            screenshot: ScreenShot = self.sct.grab(monitor)
            
            # Convert to numpy array for further processing
            image = np.array(screenshot)
            
            return {
                "image": image,
                "platform": self.platform,
                "resolution": (screenshot.width, screenshot.height),
                "timestamp": time.time(),  # Use current time instead of screenshot.timestamp
            }
            
        except Exception as e:
            logger.error("screen_capture_failed",
                        error=str(e),
                        monitor=capture_config.monitor_number)
            raise ScreenCaptureError(f"Failed to capture screen: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup."""
        self.sct.close() 