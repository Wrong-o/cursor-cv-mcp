"""
Screenshot module with automatic image analysis.
Provides functions to capture and analyze screenshots from specific monitors.
"""

import ast
import base64
import io
import json
import os
import requests
import sys
import time
import platform
from datetime import datetime
from PIL import Image
from typing import Dict, Tuple, Optional, Any, Union

# Check if we have the cv2 module for OCR
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Check if we have pytesseract for OCR
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# Import mss for screenshot capture
import mss
import mss.tools

try:
    from .config import get_screenshots_dir
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

# Define a default screenshots directory (can be overridden)
SCREENSHOTS_DIR = get_screenshots_dir() if HAS_CONFIG else os.path.expanduser("~/screenshots")

def get_available_monitors(url: str = "http://localhost:8001/health") -> Optional[Dict[str, Any]]:
    """
    Get information about available monitors directly using mss.
    
    Args:
        url: DEPRECATED - kept for backward compatibility
        
    Returns:
        Dict containing monitor information or None if unavailable
    """
    try:
        with mss.mss() as sct:
            monitors_info = {
                "monitors": [],
                "primary": 0
            }
            
            # Get primary monitor (monitor 0 is a special case - it's the entire virtual screen)
            primary_idx = 0  # Default to first physical monitor
            
            # Collect monitor information
            for idx, monitor in enumerate(sct.monitors[1:], 1):  # Skip monitor 0 (entire virtual screen)
                monitor_info = {
                    "id": idx,
                    "width": monitor["width"],
                    "height": monitor["height"],
                    "left": monitor["left"],
                    "top": monitor["top"],
                }
                monitors_info["monitors"].append(monitor_info)
                
                # Check if this is the primary monitor (typically at position 0,0)
                if monitor["left"] == 0 and monitor["top"] == 0:
                    primary_idx = idx
            
            monitors_info["primary"] = primary_idx
            return monitors_info
    except Exception as e:
        print(f"Error getting monitor info: {e}")
        return None

def get_screenshot_with_analysis(
    url: str = "http://localhost:8001/sse",  # Kept for backward compatibility
    output_file: Optional[str] = None,
    monitor: int = 0,
    debug: bool = False,
    screenshots_dir: Optional[str] = None
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Capture a screenshot directly using mss, save it, and analyze its content.
    
    Args:
        url: DEPRECATED - kept for backward compatibility
        output_file: The output file path, defaults to timestamp-based filename
        monitor: Monitor number to capture (default: 0)
        debug: Whether to print debug information
        screenshots_dir: Directory to save screenshots (default: SCREENSHOTS_DIR)
    
    Returns:
        Tuple of (Path to the saved screenshot file, Analysis results)
    """
    # Use the provided screenshots directory or fall back to the default
    target_dir = screenshots_dir if screenshots_dir is not None else SCREENSHOTS_DIR
    
    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(target_dir, f"screenshot_{timestamp}.jpg")
    elif not os.path.isabs(output_file):
        # If output_file is not an absolute path, join it with target_dir
        output_file = os.path.join(target_dir, output_file)
    
    try:
        # Get monitor information
        with mss.mss() as sct:
            # Monitor 0 is the entire virtual screen
            # Physical monitors start at index 1
            if monitor == 0:
                # Use the primary monitor instead of the entire virtual screen
                monitors_info = get_available_monitors()
                if monitors_info and "primary" in monitors_info:
                    monitor = monitors_info["primary"]
                else:
                    monitor = 1  # Default to first physical monitor
            
            # Ensure the monitor exists
            if monitor >= len(sct.monitors):
                print(f"Error: Monitor {monitor} not found. Using monitor 1.")
                monitor = 1
            
            # Capture the screen
            monitor_dict = sct.monitors[monitor]
            if debug:
                print(f"Capturing monitor {monitor}: {monitor_dict}")
            
            screenshot = sct.grab(monitor_dict)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            
            # Save the image
            img.save(output_file)
            print(f"Screenshot saved to {output_file}")
            
            # Get platform and resolution info
            platform_name = platform.system()
            resolution = screenshot.size
            print(f"Resolution: {resolution}")
            print(f"Platform: {platform_name}")
            
            # Convert to bytes for analysis
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_data = img_byte_arr.getvalue()
            
            # Analyze the image
            print("\nAnalyzing screenshot...")
            analysis_results = analyze_image(output_file, image_data)
            
            return output_file, analysis_results
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
        return None, None

def analyze_image(image_path: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Analyze the image content.
    
    Args:
        image_path: Path to the image file
        image_data: Raw image data (optional)
    
    Returns:
        Dict containing analysis results
    """
    results = {
        "basic_info": {},
        "text_content": None,
        "analysis": None
    }
    
    # Get basic image info
    try:
        if image_data:
            img = Image.open(io.BytesIO(image_data))
        else:
            img = Image.open(image_path)
        
        results["basic_info"] = {
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
            "file_size": os.path.getsize(image_path) if os.path.exists(image_path) else None
        }
        
        print(f"Image info: {img.format} image, {img.size[0]}x{img.size[1]}, {img.mode} mode")
        
        # Try to extract text using OCR if available
        if HAS_TESSERACT and HAS_CV2:
            print("Performing OCR to extract text...")
            # Convert PIL Image to OpenCV format
            if image_data:
                nparr = np.frombuffer(image_data, np.uint8)
                cv_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                cv_img = cv2.imread(image_path)
            
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            
            # Perform OCR
            text = pytesseract.image_to_string(gray)
            results["text_content"] = text
            
            # Print a sample of the extracted text
            text_sample = text[:200] + "..." if len(text) > 200 else text
            print(f"\nExtracted text sample:\n{text_sample}")
        else:
            if not HAS_TESSERACT:
                print("OCR not available: pytesseract not installed")
            if not HAS_CV2:
                print("OCR not available: OpenCV not installed")
            print("Install with: pip install pytesseract opencv-python")
            
            results["analysis"] = "OCR not available. Install pytesseract and opencv-python for text extraction."
    except Exception as e:
        print(f"Error analyzing image: {e}")
        results["analysis"] = f"Error: {str(e)}"
    
    return results 