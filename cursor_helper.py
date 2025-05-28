#!/usr/bin/env python
"""
Direct helper script for Cursor to take screenshots.
This file can be called directly from Cursor with a ! command.
"""

import os
import sys
import json
from datetime import datetime
from mcp_cv_tool.screenshot import get_screenshot_with_analysis, SCREENSHOTS_DIR

def take_screenshot_for_cursor():
    """Take a screenshot and format output for Cursor."""
    # Parse monitor number if provided
    monitor = 1  # Default to monitor 1 (first physical monitor)
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        monitor = int(sys.argv[1])
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cursor_screenshot_{timestamp}.jpg"
    
    # Create screenshots directory if needed
    screenshots_dir = os.path.join(os.getcwd(), "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    
    print(f"Taking screenshot from monitor {monitor}...")
    
    # Take the screenshot
    screenshot_path, analysis = get_screenshot_with_analysis(
        url="http://localhost:8001/sse",
        output_file=output_file,
        monitor=monitor,
        screenshots_dir=screenshots_dir
    )
    
    if not screenshot_path:
        print("Failed to capture screenshot. Make sure the screenshot server is running:")
        print("cd /home/jayb/projects/ml_practise/cursor-cv-mcp")
        print("source venv/bin/activate")
        print("uvicorn mcp_cv_tool.server:app --port 8001")
        return
    
    # Print success message
    print(f"\n✅ Screenshot captured: {screenshot_path}")
    print(f"Resolution: {analysis['basic_info']['size'][0]}x{analysis['basic_info']['size'][1]}")
    
    # Print extracted text if available
    if "text_content" in analysis and analysis["text_content"]:
        text = analysis["text_content"]
        print("\n📝 Extracted text from screen:")
        print("-" * 50)
        print(text[:500] + "..." if len(text) > 500 else text)
        print("-" * 50)
    
    # Print a helpful message for the user
    print("\n📋 To use this in Cursor:")
    print("1. I've captured your screen and can see what you're looking at")
    print("2. Now I can help fix the div positioning issue")
    print("3. Let me know which specific element needs adjustment")

if __name__ == "__main__":
    take_screenshot_for_cursor() 