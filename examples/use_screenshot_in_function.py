#!/usr/bin/env python
"""
Example of using the screenshot functionality from other functions.
This demonstrates how to import and use the screenshot tool in your own code.
"""

import os
from mcp_cv_tool.screenshot import get_screenshot_with_analysis, analyze_image, get_available_monitors

def take_screenshot_and_analyze(monitor_number=0, save_path=None):
    """
    Take a screenshot from a specific monitor and analyze it.
    
    Args:
        monitor_number: The monitor number to capture (default: 0)
        save_path: Path to save the screenshot (default: auto-generated)
        
    Returns:
        Tuple of (Screenshot path, Analysis results)
    """
    print(f"Taking screenshot from monitor {monitor_number}...")
    screenshot_path, analysis = get_screenshot_with_analysis(
        monitor=monitor_number,
        output_file=save_path
    )
    
    if screenshot_path and analysis:
        print(f"Screenshot taken and saved to: {screenshot_path}")
        
        # Extract text content from the analysis
        text_content = analysis.get("text_content", "No text found")
        print(f"\nExtracted text: {text_content[:100]}...")
        
        return screenshot_path, analysis
    else:
        print("Failed to take screenshot")
        return None, None

def analyze_existing_image(image_path):
    """
    Analyze an existing image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Analysis results dictionary
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return None
    
    print(f"Analyzing image: {image_path}")
    analysis = analyze_image(image_path)
    
    if analysis:
        print("Analysis complete!")
        text_content = analysis.get("text_content", "No text found")
        print(f"\nExtracted text: {text_content[:100]}...")
        
        return analysis
    else:
        print("Failed to analyze image")
        return None

def list_all_monitors():
    """
    List all available monitors.
    
    Returns:
        List of monitor information
    """
    monitors = get_available_monitors()
    
    if monitors:
        print("\nAvailable monitors:")
        for idx, monitor in enumerate(monitors["monitors"]):
            print(f"  Monitor {idx}: {monitor.get('width', 'N/A')}x{monitor.get('height', 'N/A')}")
        print(f"\nPrimary monitor: {monitors.get('primary', 0)}")
        
        return monitors["monitors"]
    else:
        print("Failed to get monitor information")
        return []

def demo():
    """Run a demonstration of the screenshot functionality."""
    # First, list all available monitors
    monitors = list_all_monitors()
    
    if not monitors:
        print("No monitors detected. Exiting.")
        return
    
    # Take a screenshot from the primary monitor
    primary_screenshot, primary_analysis = take_screenshot_and_analyze(0)
    
    # If there are multiple monitors, take a screenshot from the second one too
    if len(monitors) > 1:
        secondary_screenshot, secondary_analysis = take_screenshot_and_analyze(1)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    demo() 