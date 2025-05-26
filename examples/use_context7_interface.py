#!/usr/bin/env python
"""
Example demonstrating how to use the Context7-like interface for screenshots.
This shows how to use the screenshot functionality in a similar way to Context7 calls.
"""

import json
from mcp_cv_tool.context7_integration import call_function

def demonstrate_context7_interface():
    """Demonstrate the Context7-like interface for screenshots."""
    
    print("=== Context7-like Screenshot Interface Demo ===\n")
    
    # Example 1: List available monitors
    print("1. Listing available monitors...")
    result = call_function("mcp_screenshot_list_monitors", {})
    
    if result["success"]:
        print(f"Found {len(result['monitors'])} monitors")
        print(f"Primary monitor index: {result['primary']}")
        for idx, monitor in enumerate(result["monitors"]):
            print(f"  Monitor {idx}: {monitor.get('width', 'N/A')}x{monitor.get('height', 'N/A')}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Example 2: Capture a screenshot
    print("2. Capturing a screenshot from the primary monitor...")
    capture_params = {
        "monitor": 0,
        "output_file": "context7_screenshot.jpg",
        "analyze": True
    }
    
    result = call_function("mcp_screenshot_capture", capture_params)
    
    if result["success"]:
        print(f"Screenshot saved to: {result['screenshot_path']}")
        print(f"Resolution: {result['resolution']}")
        
        # Display part of the extracted text if available
        if "analysis" in result and "text_content" in result["analysis"]:
            text = result["analysis"]["text_content"]
            text_sample = text[:200] + "..." if len(text) > 200 else text
            print(f"\nExtracted text sample:\n{text_sample}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Example 3: Analyze an existing image
    print("3. Analyzing an existing image...")
    analysis_params = {
        "image_path": "context7_screenshot.jpg"
    }
    
    result = call_function("mcp_screenshot_analyze_image", analysis_params)
    
    if result["success"]:
        print("Image analysis complete!")
        basic_info = result["analysis"]["basic_info"]
        print(f"Image info: {basic_info.get('format')} image, "
              f"{basic_info.get('size', (0, 0))[0]}x{basic_info.get('size', (0, 0))[1]}, "
              f"{basic_info.get('mode')} mode")
        
        # Display part of the extracted text if available
        if "text_content" in result["analysis"]:
            text = result["analysis"]["text_content"]
            text_sample = text[:200] + "..." if len(text) > 200 else text
            print(f"\nExtracted text sample:\n{text_sample}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Example 4: Invalid function call (for error handling demonstration)
    print("4. Attempting to call an invalid function (for demonstration)...")
    result = call_function("non_existent_function", {})
    
    print(f"Success: {result['success']}")
    print(f"Error: {result.get('error', 'No error')}")
    print(f"Available functions: {', '.join(result.get('available_functions', []))}")
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    demonstrate_context7_interface() 