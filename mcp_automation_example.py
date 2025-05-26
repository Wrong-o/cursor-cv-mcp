#!/usr/bin/env python3
"""
Example script demonstrating how to use the MCP automation functions.
"""

import requests
import json
import time
import sys

SERVER_URL = "http://localhost:8001"

def call_mcp_function(function_name, params=None):
    """Call an MCP function with parameters."""
    if params is None:
        params = {}
        
    try:
        response = requests.post(
            f"{SERVER_URL}/mcp/call_function",
            json={
                "function_name": function_name,
                "params": params
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling MCP function: {e}")
        return None

def list_functions():
    """List all available MCP functions."""
    try:
        response = requests.get(f"{SERVER_URL}/mcp/list_functions")
        return response.json()
    except Exception as e:
        print(f"Error listing functions: {e}")
        return None

def main():
    """Run the automation example."""
    # List available functions
    print("Listing available MCP functions...")
    functions_result = list_functions()
    if functions_result:
        print("Available functions:")
        for function in functions_result.get("functions", []):
            print(f"- {function['name']}")
    
    print("\n" + "-" * 50 + "\n")
    
    # Get current mouse position
    print("Getting current mouse position...")
    mouse_pos = call_mcp_function("mcp_get_mouse_position")
    if mouse_pos and mouse_pos.get("success"):
        x, y = mouse_pos["position"]
        print(f"Current mouse position: ({x}, {y})")
    
    # Take a screenshot
    print("\nCapturing screenshot...")
    screenshot = call_mcp_function("mcp_screenshot_capture", {"monitor": 1})
    if screenshot and screenshot.get("success"):
        print(f"Screenshot saved to {screenshot['screenshot_path']}")
    
    # Click example (uncomment to test)
    # WARNING: This will actually move and click your mouse
    """
    print("\nClicking at position (500, 500)...")
    click_result = call_mcp_function("mcp_mouse_click", {"x": 500, "y": 500})
    if click_result and click_result.get("success"):
        print(f"Clicked at position {click_result['position']}")
    """
    
    # Type text example (uncomment to test)
    # WARNING: This will actually type text wherever your cursor is
    """
    print("\nTyping 'Hello from MCP!'...")
    type_result = call_mcp_function("mcp_type_text", {"text": "Hello from MCP!"})
    if type_result and type_result.get("success"):
        print(f"Typed text: {type_result['text']}")
    """
    
    # Press key example (uncomment to test)
    # WARNING: This will actually press the key
    """
    print("\nPressing 'enter' key...")
    key_result = call_mcp_function("mcp_press_key", {"key": "enter"})
    if key_result and key_result.get("success"):
        print(f"Pressed key: {key_result['key']}")
    """
    
    # Highlight area example (uncomment to test)
    # WARNING: This will move your mouse to highlight an area
    """
    print("\nHighlighting area (100, 100, 200, 200)...")
    highlight_result = call_mcp_function("mcp_highlight_area", {
        "x": 100, "y": 100, "width": 200, "height": 200
    })
    if highlight_result and highlight_result.get("success"):
        print(f"Highlighted area: {highlight_result['area']}")
    """
    
    print("\n" + "-" * 50)
    print("Example completed! To perform actual automation:")
    print("1. Uncomment the examples in this script")
    print("2. Or call the MCP functions directly in your own script")
    print("3. Visit http://localhost:8001/docs for API documentation")

if __name__ == "__main__":
    main() 