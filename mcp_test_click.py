#!/usr/bin/env python3
"""
Test script to click at a specific position using the MCP API.
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
        print(f"Calling {function_name} with params: {params}")
        response = requests.post(
            f"{SERVER_URL}/mcp/call_function",
            json={
                "function_name": function_name,
                "params": params
            },
            timeout=30  # Longer timeout for debugging
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error calling MCP function: {e}")
        return None

def main():
    """Run the test script."""
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(2)
    
    # Get current mouse position
    print("Getting current mouse position...")
    mouse_pos = call_mcp_function("mcp_get_mouse_position")
    if mouse_pos and mouse_pos.get("success"):
        try:
            x, y = mouse_pos["position"]
            print(f"Current mouse position: ({x}, {y})")
        except (KeyError, TypeError):
            print("Could not extract position from response")
    
    # Move and click at position (500, 500)
    print("\nClicking at position (500, 500)...")
    click_result = call_mcp_function("mcp_mouse_click", {"x": 500, "y": 500})
    
    # Type some text
    print("\nTyping text...")
    type_result = call_mcp_function("mcp_type_text", {"text": "Hello from MCP!"})
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 