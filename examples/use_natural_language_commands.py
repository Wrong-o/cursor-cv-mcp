#!/usr/bin/env python
"""
Example demonstrating how to use natural language commands for screenshots.
This shows how you can integrate the screenshot functionality with Cursor.
"""

import os
import sys
import json
import time
from datetime import datetime
from mcp_cv_tool.cursor_integration import execute_cursor_command

# Simulation mode for demonstration
SIMULATION_MODE = False

def simulate_screenshot(command):
    """Simulate a screenshot capture for demonstration purposes."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = f"simulated_screenshot_{timestamp}.jpg"
    
    # Create a simulated response
    result = {
        "success": True,
        "screenshot_path": screenshot_path,
        "resolution": (1920, 1080),
        "analysis": {
            "basic_info": {
                "format": "JPEG",
                "size": (1920, 1080),
                "mode": "RGB",
                "file_size": 12345
            },
            "text_content": "This is simulated text from the screenshot.\nIt would contain OCR-extracted text from your screen.\nFor example: <div class=\"container\">\n  <div class=\"header\">This is misaligned</div>\n</div>"
        },
        "cursor_context": {
            "screenshot_path": screenshot_path,
            "command_text": command,
            "has_text": True
        }
    }
    
    print(f"SIMULATION: Captured screenshot (simulated) as {screenshot_path}")
    print(f"SIMULATION: Resolution would be {result['resolution']}")
    
    return result

def demonstrate_natural_language_commands():
    """Demonstrate natural language commands for screenshots."""
    
    print("=== Natural Language Screenshot Commands Demo ===\n")
    
    # Get the command from command line arguments or use a default
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    else:
        command = "use screenshot"
    
    print(f"Executing command: '{command}'")
    print("-" * 50)
    
    # Execute the command (or simulate it)
    if SIMULATION_MODE:
        result = simulate_screenshot(command)
    else:
        result = execute_cursor_command(command)
    
    # Print the result
    if result["success"]:
        print("Command executed successfully!")
        print(f"Screenshot saved to: {result['screenshot_path']}")
        
        if "resolution" in result:
            print(f"Resolution: {result['resolution']}")
        
        # Display part of the extracted text if available
        if "analysis" in result and "text_content" in result["analysis"]:
            text = result["analysis"]["text_content"]
            text_sample = text[:200] + "..." if len(text) > 200 else text
            print(f"\nExtracted text sample:\n{text_sample}")
            
        # Show how this could be used in Cursor
        print("\nIn Cursor, this could be used like:")
        print(f"User: 'Fix the div position. {command}'")
        print("Cursor: 'I've taken a screenshot and analyzed the layout. I can see the div positioning issue...'")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n=== Demo Complete ===")

def test_different_commands():
    """Test different natural language commands."""
    
    commands = [
        "use screenshot",
        "take a screenshot",
        "capture screenshot from monitor 0",
        "screenshot of monitor 1",
        "get screenshot without analysis",
        "analyze the screen",
        "screenshot with analysis",
        "Fix the div position. use screenshot"
    ]
    
    print("=== Testing Different Command Phrases ===\n")
    
    for cmd in commands:
        print(f"Testing: '{cmd}'")
        if SIMULATION_MODE:
            result = simulate_screenshot(cmd)
            print(f"  Success: {result['success']}")
            print(f"  Action: Simulated screenshot capture")
        else:
            result = execute_cursor_command(cmd)
            print(f"  Success: {result['success']}")
            if result["success"]:
                print(f"  Action: Captured screenshot from monitor {result.get('params', {}).get('monitor', '?')}")
            else:
                print(f"  Error: {result.get('error', 'Unknown error')}")
        print()
    
    print("=== Testing Complete ===")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-commands":
        test_different_commands()
    else:
        demonstrate_natural_language_commands() 