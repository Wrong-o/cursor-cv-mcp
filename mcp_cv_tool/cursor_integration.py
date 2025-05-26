"""
Cursor integration for the screenshot functionality.
This allows using the screenshot tool with natural language commands in Cursor.
"""

import json
import os
import re
from typing import Dict, Any, Optional, List

from .context7_integration import call_function

# Command patterns to match in natural language
SCREENSHOT_PATTERNS = [
    r'(?:take|capture|get)\s+(?:a\s+)?screenshot',
    r'use\s+screenshot',
    r'screenshot\s+(?:of|from)',
    r'screenshot\s+monitor\s+(\d+)',
]

def parse_screenshot_command(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a natural language command for screenshots.
    
    Args:
        text: The natural language command text
        
    Returns:
        Dictionary with parsed command parameters or None if not a screenshot command
    """
    text = text.lower()
    
    # Check if any screenshot pattern matches
    for pattern in SCREENSHOT_PATTERNS:
        match = re.search(pattern, text)
        if match:
            # Initialize with default parameters
            params = {
                "monitor": 0,
                "analyze": True
            }
            
            # Check for monitor number specification
            monitor_match = re.search(r'monitor\s+(\d+)', text)
            if monitor_match:
                params["monitor"] = int(monitor_match.group(1))
            
            # Check for analysis options
            if 'with analysis' in text or 'analyze' in text:
                params["analyze"] = True
            elif 'without analysis' in text or 'no analysis' in text:
                params["analyze"] = False
                
            return {
                "command": "screenshot",
                "params": params
            }
    
    return None

def execute_cursor_command(command_text: str) -> Dict[str, Any]:
    """
    Execute a Cursor command from natural language.
    
    Args:
        command_text: The natural language command
        
    Returns:
        Result of the command execution
    """
    # Parse the command
    parsed_command = parse_screenshot_command(command_text)
    
    if not parsed_command:
        return {
            "success": False,
            "error": "Command not recognized. Try 'use screenshot' or 'take screenshot from monitor 1'.",
            "command_text": command_text
        }
    
    # Execute the screenshot command
    if parsed_command["command"] == "screenshot":
        result = call_function("mcp_screenshot_capture", parsed_command["params"])
        
        # Add additional context for Cursor
        if result["success"]:
            result["cursor_context"] = {
                "screenshot_path": result["screenshot_path"],
                "command_text": command_text,
                "has_text": "text_content" in result.get("analysis", {}) and result["analysis"]["text_content"]
            }
            
        return result
    
    # Default error response
    return {
        "success": False,
        "error": "Unsupported command",
        "command_text": command_text
    }

# Register command handlers
CURSOR_COMMANDS = {
    "screenshot": execute_cursor_command
} 