#!/usr/bin/env python
"""
Cursor plugin for screenshot and OCR functionality.
This file registers the plugin with Cursor's plugin system.
"""

import os
import json
import sys
import re
from typing import Dict, Any, List, Optional

# Make sure we can import from our package
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from mcp_cv_tool.cursor_integration import execute_cursor_command

def cursor_plugin_init() -> Dict[str, Any]:
    """
    Initialize the Cursor plugin.
    
    Returns:
        Plugin configuration dictionary
    """
    return {
        "name": "Screenshot and OCR",
        "version": "1.0.0",
        "description": "Capture screenshots and extract text with OCR",
        "commands": [
            {
                "name": "screenshot",
                "description": "Capture a screenshot and analyze it",
                "patterns": [
                    r"(?:use|take|capture|get)\s+(?:a\s+)?screenshot",
                    r"screenshot\s+(?:of|from)\s+monitor\s+(\d+)",
                    r"screenshot\s+monitor\s+(\d+)",
                    r"screenshot\s+(?:of|from)\s+screen\s+(\d+)",
                    r"screenshot\s+screen\s+(\d+)",
                    r"screenshot\s+(?:with|without)\s+analysis"
                ]
            }
        ],
        "triggers": [
            {
                "pattern": r"fix.*(?:position|alignment|layout).*(?:use|take|get)?\s*screenshot",
                "command": "screenshot",
                "description": "Capture screenshot when fixing layout issues"
            }
        ]
    }

def parse_monitor_number(command_text: str) -> int:
    """Extract monitor number from command text."""
    monitor_patterns = [
        r"monitor\s+(\d+)",
        r"screen\s+(\d+)"
    ]
    
    for pattern in monitor_patterns:
        match = re.search(pattern, command_text.lower())
        if match:
            return int(match.group(1))
    return 0

def cursor_plugin_execute(command: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a command from Cursor.
    
    Args:
        command: The command name
        args: Command arguments
        
    Returns:
        Command execution result
    """
    if command == "screenshot":
        # Get the full command text if available
        command_text = args.get("text", "use screenshot")
        
        # Extract monitor number if specified
        monitor = parse_monitor_number(command_text)
        
        # Execute the command
        result = execute_cursor_command(command_text)
        
        if result["success"]:
            # Add helpful context for Cursor
            result["cursor_context"] = {
                "screenshot_path": result["screenshot_path"],
                "monitor": monitor,
                "command": command_text,
                "has_text": "text_content" in result.get("analysis", {})
            }
            
            # Add extracted text if available
            if "analysis" in result and "text_content" in result["analysis"]:
                result["cursor_context"]["extracted_text"] = result["analysis"]["text_content"]
        
        return result
    
    return {
        "success": False,
        "error": f"Unknown command: {command}"
    }

def cursor_plugin_help() -> str:
    """
    Return help information for the plugin.
    
    Returns:
        Help text as markdown
    """
    return """
# Screenshot and OCR Plugin

This plugin allows you to capture screenshots and extract text using OCR.

## Commands

- `use screenshot` - Capture a screenshot from the primary monitor
- `take a screenshot` - Capture a screenshot from the primary monitor
- `capture screenshot from monitor X` - Capture from a specific monitor
- `screenshot of monitor X` - Capture from a specific monitor
- `screenshot with analysis` - Capture and perform detailed analysis

## Usage in Cursor

You can use these commands directly in your Cursor prompts:

```
Fix the div position. use screenshot
```

The plugin will capture a screenshot, analyze it, and make the information
available to Cursor for further processing.

## Examples

1. Fix layout issues:
   ```
   Fix the div position. use screenshot
   ```

2. Capture from specific monitor:
   ```
   screenshot of monitor 1
   ```

3. Analyze text on screen:
   ```
   What's on my screen? take a screenshot
   ```
    """

# Register the plugin functions
__all__ = ["cursor_plugin_init", "cursor_plugin_execute", "cursor_plugin_help"] 