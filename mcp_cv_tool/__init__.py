"""
Cursor CV MCP Tool - A toolkit for screen capture and analysis.
"""

__version__ = "0.1.0"

# Export the screenshot functionality
from .screenshot import get_screenshot_with_analysis, analyze_image, get_available_monitors

# Export the Context7-like interface
from .context7_integration import call_function, FUNCTION_REGISTRY

# Export the Cursor integration
from .cursor_integration import execute_cursor_command, parse_screenshot_command 