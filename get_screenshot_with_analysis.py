#!/usr/bin/env python
"""
Enhanced screenshot tool with automatic image analysis.
Connects to the SSE endpoint, gets a screenshot, saves it, and analyzes its content.
"""

import json
import sys
from mcp_cv_tool.screenshot import get_screenshot_with_analysis, get_available_monitors

def main():
    """Main function to get and analyze a screenshot."""
    url = "http://localhost:8001/sse"
    output_file = None
    monitor = 0
    debug = False
    list_monitors = False
    
    # Parse command line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--url" and i + 1 < len(sys.argv):
            url = sys.argv[i + 1]
            i += 2
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif arg == "--monitor" and i + 1 < len(sys.argv):
            monitor = int(sys.argv[i + 1])
            i += 2
        elif arg == "--debug":
            debug = True
            i += 1
        elif arg == "--list-monitors":
            list_monitors = True
            i += 1
        else:
            # For backwards compatibility
            if i == 1:
                url = arg
            elif i == 2:
                output_file = arg
            elif i == 3 and arg.lower() in ('true', 'debug', 'yes'):
                debug = True
            i += 1
    
    # If requested, list available monitors
    if list_monitors:
        monitors = get_available_monitors(url.replace("/sse", "/health"))
        if monitors:
            print("\nAvailable monitors:")
            for idx, monitor in enumerate(monitors["monitors"]):
                print(f"  Monitor {idx}: {monitor.get('width', 'N/A')}x{monitor.get('height', 'N/A')}")
            print(f"\nPrimary monitor: {monitors.get('primary', 0)}")
            print("\nUse --monitor <number> to select a specific monitor")
        return
    
    # Get and analyze screenshot
    screenshot_path, analysis = get_screenshot_with_analysis(url, output_file, monitor, debug)
    
    if screenshot_path:
        print(f"\nScreenshot captured and analyzed: {screenshot_path}")
        
        # Save analysis to JSON file
        if analysis:
            analysis_file = screenshot_path.rsplit('.', 1)[0] + '_analysis.json'
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"Analysis saved to: {analysis_file}")
    else:
        print("Failed to capture screenshot")

if __name__ == "__main__":
    main() 