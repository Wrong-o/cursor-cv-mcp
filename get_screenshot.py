#!/usr/bin/env python
"""
Simple script to get a screenshot from the CV MCP server.
Connects to the SSE endpoint, gets the first screenshot, and saves it as a JPEG file.
"""

import ast
import json
import requests
import sys
import time
from datetime import datetime

def get_screenshot(url="http://localhost:8080/sse", output_file=None, debug=False):
    """
    Connect to the SSE endpoint and get a screenshot.
    
    Args:
        url: The URL of the SSE endpoint
        output_file: The output file path, defaults to timestamp-based filename
        debug: Whether to print debug information
    
    Returns:
        Path to the saved screenshot file
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"screenshot_{timestamp}.jpg"
    
    print(f"Connecting to {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return None
    
    print("Connected. Waiting for screenshot data...")
    for line in response.iter_lines():
        if line:
            # SSE format has "data: " prefix
            line_str = line.decode('utf-8')
            if debug:
                print(f"Received line: {line_str[:100]}...")  # Print first 100 chars
            
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: " prefix
                
                try:
                    # Try to parse as Python literal (dict) instead of JSON
                    data = ast.literal_eval(data_str)
                    
                    if "error" in data and data["error"] is not None:
                        print(f"Error from server: {data['error']}")
                        continue  # Try to get the next frame
                    
                    if "image" in data:
                        # Convert hex to binary
                        image_data = bytes.fromhex(data["image"])
                        
                        # Save the image
                        with open(output_file, "wb") as f:
                            f.write(image_data)
                        
                        print(f"Screenshot saved to {output_file}")
                        print(f"Resolution: {data['resolution']}")
                        print(f"Platform: {data['platform']}")
                        print(f"Timestamp: {data['timestamp']}")
                        return output_file
                    else:
                        print("No image data found in response")
                        if debug:
                            print(f"Data keys: {data.keys()}")
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing data: {e}")
                    print(f"Data (first 100 chars): {data_str[:100]}...")
                    continue
            elif line_str.startswith(":"):
                # This is an SSE comment/ping
                if debug:
                    print(f"SSE ping/comment: {line_str}")
            else:
                print(f"Received non-data line: {line_str}")
    
    print("No valid screenshot data received")
    return None

if __name__ == "__main__":
    url = "http://localhost:8080/sse"
    output_file = None
    debug = False
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3].lower() in ('true', 'debug', 'yes'):
        debug = True
    
    get_screenshot(url, output_file, debug) 