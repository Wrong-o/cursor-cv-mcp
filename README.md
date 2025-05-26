# Cursor CV MCP

Computer Vision and Automation API with MCP (Modular Command Protocol) support for cursor-based interactions.

## Overview

This project provides a FastAPI server with capabilities for:
- Screenshot capture and analysis with OCR
- Mouse automation (clicking, moving)
- Keyboard automation (typing, key presses)
- Multi-monitor support
- MCP standard API endpoints

The server runs locally and provides a standardized REST API following the MCP standard, making it easy to integrate with other tools and AI agents.

## Features

### Screenshot Capabilities
- Capture screenshots from any connected monitor
- Analyze images with OCR to extract text content
- Get detailed monitor information
- Save screenshots to disk

### Automation Capabilities
- Mouse control (position, clicks)
- Keyboard input (typing text, pressing keys)
- Visual area highlighting
- Both PyAutoGUI and xdotool support (fallback mechanism)

### MCP Standard Integration
- Standardized `/mcp/list_functions` endpoint to discover available functions
- Standardized `/mcp/call_function` endpoint to invoke any function with parameters
- Consistent response format following MCP standards
- Interactive API documentation via Swagger UI

## Installation

### Prerequisites
- Python 3.8+
- X11 display server (for Linux)
- Web browser

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cursor-cv-mcp.git
   cd cursor-cv-mcp
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install system dependencies (Linux only):
   ```bash
   sudo apt-get install xdotool tesseract-ocr
   ```

## Usage

### Starting the Server

Run the server script:
```bash
./start-screenshot-server
```

This will start the FastAPI server on port 8001 and make it accessible at http://localhost:8001.

### API Documentation

Once the server is running, you can access the interactive API documentation at:
- http://localhost:8001/docs

### Example Usage

#### Python Client

```python
import requests

# List available functions
functions = requests.get("http://localhost:8001/mcp/list_functions").json()
print(functions)

# Capture a screenshot
screenshot = requests.post(
    "http://localhost:8001/mcp/call_function",
    json={
        "function_name": "mcp_screenshot_capture",
        "params": {"monitor": 1}
    }
).json()
print(f"Screenshot saved to: {screenshot['screenshot_path']}")

# Move mouse and click
click_result = requests.post(
    "http://localhost:8001/mcp/call_function",
    json={
        "function_name": "mcp_mouse_click",
        "params": {"x": 500, "y": 500}
    }
).json()

# Type text
type_result = requests.post(
    "http://localhost:8001/mcp/call_function",
    json={
        "function_name": "mcp_type_text",
        "params": {"text": "Hello world!"}
    }
).json()
```

#### Curl Examples

```bash
# List available functions
curl -X GET http://localhost:8001/mcp/list_functions

# Capture a screenshot
curl -X POST http://localhost:8001/mcp/call_function \
  -H "Content-Type: application/json" \
  -d '{"function_name": "mcp_screenshot_capture", "params": {"monitor": 1}}'

# Click at a position
curl -X POST http://localhost:8001/mcp/call_function \
  -H "Content-Type: application/json" \
  -d '{"function_name": "mcp_mouse_click", "params": {"x": 500, "y": 500}}'

# Type text
curl -X POST http://localhost:8001/mcp/call_function \
  -H "Content-Type: application/json" \
  -d '{"function_name": "mcp_type_text", "params": {"text": "Hello world!"}}'
```

## Available MCP Functions

| Function Name | Description | Parameters |
|---------------|-------------|------------|
| `mcp_screenshot_capture` | Captures a screenshot | `monitor`, `output_file`, `debug`, `analyze` |
| `mcp_screenshot_analyze_image` | Analyzes an existing image | `image_path` |
| `mcp_screenshot_list_monitors` | Lists available monitors | - |
| `mcp_mouse_click` | Clicks at a position | `x`, `y`, `button`, `clicks` |
| `mcp_type_text` | Types text | `text`, `interval` |
| `mcp_press_key` | Presses a specific key | `key` |
| `mcp_get_mouse_position` | Gets current mouse position | - |
| `mcp_highlight_area` | Highlights an area | `x`, `y`, `width`, `height`, `duration` |

## Troubleshooting

### X11 Display Issues

If you encounter X11 display issues, try:

```bash
xhost +local:
export DISPLAY=:0
```

Then restart the server.

### Mouse Movement Not Working

If the mouse isn't moving:
1. Verify the server has X11 display access
2. Check that PyAutoGUI is installed correctly
3. The fallback xdotool method should automatically activate

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 