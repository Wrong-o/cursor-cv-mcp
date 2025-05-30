# Cursor Screenshot Tool MCP

A Managed Code Plugin (MCP) for Cursor IDE that provides screenshot capture and computer vision capabilities for AI assistants.

## Features

- Capture screenshots of your monitor
- Analyze UI elements with computer vision
- Automate mouse clicks and keyboard input
- OCR text extraction from images
- Real-time screen capture via server-sent events

## Installation

### Option 1: Unified Installer (Recommended)

Run the unified installer script which handles virtual environment creation, package installation, and MCP registration:

```bash
python3 install.py
```

Options:
- `--venv PATH` - Specify the virtual environment path (default: `cv_venv`)
- `--force` - Force recreation of the virtual environment
- `--no-venv` - Don't create a virtual environment (use system Python)

### Option 2: Manual Installation

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Copy the MCP configuration to Cursor:
   ```bash
   # Linux
   cp mcp_cv_tool.json ~/.cursor/mcp/tools/screenshot-tool.json
   
   # macOS
   cp mcp_cv_tool.json ~/Library/Application\ Support/cursor/mcp/tools/screenshot-tool.json
   
   # Windows
   copy mcp_cv_tool.json %APPDATA%\cursor\mcp\tools\screenshot-tool.json
   ```

3. Restart Cursor IDE

## Usage in Cursor

After installation, you can use the screenshot tool via Claude in Cursor. Try these prompts:

- "Use the screenshot tool to capture my screen"
- "Capture a screenshot of my monitor and analyze the UI elements"
- "Capture my screen and find all buttons"
- "Use the screenshot tool to click on the button that says 'Submit'"

## Available Functions

### mcp_screenshot_capture
Captures a screenshot of your screen from the specified monitor.

Parameters:
- `monitor` (integer): Monitor number to capture (1-based index)
- `output_file` (string, optional): Path to save the screenshot
- `debug` (boolean): Enable debug mode
- `analyze` (boolean): Analyze screenshot with OCR

### mcp_find_element
Finds UI elements like buttons, inputs, or text areas in a screenshot.

Parameters:
- `screenshot_path` (string): Path to screenshot image
- `element_type` (string): Type of element to find (button, input, checkbox, etc.)
- `text` (string, optional): Text content to search for
- `confidence` (number): Confidence threshold (0.0-1.0)

### mcp_mouse_click
Clicks at a specific screen position.

Parameters:
- `x` (integer): X coordinate
- `y` (integer): Y coordinate
- `button` (string): Mouse button (left, right, middle)

### mcp_keyboard_input
Types text at the current cursor position.

Parameters:
- `text` (string): Text to type. Special keys: \n (Enter), \t (Tab), \b (Backspace), \e (Escape), \s (Space), \u (Up), \d (Down), \l (Left), \r (Right)
- `interval` (number): Time between keypresses in seconds

## Troubleshooting

If the tool is not being recognized by Cursor:

1. Check if the configuration file is in the right location:
   - Linux: `~/.cursor/mcp/tools/screenshot-tool.json`
   - macOS: `~/Library/Application Support/cursor/mcp/tools/screenshot-tool.json`
   - Windows: `%APPDATA%\cursor\mcp\tools\screenshot-tool.json`

2. Ensure the package is installed:
   ```bash
   pip show cursor-cv-mcp
   ```

3. Check Cursor's developer tools console for any errors (View → Developer → Toggle Developer Tools)

4. Try running the server manually:
   ```bash
   python -m mcp_cv_tool.server
   ```

5. Restart Cursor IDE

## Requirements

- Python 3.9+
- Cursor IDE
- Additional Python libraries (installed automatically):
  - fastapi
  - uvicorn
  - mss (screen capture)
  - opencv-python
  - pyautogui (optional, for mouse/keyboard control)
  - pytesseract (optional, for OCR) 