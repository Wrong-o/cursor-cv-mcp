# Installing cursor-cv-mcp in Cursor IDE

This tool provides computer vision capabilities for Cursor IDE, allowing AI assistants to capture screenshots, analyze UI elements, and automate interactions with your screen.

## Installation Options

1. Clone this repository:
   ```bash
   git clone https://github.com/Wrong-o/cursor-cv-mcp
   cd cursor-cv-mcp
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

3. Register the MCP tool with Cursor:
   - Copy the `mcp_cv_tool.json` file to your Cursor MCP tools directory:
     - Linux: `~/.cursor/mcp/tools/`
     - macOS: `~/Library/Application Support/cursor/mcp/tools/`
     - Windows: `%APPDATA%\cursor\mcp\tools\`

4. Restart Cursor IDE

## Verifying Installation

1. Open Cursor IDE
2. Start a new chat with Claude
3. Type: "Use the screenshot tool to capture my screen"
4. If installed correctly, Claude should be able to capture and analyze your screen

## Troubleshooting

If the tool is not working correctly:

1. Check the Cursor Developer Console (View -> Developer -> Toggle Developer Tools)
2. Look for any error messages related to MCP or cursor-cv-mcp
3. Ensure the server is running (you should see a process running on port 8001)
4. Try manually starting the server: `python -m mcp_cv_tool.server`

## Documentation

For detailed documentation on all available functions and capabilities, see the main [README.md](README.md). 