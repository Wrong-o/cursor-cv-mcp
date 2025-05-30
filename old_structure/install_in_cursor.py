#!/usr/bin/env python3
"""
Install cursor-cv-mcp in Cursor IDE
This script helps set up the Computer Vision MCP tool in Cursor
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path

def get_cursor_mcp_dir():
    """Get the Cursor MCP tools directory for the current platform."""
    system = platform.system()
    home = Path.home()
    
    if system == "Linux":
        return home / ".cursor" / "mcp" / "tools"
    elif system == "Darwin":  # macOS
        return home / "Library" / "Application Support" / "cursor" / "mcp" / "tools"
    elif system == "Windows":
        return Path(os.environ.get("APPDATA", "")) / "cursor" / "mcp" / "tools"
    else:
        print(f"Unsupported platform: {system}")
        sys.exit(1)

def install_package():
    """Install the package using pip."""
    print("Installing cursor-cv-mcp package...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ Package installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install package: {e}")
        sys.exit(1)

def copy_mcp_config():
    """Copy the MCP config file to Cursor's MCP directory."""
    mcp_dir = get_cursor_mcp_dir()
    mcp_file = Path("mcp_cv_tool.json")
    
    if not mcp_file.exists():
        print(f"❌ MCP configuration file not found: {mcp_file}")
        sys.exit(1)
    
    # Create directory if it doesn't exist
    mcp_dir.mkdir(parents=True, exist_ok=True)
    
    dest_file = mcp_dir / "screenshot-tool.json"
    
    print(f"Copying MCP config to: {dest_file}")
    try:
        shutil.copy2(mcp_file, dest_file)
        print(f"✅ MCP config installed at: {dest_file}")
    except Exception as e:
        print(f"❌ Failed to copy MCP config: {e}")
        sys.exit(1)

def check_installation():
    """Verify the installation by checking if the MCP config exists in Cursor's directory."""
    mcp_dir = get_cursor_mcp_dir()
    installed_config = mcp_dir / "screenshot-tool.json"
    
    if installed_config.exists():
        print(f"✅ Configuration file found at: {installed_config}")
        return True
    else:
        print(f"❌ Configuration file not found at: {installed_config}")
        return False

def main():
    """Main installation function."""
    print("==== Cursor CV MCP Tool Installer ====")
    
    # Install the package
    install_package()
    
    # Copy MCP config
    copy_mcp_config()
    
    # Verify installation
    if check_installation():
        print("\n✨ Installation complete! ✨")
        print("\nPlease restart Cursor IDE to complete the installation.")
        print("To verify, open Claude in Cursor and type: 'Use the screenshot tool to capture my screen'")
    else:
        print("\n⚠️ Installation may have issues. Please check the following:")
        print("1. Ensure Cursor IDE is installed properly")
        print("2. Check if you have permission to write to the Cursor MCP directory")
        print("3. Try running this script with administrator privileges")
        print("4. Manually copy mcp_cv_tool.json to your Cursor MCP directory")
        
        # Print Cursor MCP directory for reference
        print(f"\nCursor MCP directory: {get_cursor_mcp_dir()}")

if __name__ == "__main__":
    main() 