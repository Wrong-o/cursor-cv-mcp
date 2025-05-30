#!/usr/bin/env python3
"""
Unified installer for cursor-cv-mcp
This script performs all installation steps for the Cursor CV MCP tool:
1. Creates a virtual environment (if requested)
2. Installs the package
3. Registers the MCP with Cursor
"""

import os
import sys
import shutil
import platform
import subprocess
import argparse
from pathlib import Path

def create_venv(venv_path, force=False):
    """Create a virtual environment for the MCP tool."""
    venv_path = Path(venv_path)
    
    if venv_path.exists() and force:
        print(f"Removing existing virtual environment at {venv_path}")
        shutil.rmtree(venv_path)
    
    if not venv_path.exists():
        print(f"Creating virtual environment at {venv_path}")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print(f"✅ Virtual environment created at {venv_path}")
            
            # Get the Python executable from the virtual environment
            if platform.system() == "Windows":
                venv_python = venv_path / "Scripts" / "python.exe"
            else:
                venv_python = venv_path / "bin" / "python"
                
            # Update pip in the virtual environment
            print("Updating pip in virtual environment...")
            subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            return venv_python
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return None
    else:
        print(f"Using existing virtual environment at {venv_path}")
        # Get the Python executable from the virtual environment
        if platform.system() == "Windows":
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"
        return venv_python

def install_package(python_exe, editable=True):
    """Install the package using pip."""
    print("Installing cursor-cv-mcp package...")
    cmd = [str(python_exe), "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(".")
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Package installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install package: {e}")
        return False

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

def register_mcp_with_cursor():
    """Copy the MCP config file to Cursor's MCP directory."""
    mcp_dir = get_cursor_mcp_dir()
    mcp_file = Path("mcp_cv_tool.json")
    
    if not mcp_file.exists():
        print(f"❌ MCP configuration file not found: {mcp_file}")
        return False
    
    # Create directory if it doesn't exist
    mcp_dir.mkdir(parents=True, exist_ok=True)
    
    dest_file = mcp_dir / "screenshot-tool.json"
    
    print(f"Copying MCP config to: {dest_file}")
    try:
        shutil.copy2(mcp_file, dest_file)
        print(f"✅ MCP config installed at: {dest_file}")
        return True
    except Exception as e:
        print(f"❌ Failed to copy MCP config: {e}")
        return False

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
    parser = argparse.ArgumentParser(description="Install cursor-cv-mcp in Cursor IDE")
    parser.add_argument("--venv", default="cv_venv", help="Path to create/use virtual environment")
    parser.add_argument("--force", action="store_true", help="Force recreation of virtual environment")
    parser.add_argument("--no-venv", action="store_true", help="Don't create a virtual environment")
    args = parser.parse_args()
    
    print("==== Cursor CV MCP Tool Installer ====")
    
    python_exe = sys.executable
    
    # Create virtual environment if requested
    if not args.no_venv:
        venv_python = create_venv(args.venv, args.force)
        if venv_python:
            python_exe = venv_python
    
    # Install the package
    install_success = install_package(python_exe)
    if not install_success:
        print("⚠️ Package installation failed. Attempting to continue...")
    
    # Register with Cursor
    register_success = register_mcp_with_cursor()
    
    # Verify installation
    if register_success and check_installation():
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