#!/usr/bin/env python3
"""
Python wrapper for Cursor to use.
This script ensures that the MCP tool package is in the Python path.
"""

import os
import sys
import site
import subprocess
import glob
from pathlib import Path

def find_site_packages(venv_path):
    """Find the site-packages directory for any Python version in the venv."""
    # Look for any Python version's site-packages
    site_packages_glob = os.path.join(venv_path, "lib", "python*", "site-packages")
    site_packages_paths = glob.glob(site_packages_glob)
    
    if site_packages_paths:
        return site_packages_paths[0]
    
    # Try Windows path structure
    win_site_packages = os.path.join(venv_path, "Lib", "site-packages")
    if os.path.exists(win_site_packages):
        return win_site_packages
    
    return None

# Find potential virtual environments
current_dir = os.path.dirname(os.path.abspath(__file__))
potential_venvs = [
    os.path.join(current_dir, "cv_venv"),
    os.path.join(current_dir, "venv"),
    os.path.join(os.path.dirname(current_dir), "cv_venv"),
    os.path.join(os.path.dirname(current_dir), "venv")
]

# Try to find a site-packages directory
for venv_path in potential_venvs:
    site_packages = find_site_packages(venv_path)
    if site_packages and os.path.exists(site_packages):
        sys.path.insert(0, site_packages)
        print(f"Added {site_packages} to Python path")
        break
else:
    print("Warning: Could not find a valid virtual environment site-packages")
    print("Falling back to system Python path")

# Print Python info for debugging
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")

# Import the module to verify it can be found
try:
    import mcp_cv_tool
    print(f"Successfully imported mcp_cv_tool from {mcp_cv_tool.__file__}")
except ImportError as e:
    print(f"Error importing mcp_cv_tool: {e}")
    print("Attempting to install the package...")
    try:
        # Try to install the package in the current environment
        setup_path = os.path.join(os.path.dirname(current_dir), "setup.py")
        if os.path.exists(setup_path):
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", os.path.dirname(setup_path)], check=True)
            print("Package installed successfully via pip")
            # Try importing again
            try:
                import mcp_cv_tool
                print(f"Successfully imported mcp_cv_tool after installation from {mcp_cv_tool.__file__}")
            except ImportError as e2:
                print(f"Still unable to import mcp_cv_tool after installation: {e2}")
        else:
            print(f"Could not find setup.py at {setup_path}")
    except Exception as e:
        print(f"Failed to install package: {e}")

# Execute the original command
if len(sys.argv) > 1:
    # The first argument is the module to run
    module = sys.argv[1]
    args = sys.argv[2:]
    
    print(f"Executing: {module} with args {args}")
    
    try:
        if module == "mcp_cv_tool.server":
            # Import the module but don't pass sys.argv to it
            import mcp_cv_tool.server
            # Save original sys.argv and restore it after execution
            original_argv = sys.argv
            sys.argv = [sys.argv[0]]
            try:
                mcp_cv_tool.server.run_server()
            finally:
                sys.argv = original_argv
        else:
            # Default to subprocess execution
            subprocess.run([sys.executable, "-m", module] + args)
    except Exception as e:
        print(f"Error executing {module}: {e}")
        sys.exit(1)
else:
    print("No module specified to run")
    sys.exit(1) 