"""
Configuration settings for the Cursor CV MCP tool.
"""

import os
import json
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "screenshots_dir": os.path.expanduser("~/screenshots"),
    "debug": False,
    "compression_quality": 80,
    "default_monitor": 1
}

# Configuration file path
CONFIG_FILE = os.path.expanduser("~/.cursor_cv_mcp.json")

def load_config() -> Dict[str, Any]:
    """
    Load configuration from file or return default config.
    
    Returns:
        Configuration dictionary
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                config = DEFAULT_CONFIG.copy()
                config.update(user_config)
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
    
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Success status
    """
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def get_screenshots_dir() -> str:
    """
    Get the configured screenshots directory.
    
    Returns:
        Path to screenshots directory
    """
    config = load_config()
    screenshots_dir = config.get("screenshots_dir", DEFAULT_CONFIG["screenshots_dir"])
    
    # Create the directory if it doesn't exist
    os.makedirs(screenshots_dir, exist_ok=True)
    
    return screenshots_dir

def set_screenshots_dir(path: str) -> bool:
    """
    Set the screenshots directory.
    
    Args:
        path: New screenshots directory path
    
    Returns:
        Success status
    """
    config = load_config()
    config["screenshots_dir"] = os.path.expanduser(path)
    return save_config(config) 