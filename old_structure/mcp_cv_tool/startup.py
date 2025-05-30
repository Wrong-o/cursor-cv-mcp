#!/usr/bin/env python3
"""
Startup script for the Cursor CV MCP server.
This script is executed when Cursor starts.
"""

import os
import sys
import logging
import subprocess
import time
import signal
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=os.path.join(os.path.expanduser("~"), ".cursor", "logs", "cursor-cv-mcp.log"),
    filemode="a"
)
logger = logging.getLogger(__name__)

def start_server():
    """Start the Cursor CV MCP server as a background process."""
    try:
        # Get the current Python executable
        python_exec = sys.executable
        
        # Prepare the command to start the server
        cmd = [python_exec, "-m", "mcp_cv_tool.server"]
        
        # Start the server as a background process
        logger.info(f"Starting server with command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True  # Start in a new session so it's not killed when Cursor exits
        )
        
        # Wait a bit to make sure the process started successfully
        time.sleep(2)
        
        if process.poll() is None:
            # Process is still running, which is good
            logger.info(f"Server started successfully with PID {process.pid}")
            
            # Write PID to a file for later reference
            pid_file = Path(os.path.expanduser("~")) / ".cursor" / "cursor-cv-mcp.pid"
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(process.pid))
            
            return True
        else:
            # Process exited already
            stdout, stderr = process.communicate()
            logger.error(f"Server failed to start: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        return False

def stop_existing_server():
    """Stop any existing server process."""
    try:
        # Check for PID file
        pid_file = Path(os.path.expanduser("~")) / ".cursor" / "cursor-cv-mcp.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text().strip())
            logger.info(f"Found existing server with PID {pid}")
            
            # Try to terminate the process
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to PID {pid}")
                
                # Wait a bit for the process to terminate
                time.sleep(1)
                
                # Check if it's still running
                try:
                    os.kill(pid, 0)  # This will raise an exception if the process is gone
                    # Process is still running, force kill
                    logger.info(f"Process {pid} still running, sending SIGKILL")
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    # Process is gone
                    logger.info(f"Process {pid} terminated successfully")
            except OSError:
                # Process doesn't exist
                logger.info(f"No process with PID {pid} found")
            
            # Remove the PID file
            pid_file.unlink()
    except Exception as e:
        logger.error(f"Error stopping existing server: {e}")

def main():
    """Main entry point."""
    logger.info("Starting Cursor CV MCP Server startup script")
    
    # Stop any existing server
    stop_existing_server()
    
    # Start the server
    if start_server():
        logger.info("Cursor CV MCP Server startup complete")
    else:
        logger.error("Failed to start Cursor CV MCP Server")

if __name__ == "__main__":
    main() 