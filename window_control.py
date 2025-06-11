import platform
import subprocess
from typing import List, Dict, Any, Optional
import os

# Conditional imports based on platform
system = platform.system()
PYGETWINDOW_AVAILABLE = False

if system != "Linux":  # Only try to import PyGetWindow on non-Linux platforms
    try:
        import pygetwindow as gw
        PYGETWINDOW_AVAILABLE = True
    except ImportError:
        PYGETWINDOW_AVAILABLE = False

def get_open_windows() -> List[Dict[str, Any]]:
    """
    Get a list of all open windows across different platforms.
    
    Returns:
        List of dictionaries containing window information:
        - title: Window title
        - id: Window identifier (when available)
        - position: x, y coordinates (when available)
        - size: width, height (when available)
    """
    system = platform.system()
    
    # Try using pygetwindow first (on supported platforms)
    if system != "Linux" and PYGETWINDOW_AVAILABLE:
        try:
            windows = []
            for window in gw.getAllWindows():
                windows.append({
                    "title": window.title,
                    "position": {"x": window.left, "y": window.top},
                    "size": {"width": window.width, "height": window.height}
                })
            return windows
        except Exception as e:
            print(f"PyGetWindow error: {e}")
            # Fall back to platform-specific methods
    
    # Platform-specific fallback methods
    if system == "Windows":
        return _get_windows_windows()
    elif system == "Darwin":  # macOS
        return _get_windows_macos()
    elif system == "Linux":
        return _get_windows_linux()
    else:
        print(f"Unsupported platform: {system}")
        return []

def _get_windows_windows() -> List[Dict[str, Any]]:
    """Get open windows on Windows using EnumWindows"""
    try:
        # Try using win32gui if available
        import win32gui
        
        def callback(hwnd, windows_list):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    rect = win32gui.GetWindowRect(hwnd)
                    x, y, right, bottom = rect
                    width = right - x
                    height = bottom - y
                    windows_list.append({
                        "title": title,
                        "id": hwnd,
                        "position": {"x": x, "y": y},
                        "size": {"width": width, "height": height}
                    })
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        return windows
    except ImportError:
        print("win32gui not available")
        
        # Fall back to using just process information
        try:
            import psutil
            windows = []
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if proc.info['name'] != 'System' and proc.info['name'] != 'Registry':
                        windows.append({
                            "title": proc.info['name'],
                            "id": proc.info['pid'],
                            "process_name": proc.info['name']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return windows
        except ImportError:
            print("psutil not available")
            return []

def _get_windows_macos() -> List[Dict[str, Any]]:
    """Get open windows on macOS using AppleScript"""
    try:
        script = """
        tell application "System Events"
            set appList to {}
            set allProcesses to processes whose background only is false
            repeat with oneProcess in allProcesses
                set procName to name of oneProcess
                set procWindows to {}
                try
                    repeat with i from 1 to count windows of oneProcess
                        set procWindow to name of window i of oneProcess
                        if procWindow is not "" then
                            copy {title:procWindow, app:procName} to end of procWindows
                        end if
                    end repeat
                end try
                if length of procWindows > 0 then
                    copy {name:procName, windows:procWindows} to end of appList
                end if
            end repeat
            return appList
        end tell
        """
        result = subprocess.run(['osascript', '-e', script], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            # Fall back to using just process information
            return _get_processes_fallback()
        
        # Parse the AppleScript output and convert to our format
        windows = []
        for line in result.stdout.strip().split(", {"):
            if "title" in line:
                parts = line.replace("{", "").replace("}", "").split(", ")
                title = ""
                app = ""
                for part in parts:
                    if part.startswith("title:"):
                        title = part.replace("title:", "").strip()
                    elif part.startswith("app:"):
                        app = part.replace("app:", "").strip()
                
                if title:
                    windows.append({
                        "title": title,
                        "application": app,
                        "position": {"x": 0, "y": 0},  # Not available in this method
                        "size": {"width": 0, "height": 0}  # Not available in this method
                    })
        
        return windows
    except Exception as e:
        print(f"Error getting macOS windows: {e}")
        return _get_processes_fallback()

def _get_windows_linux() -> List[Dict[str, Any]]:
    """Get open windows on Linux using wmctrl if available"""
    try:
        result = subprocess.run(['wmctrl', '-l', '-G'], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            # Try xdotool or fall back to process list
            return _try_xdotool_linux()
        
        windows = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(None, 8)
                if len(parts) >= 9:
                    window_id, desktop, x, y, width, height, host, *_ = parts[:8]
                    title = parts[8] if len(parts) > 8 else ""
                    
                    windows.append({
                        "title": title,
                        "id": window_id,
                        "desktop": desktop,
                        "position": {"x": int(x), "y": int(y)},
                        "size": {"width": int(width), "height": int(height)},
                        "host": host
                    })
        
        return windows
    except FileNotFoundError:
        # Try alternative method with xdotool
        return _try_xdotool_linux()
    except Exception as e:
        print(f"Error getting Linux windows with wmctrl: {e}")
        return _get_processes_fallback()

def _try_xdotool_linux() -> List[Dict[str, Any]]:
    """Try to get window information using xdotool"""
    try:
        win_ids_result = subprocess.run(['xdotool', 'search', '--onlyvisible', '--name', '.*'], 
                                      capture_output=True, text=True)
        
        if win_ids_result.returncode != 0:
            return _get_processes_fallback()
        
        windows = []
        for win_id in win_ids_result.stdout.strip().split('\n'):
            if win_id:
                # Get window name
                name_result = subprocess.run(['xdotool', 'getwindowname', win_id], 
                                           capture_output=True, text=True)
                
                # Get window geometry
                geom_result = subprocess.run(['xdotool', 'getwindowgeometry', win_id], 
                                           capture_output=True, text=True)
                
                title = name_result.stdout.strip() if name_result.returncode == 0 else ""
                
                # Parse geometry output
                position = {"x": 0, "y": 0}
                size = {"width": 0, "height": 0}
                
                if geom_result.returncode == 0:
                    for line in geom_result.stdout.strip().split('\n'):
                        if "Position:" in line:
                            pos_parts = line.split("Position:")[1].strip().split(",")
                            if len(pos_parts) == 2:
                                position["x"] = int(pos_parts[0])
                                position["y"] = int(pos_parts[1])
                        elif "Geometry:" in line:
                            size_parts = line.split("Geometry:")[1].strip().split("x")
                            if len(size_parts) == 2:
                                size["width"] = int(size_parts[0])
                                size["height"] = int(size_parts[1])
                
                windows.append({
                    "title": title,
                    "id": win_id,
                    "position": position,
                    "size": size
                })
        
        return windows
    except Exception as e:
        print(f"Error getting Linux windows with xdotool: {e}")
        return _get_processes_fallback()

def _get_processes_fallback() -> List[Dict[str, Any]]:
    """Fall back to just listing processes when window managers are not available"""
    try:
        import psutil
        windows = []
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            try:
                windows.append({
                    "title": proc.info['name'],
                    "id": proc.info['pid'],
                    "user": proc.info['username'],
                    "process_name": proc.info['name']
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return windows
    except ImportError:
        print("psutil not available")
        return []

def activate_window(window_id: str = None, window_title: str = None) -> bool:
    """
    Activate (bring to foreground) a specific window.
    
    Args:
        window_id: ID of the window to activate (platform-specific)
        window_title: Title of the window to activate (if window_id not provided)
        
    Returns:
        Success status
    """
    system = platform.system()
    
    if system == "Windows":
        return _activate_window_windows(window_id, window_title)
    elif system == "Darwin":  # macOS
        return _activate_window_macos(window_title)
    elif system == "Linux":
        return _activate_window_linux(window_id, window_title)
    else:
        print(f"Unsupported platform: {system}")
        return False

def launch_application(app_name: str, app_path: str = None) -> bool:
    """
    Launch an application.
    
    Args:
        app_name: Name of the application to launch
        app_path: Full path to the application (if needed)
        
    Returns:
        Success status - True if application was launched successfully, False otherwise
    """
    system = platform.system()
    
    # Try using the full path if provided
    if app_path and os.path.exists(app_path):
        try:
            if system == "Windows":
                process = subprocess.Popen([app_path], shell=True)
            else:
                process = subprocess.Popen([app_path])
            
            # Check if process is still running after a brief moment
            import time
            time.sleep(0.5)
            if process.poll() is None:  # None means still running
                return True
            # If process exited immediately, it might be an error
            return process.returncode == 0
        except Exception as e:
            print(f"Error launching application from path {app_path}: {e}")
            return False
    
    # Platform-specific application launch methods
    if system == "Windows":
        try:
            process = subprocess.Popen([app_name], shell=True)
            import time
            time.sleep(0.5)
            if process.poll() is None:  # None means still running
                return True
            return process.returncode == 0
        except Exception as e:
            print(f"Error launching application {app_name}: {e}")
            return False
    elif system == "Darwin":  # macOS
        try:
            script = f"""
            tell application "{app_name}"
                activate
            end tell
            """
            result = subprocess.run(['osascript', '-e', script], 
                                  capture_output=True, text=True)
            
            # Check if the application is actually running now
            verify_script = f"""
            tell application "System Events"
                return exists process "{app_name}"
            end tell
            """
            verify_result = subprocess.run(['osascript', '-e', verify_script], 
                                        capture_output=True, text=True)
            
            return "true" in verify_result.stdout.lower()
        except Exception as e:
            print(f"Error launching application {app_name}: {e}")
            return False
    elif system == "Linux":
        try:
            # Try different ways to launch applications on Linux
            methods = [
                [app_name.lower()],  # Try lowercase
                [app_name],          # Try as-is
                ["gtk-launch", app_name],
                ["xdg-open", app_name]
            ]
            
            for method in methods:
                try:
                    process = subprocess.Popen(method, 
                                            stdout=subprocess.DEVNULL, 
                                            stderr=subprocess.DEVNULL)
                    
                    # Check if process is still running after a brief moment
                    import time
                    time.sleep(0.5)
                    
                    # First, check if the process we launched is still running
                    if process.poll() is None:
                        return True
                    
                    # If the process exited, it might have spawned another process
                    # Let's check if a process with the app_name is running
                    try:
                        # Use ps and grep to find if the application is running
                        check_cmd = f"ps aux | grep -i '{app_name}' | grep -v grep"
                        check_result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE)
                        if check_result.stdout and len(check_result.stdout) > 0:
                            return True
                    except Exception as check_err:
                        print(f"Error checking if app is running: {check_err}")
                
                except Exception as method_err:
                    print(f"Error with launch method {method}: {method_err}")
                    continue
                    
            return False
        except Exception as e:
            print(f"Error launching application {app_name}: {e}")
            return False
    else:
        print(f"Unsupported platform: {system}")
        return False

def _activate_window_windows(window_id=None, window_title=None):
    """Activate a window on Windows"""
    try:
        import win32gui
        
        if window_id:
            # Try to use the window handle directly
            try:
                win32gui.SetForegroundWindow(int(window_id))
                return True
            except Exception as e:
                print(f"Error activating window by ID: {e}")
        
        if window_title:
            # Find the window by title
            def callback(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if window_title.lower() in title.lower():
                        results.append(hwnd)
                return True
            
            found_windows = []
            win32gui.EnumWindows(callback, found_windows)
            
            if found_windows:
                try:
                    win32gui.SetForegroundWindow(found_windows[0])
                    return True
                except Exception as e:
                    print(f"Error activating window: {e}")
        
        return False
    except ImportError:
        print("win32gui not available")
        return False

def _activate_window_macos(window_title=None):
    """Activate a window on macOS"""
    if not window_title:
        return False
        
    try:
        # Try to activate the application with the given title
        script = f"""
        tell application "System Events"
            set frontApp to first application process whose frontmost is true
            set frontAppName to name of frontApp
            
            set appList to every application process whose visible is true
            repeat with oneApp in appList
                set appName to name of oneApp
                if appName contains "{window_title}" then
                    set frontmost of oneApp to true
                    return true
                end if
                
                repeat with oneWindow in windows of oneApp
                    if name of oneWindow contains "{window_title}" then
                        set frontmost of oneApp to true
                        return true
                    end if
                end repeat
            end repeat
            
            return false
        end tell
        """
        
        result = subprocess.run(['osascript', '-e', script], 
                               capture_output=True, text=True)
        return "true" in result.stdout.lower()
    except Exception as e:
        print(f"Error activating window on macOS: {e}")
        return False

def _activate_window_linux(window_id=None, window_title=None):
    """Activate a window on Linux"""
    # Try wmctrl first
    try:
        if window_id:
            result = subprocess.run(['wmctrl', '-i', '-a', window_id], 
                                  capture_output=True, text=True)
            return result.returncode == 0
                
        if window_title:
            result = subprocess.run(['wmctrl', '-a', window_title], 
                                  capture_output=True, text=True)
            return result.returncode == 0
                
        return False
    except FileNotFoundError:
        # Try xdotool as an alternative
        try:
            if window_id:
                result = subprocess.run(['xdotool', 'windowactivate', window_id], 
                                      capture_output=True, text=True)
                return result.returncode == 0
                    
            if window_title:
                result = subprocess.run(['xdotool', 'search', '--name', window_title, 'windowactivate'], 
                                      capture_output=True, text=True)
                return result.returncode == 0
                    
            return False
        except FileNotFoundError:
            print("Neither wmctrl nor xdotool available on this Linux system")
            return False
        except Exception as e:
            print(f"Error activating window on Linux: {e}")
            return False 