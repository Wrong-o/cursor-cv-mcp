import platform
import os
import subprocess

def get_os_info():
    """
    Returns detailed information about the operating system.
    
    Returns:
        str: A string containing OS name and version
    """
    system = platform.system()
    
    if system == "Linux":
        # Get more detailed Linux information
        try:
            # Try to get distribution info
            dist_info = platform.freedesktop_os_release()
            if "PRETTY_NAME" in dist_info:
                return f"{dist_info['PRETTY_NAME']} {platform.release()}"
        except:
            # Fallback if freedesktop_os_release() is not available
            try:
                # Try to get from lsb_release command
                result = subprocess.run(['lsb_release', '-ds'], 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE,
                                        text=True)
                if result.returncode == 0 and result.stdout.strip():
                    return f"{result.stdout.strip()} {platform.release()}"
            except:
                pass
        
        # Fallback to basic info
        return f"Linux {platform.release()}"
    
    elif system == "Darwin":
        # macOS
        mac_ver = platform.mac_ver()
        return f"macOS {mac_ver[0]} ({platform.machine()})"
    
    elif system == "Windows":
        # Windows
        win_ver = platform.win32_ver()
        return f"Windows {win_ver[0]} {win_ver[1]}"
    
    else:
        # Generic fallback
        return f"{platform.system()} {platform.release()}"

def get_os_name():
    """
    Returns just the name of the operating system without version details.
    
    Returns:
        str: The name of the operating system (Linux, Windows, macOS)
    """
    system = platform.system()
    
    if system == "Darwin":
        return "macOS"
    else:
        return system

if __name__ == "__main__":
    # Test the functions
    print(f"OS Info: {get_os_info()}")
    print(f"OS Name: {get_os_name()}") 