import os
import platform
import subprocess
import pathlib

def open_downloads_folder():
    """
    Opens the user's Downloads folder based on the operating system.
    
    Returns:
        bool: True if the folder was opened successfully, False otherwise.
    """
    try:
        system = platform.system().lower()
        
        if system == 'linux':
            # Linux - Use xdg-open to open the Downloads directory
            downloads_path = os.path.expanduser("~/Downloads")
            subprocess.Popen(['xdg-open', downloads_path])
            return True
            
        elif system == 'darwin':
            # macOS - Use open command to open the Downloads directory
            downloads_path = os.path.expanduser("~/Downloads")
            subprocess.Popen(['open', downloads_path])
            return True
            
        elif system == 'windows':
            # Windows - Use explorer to open the Downloads directory
            username = os.getenv('USERNAME')
            downloads_path = os.path.join('C:', os.sep, 'Users', username, 'Downloads')
            subprocess.Popen(['explorer', downloads_path])
            return True
            
        else:
            print(f"Unsupported operating system: {system}")
            return False
            
    except Exception as e:
        print(f"Error opening Downloads folder: {str(e)}")
        return False

def get_folder_contents(folder_path):
    """
    Get the contents of a folder.
    
    Args:
        folder_path (str): Path to the folder to list. Can use ~ for home directory.
        
    Returns:
        dict: Dictionary with 'success' flag and 'files' list or 'error' string.
    """
    try:
        # Expand user directory if path contains ~
        expanded_path = os.path.expanduser(folder_path)
        
        # Check if the path exists and is a directory
        if not os.path.exists(expanded_path):
            return {"success": False, "error": f"Path does not exist: {folder_path}"}
        if not os.path.isdir(expanded_path):
            return {"success": False, "error": f"Path is not a directory: {folder_path}"}
        
        # Get the file list
        file_list = []
        for item in os.listdir(expanded_path):
            item_path = os.path.join(expanded_path, item)
            is_dir = os.path.isdir(item_path)
            file_size = 0 if is_dir else os.path.getsize(item_path)
            
            # Get last modified time
            try:
                mtime = os.path.getmtime(item_path)
            except:
                mtime = 0
                
            file_list.append({
                "name": item,
                "path": item_path,
                "is_directory": is_dir,
                "size": file_size,
                "modified_time": mtime
            })
        
        # Sort files: directories first, then alphabetically
        file_list.sort(key=lambda x: (not x["is_directory"], x["name"].lower()))
        
        return {
            "success": True,
            "folder_path": expanded_path,
            "files": file_list
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Test the function
    open_downloads_folder() 