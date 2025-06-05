import requests
import tempfile
import os
import subprocess
import sys
import urllib.parse

def play_speech(text):
    # URL encode the text
    encoded_text = urllib.parse.quote(text)
    
    # API endpoint
    url = f"http://127.0.0.1:8002/read_with_voice?text={encoded_text}"
    
    try:
        # Get the audio data
        response = requests.get(url)
        response.raise_for_status()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(response.content)
        
        # Play the audio using VLC
        try:
            subprocess.run(['cvlc', '--play-and-exit', '--no-video', temp_file_path], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            # If cvlc not found, try regular vlc
            try:
                subprocess.run(['vlc', '--play-and-exit', '--no-video', temp_file_path], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                print("Error: VLC player not found. Please install VLC or specify another player.")
                return False
        
        # Remove the temporary file after playing
        os.unlink(temp_file_path)
        return True
        
    except requests.RequestException as e:
        print(f"Error fetching audio: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Get text from command line argument or use default
    text = sys.argv[1] if len(sys.argv) > 1 else "Hello there! This is a nice message for you. Have a wonderful day!"
    
    # Play the speech
    play_speech(text) 