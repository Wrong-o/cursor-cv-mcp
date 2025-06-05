#!/usr/bin/env python3
import requests
import tempfile
import os
import subprocess
import sys
import urllib.parse

def say(text):
    """Use the read_with_voice API to speak the given text"""
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
        
        # Try different players in order of preference
        players = [
            ['cvlc', '--play-and-exit', '--no-video'],
            ['vlc', '--play-and-exit', '--no-video'],
            ['mpv', '--no-video'],
            ['mplayer']
        ]
        
        success = False
        for player_cmd in players:
            try:
                cmd = player_cmd + [temp_file_path]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success = True
                break
            except FileNotFoundError:
                continue
        
        if not success:
            print("Error: No compatible media player found. Please install VLC, MPV, or MPlayer.")
            
        # Remove the temporary file after playing
        os.unlink(temp_file_path)
        return success
        
    except requests.RequestException as e:
        print(f"Error fetching audio: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Get text from command line arguments
    if len(sys.argv) > 1:
        # Join all arguments to form the complete text
        text = " ".join(sys.argv[1:])
    else:
        text = "Hello there! This is a nice message for you. Have a wonderful day!"
    
    # Speak the text
    say(text) 