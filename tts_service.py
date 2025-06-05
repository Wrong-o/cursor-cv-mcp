import os
import tempfile
from typing import Optional
from gtts import gTTS
import threading

class TTSService:
    def __init__(self):
        self.initialized = True
        self.lock = threading.Lock()
        
    def generate_speech(self, text: str, speaker_id: Optional[int] = 0) -> str:
        """Generate speech from text and return the path to the audio file
        
        Args:
            text: Text to convert to speech
            speaker_id: Not used in gTTS but kept for API compatibility
            
        Returns:
            Path to generated audio file
        """
        # Handle empty text
        if not text.strip():
            return None
            
        try:
            # Create temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                filepath = temp_file.name
            
            # Create a lock to prevent concurrent access
            with self.lock:
                # Generate speech using gTTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(filepath)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise

# Create a singleton instance
tts_service = TTSService() 