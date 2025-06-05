import speech_recognition as sr
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter()

class MicrophoneResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None


@router.post("/listen")
async def listen_to_microphone() -> Dict[str, Any]:
    """
    Listen to the microphone and convert speech to text.
    
    Returns:
        Dict[str, Any]: A dictionary containing success status and either the transcribed text or an error message.
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("Listening to microphone... Speak now.")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            # Listen for audio input
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            print("Processing speech...")
            # Use Google's speech recognition
            text = recognizer.recognize_google(audio)
            
            return {"success": True, "text": text}
    except sr.WaitTimeoutError:
        return {"success": False, "error": "No speech detected. Timeout."}
    except sr.UnknownValueError:
        return {"success": False, "error": "Could not understand audio"}
    except sr.RequestError as e:
        return {"success": False, "error": f"Speech recognition service error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# MCP tool definition for Cursor
mcp_tool_definition = {
    "name": "listen_microphone",
    "description": "Listen to the user's microphone and convert speech to text",
    "parameters": {}
}

def get_mcp_tool():
    """
    Returns the MCP tool definition for the microphone functionality.
    
    Returns:
        dict: The tool definition for Cursor MCP.
    """
    return {
        "router": router,
        "prefix": "/microphone",
        "tags": ["microphone"],
        "tool_definition": mcp_tool_definition
    }
