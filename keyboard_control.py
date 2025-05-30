import pyautogui
import pyperclip
import time


def keyboard_type_text(text: str) -> bool:
    pyperclip.copy(text)
    time.sleep(0.1)
    pyautogui.hotkey('ctrl', 'v')
    return True

def keyboard_press_keys(keys: list[str]) -> bool:
    for key in keys:
        pyautogui.keyDown(key)
    for key in keys:
        pyautogui.keyUp(key)
    return True
