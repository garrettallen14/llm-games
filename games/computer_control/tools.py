"""Custom tools for computer control using LangChain."""

from typing import Optional, Annotated
import pyautogui
import keyboard
import time
from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field

# Configure pyautogui safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

class MouseMoveInput(BaseModel):
    x: int = Field(description="X coordinate to move mouse to")
    y: int = Field(description="Y coordinate to move mouse to")

class MouseClickInput(BaseModel):
    x: int = Field(description="X coordinate to click at")
    y: int = Field(description="Y coordinate to click at")

class KeyboardInput(BaseModel):
    text: str = Field(description="Text to type")

class KeyPressInput(BaseModel):
    key: str = Field(description="Key to press (e.g. enter, tab, space)")

class HotkeyInput(BaseModel):
    key1: str = Field(description="First key in combination")
    key2: str = Field(description="Second key in combination")

class ScrollInput(BaseModel):
    amount: int = Field(description="Amount to scroll (positive=up, negative=down)")

class MouseDragInput(BaseModel):
    x: int = Field(description="X coordinate to drag to")
    y: int = Field(description="Y coordinate to drag to")

class KeyHoldInput(BaseModel):
    key: str = Field(description="Key to hold down")

class MouseMoveTool(BaseTool):
    name: str = "mouse_move"
    description: str = "Move mouse cursor to specified coordinates"
    args_schema: type[BaseModel] = MouseMoveInput

    def _run(self, x: int, y: int) -> str:
        pyautogui.moveTo(x, y)
        return f"Moved mouse to ({x}, {y})"

class MouseClickTool(BaseTool):
    name: str = "mouse_click" 
    description: str = "Click mouse at specified coordinates"
    args_schema: type[BaseModel] = MouseClickInput

    def _run(self, x: int, y: int) -> str:
        pyautogui.click(x, y)
        time.sleep(0.5)  # Wait for UI to update
        return f"Clicked at ({x}, {y})"

class KeyboardTypeTool(BaseTool):
    name: str = "keyboard_type"
    description: str = "Type specified text"
    args_schema: type[BaseModel] = KeyboardInput

    def _run(self, text: str) -> str:
        # Type with a small delay between keystrokes for more natural typing
        pyautogui.write(text, interval=0.1)
        # Wait after typing is complete to ensure UI updates
        time.sleep(0.5)
        return f"Typed: {text}"

class KeyPressTool(BaseTool):
    name: str = "key_press"
    description: str = "Press a single key"
    args_schema: type[BaseModel] = KeyPressInput

    def _run(self, key: str) -> str:
        pyautogui.press(key)
        return f"Pressed key: {key}"

class HotkeyTool(BaseTool):
    name: str = "hotkey"
    description: str = "Press a key combination"
    args_schema: type[BaseModel] = HotkeyInput

    def _run(self, key1: str, key2: str) -> str:
        pyautogui.hotkey(key1.strip(), key2.strip())
        return f"Pressed hotkey: {key1}+{key2}"

class ScrollTool(BaseTool):
    name: str = "scroll"
    description: str = "Scroll the mouse wheel"
    args_schema: type[BaseModel] = ScrollInput

    def _run(self, amount: int) -> str:
        pyautogui.scroll(amount)
        return f"Scrolled by: {amount}"

class MouseDragTool(BaseTool):
    name: str = "mouse_drag"
    description: str = "Drag mouse to specified coordinates"
    args_schema: type[BaseModel] = MouseDragInput

    def _run(self, x: int, y: int) -> str:
        pyautogui.dragTo(x, y)
        return f"Dragged to ({x}, {y})"

class KeyHoldTool(BaseTool):
    name: str = "key_hold"
    description: str = "Hold down a key"
    args_schema: type[BaseModel] = KeyHoldInput

    def _run(self, key: str) -> str:
        keyboard.press(key)
        return f"Holding key: {key}"

class KeyReleaseTool(BaseTool):
    name: str = "key_release"
    description: str = "Release a held key"
    args_schema: type[BaseModel] = KeyHoldInput

    def _run(self, key: str) -> str:
        keyboard.release(key)
        return f"Released key: {key}"

class WaitInput(BaseModel):
    duration: float = Field(description="Duration to wait in seconds", default=0.5)

class WaitTool(BaseTool):
    name: str = "wait"
    description: str = "Wait for a specified duration (default: 0.5 seconds)"
    args_schema: type[BaseModel] = WaitInput

    def _run(self, duration: float = 0.5) -> str:
        time.sleep(duration)
        return f"Waited for {duration} seconds"
