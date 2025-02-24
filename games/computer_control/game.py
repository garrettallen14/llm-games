"""Computer control game implementation with LangChain agents."""

import time
from typing import Dict, Optional, List, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
import pyautogui
import base64
import io
from PIL import Image, ImageDraw

from base.llm_player import BaseLLMPlayer
from games.computer_control.tools import (
    MouseMoveTool, MouseClickTool, KeyboardTypeTool, KeyPressTool,
    HotkeyTool, ScrollTool, MouseDragTool, KeyHoldTool, KeyReleaseTool,
    WaitTool
)

@dataclass
class ComputerControlConfig:
    """Configuration settings for computer control game.
    
    Attributes:
        run_dir: Directory for game artifacts
        max_turns: Maximum allowed game turns
    """
    run_dir: Path
    max_turns: int = 100

class ComputerControlGame:
    """Computer control game manager using LangChain agents."""
    
    def __init__(self, run_dir: Path, max_turns: int = 100) -> None:
        """Initialize computer control game with configuration.

        Args:
            run_dir: Directory for game artifacts
            max_turns: Maximum allowed game turns
        """
        self.config = ComputerControlConfig(run_dir=run_dir, max_turns=max_turns)
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize tools
        self.tools = [
            MouseMoveTool(),
            MouseClickTool(), 
            KeyboardTypeTool(),
            KeyPressTool(),
            HotkeyTool(),
            ScrollTool(),
            MouseDragTool(),
            KeyHoldTool(),
            KeyReleaseTool(),
            WaitTool()
        ]

    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for the agent."""
        screen_info = f"""SCREEN:
- Dimensions: {self.screen_width}x{self.screen_height}
- Origin (0,0): top-left corner
- Grid: 100x100 pixel boxes for navigation (count boxes * 100 for coordinates)"""

        return {
            "role": "system",
            "content": """You control the computer using these commands:

MOUSE COMMANDS:
mouse_move   {"x": int, "y": int}           # Move cursor to coordinates
mouse_click  {"x": int, "y": int}           # Click at coordinates
mouse_drag   {"x": int, "y": int}           # Drag to coordinates
scroll       {"amount": int}                # Positive=up, negative=down

KEYBOARD COMMANDS:
keyboard_type {"text": str}                 # Type text
key_press    {"key": str}                  # Press single key (enter/tab/space/etc)
key_hold     {"key": str}                  # Hold key down
key_release  {"key": str}                  # Release held key
hotkey       {"key1": str, "key2": str}    # Press key combination

UTILITY:
wait         {"duration": float}           # Wait specified seconds (default: 0.5)

""" + screen_info + """

Guidelines:
1. You may only submit one command at a time.
2. You must only say DONE when you are sure the task is complete.

Format commands as:
TOOL: command_name
PARAMS: {param_dict}

Your current TASK,
TASK: Find the funniest cat meme that you would actually lol at

Say DONE when complete."""
        }

    def get_game_image(self) -> Optional[str]:
        """Take a screenshot and return it as a base64 encoded string with grid overlay.

        Returns:
            Base64 encoded PNG data URI or None if screenshot fails
        """
        try:
            # Take a screenshot
            screenshot = pyautogui.screenshot()
            
            # Get current mouse position
            x, y = pyautogui.position()
            
            # Convert to PIL Image for drawing
            img = Image.frombytes('RGB', screenshot.size, screenshot.tobytes())
            draw = ImageDraw.Draw(img, 'RGBA')  # Use RGBA for transparency
            
            # Draw grid overlay
            grid_spacing = 100  # Grid every 100 pixels
            grid_color = (128, 128, 128, 64)  # Semi-transparent gray
            
            # Draw vertical lines
            for i in range(0, img.width, grid_spacing):
                draw.line([(i, 0), (i, img.height)], fill=grid_color, width=1)
            
            # Draw horizontal lines
            for i in range(0, img.height, grid_spacing):
                draw.line([(0, i), (img.width, i)], fill=grid_color, width=1)
            
            # Draw cursor (red circle with crosshair)
            cursor_radius = 10
            cursor_color = "red"
            
            # Draw circle
            draw.ellipse([x - cursor_radius, y - cursor_radius, 
                         x + cursor_radius, y + cursor_radius], 
                        outline=cursor_color, width=2)
            
            # Draw crosshair
            draw.line([x - cursor_radius, y, x + cursor_radius, y], 
                     fill=cursor_color, width=2)  # Horizontal line
            draw.line([x, y - cursor_radius, x, y + cursor_radius], 
                     fill=cursor_color, width=2)  # Vertical line
            
            # Save as environment.png
            img.save("environment.png")
            
            # Save to bytes for base64
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            # Convert to base64
            b64_img = base64.b64encode(img_bytes).decode()
            
            return f"data:image/png;base64,{b64_img}"
            
        except Exception as e:
            print(f"Warning: Failed to take screenshot: {e}")
            return None

    def _parse_tool_response(self, response: str) -> Optional[Dict]:
        """Parse the tool response from the LLM.
        
        Args:
            response: Response from the LLM
            
        Returns:
            Dict containing tool name and parameters, or None if invalid
        """
        try:
            # Check for DONE response
            if "DONE" in response.upper() or "TASK IS COMPLETE" in response.upper():
                return {"done": True}
                
            lines = response.strip().split('\n')
            tool_line = None
            params_line = None
            
            for line in lines:
                if line.startswith('TOOL:'):
                    tool_line = line[5:].strip()
                elif line.startswith('PARAMS:'):
                    params_line = line[7:].strip()
            
            if not tool_line:
                return None
                
            # Find the matching tool
            tool = None
            for t in self.tools:
                if t.name == tool_line:
                    tool = t
                    break
                    
            if not tool:
                return None
                
            # Parse parameters if needed
            params = {}
            if params_line:
                try:
                    params = eval(params_line)  # Safe since we control the input format
                except:
                    return None
                    
            return {
                "tool": tool,
                "params": params
            }
            
        except Exception as e:
            print(f"Error parsing tool response: {e}")
            return None

    def run(self, players: List[BaseLLMPlayer]) -> None:
        """Run the computer control game.

        Args:
            players: List of LLM players
        """
        if not players:
            raise ValueError("No players provided")
            
        player = players[0]  # Use the first player
        
        # Initialize the player with system prompt
        player.initialize_with_prompt(self.get_system_prompt())
        
        turn = 0
        while turn < self.config.max_turns:
            # Get current state including screenshot
            state_message = {
                "role": "user",
                "content": f"Current mouse position: {pyautogui.position()}\nWhat action would you like to take?"
            }
            print(f"\n[USER] {state_message['content']}")
            
            # Get response from player
            screenshot = self.get_game_image()
            response = player.get_response(state_message, screenshot)
            print(f"\n[ASSISTANT] {response}")
            
            # Parse and execute tool
            tool_info = self._parse_tool_response(response)
            if not tool_info:
                message = "Invalid tool format. Please use TOOL: and PARAMS: format."
                print(f"\n[SYSTEM] {message}")
                player.add_message({
                    "role": "system",
                    "content": message
                })
                continue
            
            # Check if task is complete
            if "done" in tool_info:
                message = "Task completed successfully!"
                print(f"\n[SYSTEM] {message}")
                player.add_message({
                    "role": "system",
                    "content": message
                })
                break
                
            try:
                result = tool_info["tool"]._run(**tool_info["params"])
                message = f"Action completed: {result}"
                print(f"\n[SYSTEM] {message}")
                player.add_message({
                    "role": "system",
                    "content": message
                })
            except Exception as e:
                message = f"Error executing tool: {str(e)}"
                print(f"\n[SYSTEM] {message}")
                player.add_message({
                    "role": "system",
                    "content": message
                })
                continue
                
            turn += 1
            
        self.end_time = datetime.now().isoformat()
