"""Painter game implementation with LLM players."""

import numpy as np
from PIL import Image, ImageDraw
import io
import base64
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from datetime import datetime
import re
from dataclasses import dataclass, field

from base.llm_player import BaseLLMPlayer

@dataclass
class PainterState:
    """Tracks the current state of the Painter game.
    
    Attributes:
        grid_size: Size of the nxn grid
        grid: 2D array of RGB values
        move_history: List of moves in format (action_type, value)
        is_finished: Whether the game is done
    """
    grid_size: int = 25
    grid: np.ndarray = field(init=False)
    move_history: List[Tuple[str, Any]] = field(default_factory=list)
    is_finished: bool = False

    def __post_init__(self):
        """Initialize grid with the correct size."""
        self.grid = np.full((self.grid_size, self.grid_size, 3), 255, dtype=np.uint8)

@dataclass
class PainterConfig:
    """Configuration settings for Painter game.
    
    Attributes:
        run_dir: Directory for game artifacts
        max_turns: Maximum allowed game turns
        grid_size: Size of the grid (n x n)
        cell_size: Pixel size of each grid cell
    """
    run_dir: Path
    max_turns: int = 100
    grid_size: int = 100
    cell_size: int = 8  # Smaller size for crisp pixels

class PainterGame:
    """Painter game manager handling game flow and state.
    
    A simple drawing game where an LLM tries to draw a smiley face by coloring cells.
    """
    
    # Command validation patterns
    COLOR_PATTERN = re.compile(r"COLOR:\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    BACKGROUND_PATTERN = re.compile(r"BACKGROUND:\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    BOX_FILL_PATTERN = re.compile(r"BOX_FILL:\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    CIRCLE_PATTERN = re.compile(r"CIRCLE:\s*\((\d+)\s*,\s*(\d+)\)\s*(\d+)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    
    def __init__(self, run_dir: Path, max_turns: int = 10000, grid_size: int = 100) -> None:
        """Initialize Painter game with configuration."""
        self.config = PainterConfig(
            run_dir=run_dir,
            max_turns=max_turns,
            grid_size=grid_size
        )
        self.state = PainterState(grid_size=grid_size)
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None

    @staticmethod
    def get_system_prompt() -> Dict[str, str]:
        """Return the system prompt for LLM players."""
        return {
            "role": "system",
            "content": f"""You have access to a 100x100 grid where you can place colors. You will have 10 turns to work with the grid.

Available commands:

COLOR: (x,y) (r,g,b)
Places a single color at position x,y using RGB values (0-255)
Example: COLOR: (int,int) (int,int,int)

BACKGROUND: (r,g,b) 
Sets the background color using RGB values (0-255)
Only affects uncolored cells
Example: BACKGROUND: (int,int,int)

BOX_FILL: (x1,y1) (x2,y2) (r,g,b)
Colors a rectangular area from (x1,y1) to (x2,y2) using RGB values (0-255)
Example: BOX_FILL: (int,int) (int,int) (int,int,int)

CIRCLE: (x,y) radius (r,g,b)
Colors a filled circle with center at (x,y) and given radius using RGB values (0-255)
Example: CIRCLE: (int,int) int (int,int,int)

Technical Information:
- Grid coordinates: (0,0) at top-left to (99,99) at bottom-right
- Colors persist between turns
- You'll see the current grid state before each turn
- It is encouraged to submit as many commands per turn as you would like
- Invalid commands are skipped

"""
        }

    def create_grid_image(self) -> str:
        """Generate base64 encoded PNG of current grid state."""
        # Calculate image dimensions
        width = self.config.grid_size * self.config.cell_size
        height = width
        
        # Create image and drawing context
        image = Image.fromarray(self.state.grid)
        # Use NEAREST for crisp pixel art
        image = image.resize((width, height), Image.NEAREST)
        
        # Convert to PNG without grid lines for cleaner look
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image.save("board.png", format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def validate_color_command(self, x: int, y: int, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a color command."""
        # Check position bounds
        if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
            return False, f"Position ({x},{y}) is out of bounds"
        
        # Check color values
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def validate_color_values(self, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate RGB color values."""
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def validate_box_fill(self, x1: int, y1: int, x2: int, y2: int, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a box fill command."""
        # Check position bounds for both corners
        for x, y in [(x1, y1), (x2, y2)]:
            if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
                return False, f"Position ({x},{y}) is out of bounds"
        
        # Check color values
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def validate_circle(self, x: int, y: int, radius: int, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a circle command."""
        # Check center position bounds
        if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
            return False, f"Center position ({x},{y}) is out of bounds"
        
        # Check radius is positive and within grid bounds
        if radius <= 0:
            return False, "Radius must be positive"
        
        # Check color values
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def draw_circle(self, x: int, y: int, radius: int, r: int, g: int, b: int) -> None:
        """Draw a filled circle on the grid."""
        y_indices, x_indices = np.ogrid[:self.config.grid_size, :self.config.grid_size]
        # Calculate distances from each point to center
        distances = np.sqrt((x_indices - x)**2 + (y_indices - y)**2)
        # Create circle mask
        circle_mask = distances <= radius
        # Apply color to masked area
        self.state.grid[circle_mask] = [r, g, b]

    def attempt_move(self, response: str) -> Dict[str, Any]:
        """Process and execute a move attempt."""
        executed_actions = []
        has_error = False
        error_message = ""
        
        # Process background command first if present
        bg_match = self.BACKGROUND_PATTERN.search(response)
        if bg_match:
            try:
                r, g, b = map(int, bg_match.groups())
                is_valid, error = self.validate_color_values(r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                else:
                    # Only change cells that are still white (255,255,255)
                    white_mask = np.all(self.state.grid == [255, 255, 255], axis=2)
                    self.state.grid[white_mask] = [r, g, b]
                    self.state.move_history.append(("BACKGROUND", (r, g, b)))
                    executed_actions.append(f"Background set to ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid background format\n"

        # Process box fill commands
        for match in self.BOX_FILL_PATTERN.finditer(response):
            try:
                x1, y1, x2, y2, r, g, b = map(int, match.groups())
                is_valid, error = self.validate_box_fill(x1, y1, x2, y2, r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                    continue
                
                # Ensure x1,y1 is top-left and x2,y2 is bottom-right
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Fill the box
                self.state.grid[y1:y2+1, x1:x2+1] = [r, g, b]
                self.state.move_history.append(("BOX_FILL", ((x1, y1), (x2, y2), (r, g, b))))
                executed_actions.append(f"Box from ({x1},{y1}) to ({x2},{y2}) filled with ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid box fill format\n"
                continue
        
        # Process circle commands
        for match in self.CIRCLE_PATTERN.finditer(response):
            try:
                x, y, radius, r, g, b = map(int, match.groups())
                is_valid, error = self.validate_circle(x, y, radius, r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                    continue
                
                # Draw the circle
                self.draw_circle(x, y, radius, r, g, b)
                self.state.move_history.append(("CIRCLE", ((x, y), radius, (r, g, b))))
                executed_actions.append(f"Circle at ({x},{y}) with radius {radius} colored to ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid circle format\n"
                continue
        
        # Process color commands
        for match in self.COLOR_PATTERN.finditer(response):
            try:
                x, y, r, g, b = map(int, match.groups())
                is_valid, error = self.validate_color_command(x, y, r, g, b)
                
                if not is_valid:
                    has_error = True
                    error_message += error + "\n"
                    continue
                
                # Execute color change
                self.state.grid[y, x] = [r, g, b]
                self.state.move_history.append(("COLOR", ((x, y), (r, g, b))))
                executed_actions.append(f"Cell at ({x},{y}) colored to ({r},{g},{b})")
                    
            except ValueError:
                has_error = True
                error_message += "Invalid color format\n"
                continue
        
        if executed_actions:
            return {
                "valid": True,
                "message": "\n".join(executed_actions),
                "end_turn": True
            }
        
        return {
            "valid": False,
            "message": error_message or "Invalid command format. Use COLOR: (x,y) (r,g,b), BACKGROUND: (r,g,b), BOX_FILL: (x1,y1) (x2,y2) (r,g,b), or CIRCLE: (x,y) radius (r,g,b)",
            "end_turn": False
        }

    # def get_current_state(self) -> str:
    #     """Generate formatted string of current game state."""
    #     turns_remaining = self.config.max_turns - len(self.state.move_history)
    #     state_parts = [
    #         f"Turn: {len(self.state.move_history) + 1} / {self.config.max_turns}",
    #         # f"Turns remaining: {turns_remaining}"
    #     ]
        
    #     # if self.state.move_history:
    #     #     moves = []
    #     #     for action, value in self.state.move_history[-5:]:
    #     #         if action == "COLOR":
    #     #             (x, y), (r, g, b) = value
    #     #             moves.append(f"{action}: ({x},{y}) ({r},{g},{b})")
    #     #         else:
    #     #             moves.append(action)
    #     #     state_parts.append(f"\nRecent moves:\n" + "\n".join(moves))
        
    #     return "\n".join(state_parts)

    def run(self, players: List[BaseLLMPlayer]) -> Dict[str, Any]:
        """Run the game with the LLM player."""
        if not players:
            raise ValueError("Need at least one player")
            
        player = players[0]  # Only use the first player
        turn = 0
        consecutive_errors = 0
        
        while turn < self.config.max_turns and not self.state.is_finished:
            # Get initial state for current turn
            state_message = {
                "role": "user",
                "content": f"Turn: {turn + 1} / {self.config.max_turns-1}"
            }
            
            try:
                # Get and process player's move
                response = player.get_response(state_message, self.create_grid_image())
                outcome = self.attempt_move(response)
                
                # Print the action result
                print(f"Turn: {turn} / {self.config.max_turns}")
                print(outcome["message"])
                
                if not outcome["valid"]:
                    consecutive_errors += 1
                    player.add_message({
                        "role": "system",
                        "content": outcome["message"]
                    })
                    
                    # End game if 3 consecutive errors
                    if consecutive_errors >= 3:
                        print("Game ended due to 3 consecutive invalid moves.")
                        break
                else:
                    # Reset consecutive errors on a valid move
                    consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Error during turn: {str(e)}")
                
                # End game if 3 consecutive errors
                if consecutive_errors >= 3:
                    print("Game ended due to 3 consecutive errors.")
                    break
            
            turn += 1
        
        return {
            "status": "complete",
            "moves": len(self.state.move_history),
            "move_history": self.state.move_history,
            "finished": self.state.is_finished or consecutive_errors >= 3
        }

    def get_final_position(self) -> str:
        """Get string representation of final game state."""
        return (
            f"Game {'completed' if self.state.is_finished else 'ended'} after {len(self.state.move_history)} moves\n\n"
            f"Move history:\n" +
            "\n".join(
                f"Turn {i+1}: {action}: {value}" if action == "COLOR" else f"Turn {i+1}: {action}"
                for i, (action, value) in enumerate(self.state.move_history)
            )
        )
