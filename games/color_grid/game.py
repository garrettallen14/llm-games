"""ColorGrid game implementation with LLM players."""

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
class ColorGridState:
    """Tracks the current state of the ColorGrid game.
    
    Attributes:
        grid_size: Size of the nxn grid
        positions: Dictionary mapping player IDs to their (x,y) positions
        colors: Dictionary mapping player IDs to their (r,g,b) colors
        current_player: Current player's ID
        move_history: List of moves in format (player_id, action_type, value)
        consecutive_failures: Count of consecutive invalid moves
        max_failures: Maximum allowed invalid moves before forfeit
    """
    grid_size: int = 8
    positions: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    colors: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)
    current_player: int = 1
    move_history: List[Tuple[int, str, Any]] = field(default_factory=list)
    consecutive_failures: int = 0
    max_failures: int = 5

    def initialize_players(self, num_players: int):
        """Initialize positions and colors for n players"""
        # Calculate starting positions for n players
        corners = [(0, 0), (self.grid_size-1, self.grid_size-1),
                  (0, self.grid_size-1), (self.grid_size-1, 0)]
        
        for i in range(num_players):
            player_id = i + 1
            # Use corners first, then other positions if needed
            pos = corners[i] if i < len(corners) else (i % self.grid_size, i // self.grid_size)
            self.positions[player_id] = pos
            self.colors[player_id] = (255, 255, 255)  # Start with white

@dataclass
class ColorGridConfig:
    """Configuration settings for ColorGrid game.
    
    Attributes:
        run_dir: Directory for game artifacts
        max_turns: Maximum allowed game turns
        grid_size: Size of the grid (n x n)
        cell_size: Pixel size of each grid cell
    """
    run_dir: Path
    max_turns: int = 100
    grid_size: int = 8
    cell_size: int = 50

class ColorGridGame:
    """ColorGrid game manager handling game flow and state.
    
    A simple grid-based game where players can move and change colors.
    Players take turns either moving to adjacent cells or changing their color.
    """
    
    # Command validation patterns
    MOVE_PATTERN = re.compile(r"MOVE:\s*\((\d+)\s*,\s*(\d+)\)")
    COLOR_PATTERN = re.compile(r"COLOR:\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")
    SAY_PATTERN = re.compile(r"SAY:\s*(.+)")

    def __init__(self, run_dir: Path, max_turns: int = 100, grid_size: int = 10, num_players: int = 2) -> None:
        """Initialize ColorGrid game with configuration."""
        self.config = ColorGridConfig(
            run_dir=run_dir,
            max_turns=max_turns,
            grid_size=grid_size
        )
        self.state = ColorGridState(grid_size=grid_size)
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None

    @staticmethod
    def get_system_prompt() -> Dict[str, str]:
        """Return the system prompt for LLM players."""
        return {
            "role": "system",
            "content": """You are playing ColorGrid, a collaborative drawing game.

You can submit these actions in the same turn:
MOVE: (x,y)   - Move to any empty cell on the grid
COLOR: (r,g,b) - Change your color (values 0-255)
SAY: message  - Send a message to all players

Rules:
- You can move anywhere on the grid as long as the cell is empty
- Color values must be between 0-255
- You can MOVE and COLOR in the same turn
- Invalid moves will count as failures
- 5 consecutive failures will forfeit the game

Goal:
- Coordinate with the other players to draw a Smiley Face!
"""
        }

    def create_grid_image(self) -> str:
        """Generate base64 encoded PNG of current grid state."""
        # Calculate image dimensions
        width = self.config.grid_size * self.config.cell_size
        height = width
        
        # Create image and drawing context
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw grid lines
        for i in range(self.config.grid_size + 1):
            line_pos = i * self.config.cell_size
            # Vertical lines
            draw.line([(line_pos, 0), (line_pos, height)], fill='black')
            # Horizontal lines
            draw.line([(0, line_pos), (width, line_pos)], fill='black')
        
        # Draw players as squares
        for player_id, pos in self.state.positions.items():
            x, y = pos
            color = self.state.colors[player_id]
            
            # Calculate square position
            square_x = x * self.config.cell_size
            square_y = y * self.config.cell_size
            
            # Draw filled square for player
            draw.rectangle(
                [(square_x, square_y),
                 (square_x + self.config.cell_size, square_y + self.config.cell_size)],
                fill=color
            )
        
        # Convert to PNG
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image.save("board.png", format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def get_valid_moves(self, player_id: int) -> List[Tuple[int, int]]:
        """Get list of valid move positions for a player.

        Args:
            player_id: ID of player to get moves for

        Returns:
            List of valid (x,y) positions
        """
        current_pos = self.state.positions[player_id]
        valid_moves = []
        
        # Check all positions on the grid
        for x in range(self.config.grid_size):
            for y in range(self.config.grid_size):
                # Check if position is occupied
                if (x, y) not in self.state.positions.values():
                    valid_moves.append((x, y))
        
        return valid_moves

    def validate_move_command(self, x: int, y: int, player_id: int) -> Tuple[bool, Optional[str]]:
        """Validate a move command."""
        # Check bounds
        if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
            return False, f"Position ({x},{y}) is out of bounds"
        
        # Check if position is occupied
        if (x, y) in self.state.positions.values():
            return False, f"Position ({x},{y}) is already occupied"
        
        return True, None

    def validate_color_command(self, r: int, g: int, b: int) -> Tuple[bool, Optional[str]]:
        """Validate a color command."""
        for val, name in [(r, 'Red'), (g, 'Green'), (b, 'Blue')]:
            if not (0 <= val <= 255):
                return False, f"{name} value must be between 0 and 255"
        return True, None

    def attempt_move(self, response: str, player_id: int, players: List[BaseLLMPlayer]) -> Dict[str, Any]:
        """Process and execute a player's move attempt."""
        if player_id != self.state.current_player:
            return {"valid": False, "end_turn": True}
        
        move_executed = False
        color_executed = False
        error_message = ""
        
        # Try to match move command
        move_match = self.MOVE_PATTERN.search(response)
        if move_match:
            try:
                x, y = map(int, move_match.groups())
                is_valid, error = self.validate_move_command(x, y, player_id)
                
                if not is_valid:
                    error_message += error + "\n"
                else:
                    # Execute move
                    self.state.positions[player_id] = (x, y)
                    self.state.move_history.append((player_id, "MOVE", (x, y)))
                    move_executed = True
                    
            except ValueError:
                error_message += "Invalid move format\n"
        
        # Try to match color command
        color_match = self.COLOR_PATTERN.search(response)
        if color_match:
            try:
                r, g, b = map(int, color_match.groups())
                is_valid, error = self.validate_color_command(r, g, b)
                
                if not is_valid:
                    error_message += error + "\n"
                else:
                    # Execute color change
                    self.state.colors[player_id] = (r, g, b)
                    self.state.move_history.append((player_id, "COLOR", (r, g, b)))
                    color_executed = True
                    
            except ValueError:
                error_message += "Invalid color format\n"

        # Try to match say command
        say_match = self.SAY_PATTERN.search(response)
        if say_match:
            message = say_match.group(1).strip()
            # Broadcast message to all players
            for player in players:
                player.add_message({
                    "role": "user",
                    "content": f"Player {player_id} says: {message}"
                })
            self.state.move_history.append((player_id, "SAY", message))
            move_executed = True  # Count SAY as a valid action
        
        # If any action was successful, consider it a valid turn
        if move_executed or color_executed:
            return {
                "valid": True,
                "message": "Turn completed successfully",
                "end_turn": True
            }
        
        return {
            "valid": False,
            "message": error_message or "Invalid command format. Use MOVE: (x,y) and/or COLOR: (r,g,b) and/or SAY: message",
            "end_turn": False
        }

    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Generate formatted string of current game state."""
        state_parts = []
        
        if player_id:
            pos = self.state.positions[player_id]
            color = self.state.colors[player_id]
            state_parts.extend([
                f"You are Player {player_id}",
                f"Your position: ({pos[0]}, {pos[1]})",
                f"Your color: ({color[0]}, {color[1]}, {color[2]})"
            ])
        
        state_parts.extend([
            f"\nTurn: {len(self.state.move_history) + 1}",
            f"Current player: {self.state.current_player}"
        ])
        
        if self.state.move_history:
            moves = []
            for p_id, action, value in self.state.move_history[-5:]:
                moves.append(f"Player {p_id} - {action}: {value}")
            state_parts.append(f"\nRecent moves:\n" + "\n".join(moves))
        
        return "\n".join(state_parts)

    def print_player_action(self, player_id: int, response: str):
        """Print player actions in a clean format."""
        move_match = self.MOVE_PATTERN.search(response)
        color_match = self.COLOR_PATTERN.search(response)
        say_match = self.SAY_PATTERN.search(response)
        
        print(f"Player {player_id}:", end=" ")
        if move_match:
            x, y = map(int, move_match.groups())
            print(f"MOVE({x},{y})", end=" ")
        if color_match:
            r, g, b = map(int, color_match.groups())
            print(f"COLOR({r},{g},{b})", end=" ")
        if say_match:
            message = say_match.group(1).strip()
            print(f"SAY: {message}", end="")
        print()  # New line

    def run(self, players: List[BaseLLMPlayer]) -> Dict[str, Any]:
        """Run the game with provided players."""
        turn = 0
        num_players = len(players)
        
        # Initialize all players in state
        self.state.initialize_players(num_players)
        
        while turn < self.config.max_turns:
            # Calculate whose turn it is (cycles through all players)
            player_idx = turn % num_players
            current_player = players[player_idx]
            self.state.current_player = current_player.player_id
            
            # Get initial state for current player's turn
            state_message = {
                "role": "user",
                "content": self.get_current_state(self.state.current_player)
            }
            
            try:
                # Get and process player's move
                response = current_player.get_response(state_message, self.create_grid_image())
                outcome = self.attempt_move(response, self.state.current_player, players)
                
                if outcome["valid"]:
                    self.print_player_action(self.state.current_player, response)
                else:
                    # Handle invalid move
                    current_player.add_message({
                        "role": "system",
                        "content": outcome["message"]
                    })
                
            except Exception as e:
                print(f"Error during turn: {str(e)}")
                continue
            
            turn += 1
        
        return {
            "status": "complete",
            "moves": len(self.state.move_history),
            "move_history": self.state.move_history
        }

    def get_final_position(self) -> str:
        """Get string representation of final game state.

        Returns:
            Formatted final game state string
        """
        return (
            f"Game completed after {len(self.state.move_history)} moves\n"
            f"Final positions:\n" +
            "\n".join(
                f"Player {pid}: position={pos}, color={self.state.colors[pid]}"
                for pid, pos in self.state.positions.items()
            ) +
            f"\n\nMove history:\n" +
            "\n".join(
                f"Turn {i+1}: Player {p_id} - {action}: {value}"
                for i, (p_id, action, value) in enumerate(self.state.move_history)
            )
        )