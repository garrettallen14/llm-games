from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import random
import json
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from PIL import Image, ImageDraw, ImageFont
import os
import base64
from io import BytesIO
import re

from base.llm_player import BaseLLMPlayer

class Move(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

@dataclass
class Game2048State:
    """Tracks the current state of the 2048 game"""
    grid: np.ndarray = field(default_factory=lambda: np.zeros((4, 4), dtype=int))
    score: int = 0
    moves: int = 0
    move_history: List[Dict] = field(default_factory=list)
    game_over: bool = False
    highest_tile: int = 0
    consecutive_invalid_moves: int = 0
    
    # Additional metrics
    total_invalid_moves: int = 0  # Total number of invalid moves
    total_merges: int = 0  # Number of successful merges
    empty_cells: int = field(default_factory=lambda: 16)  # Number of empty cells
    largest_merge: int = 0  # Largest merge performed
    moves_since_last_merge: int = 0  # Moves since last successful merge
    total_spawned: Dict[int, int] = field(default_factory=lambda: {2: 0, 4: 0})  # Count of spawned values

    def update_metrics(self, grid: np.ndarray):
        """Update metrics based on current grid state"""
        self.empty_cells = np.count_nonzero(grid == 0)
        self.highest_tile = int(np.max(grid))

@dataclass
class Game2048Config:
    """Configuration for 2048 game"""
    run_dir: Path
    max_turns: int = 1000000000  # Reasonable limit for max moves
    spawn_rates: Dict[int, float] = field(default_factory=lambda: {2: 0.9, 4: 0.1})  # 90% chance for 2, 10% for 4
    target_tile: int = 2048
    grid_size: int = 4
    max_invalid_moves: int = 5  # Maximum consecutive invalid moves before game over

class Game2048Game:
    """Implementation of the 2048 sliding tile game"""
    
    def __init__(self, run_dir: Path, max_turns: int = 1000000000):
        """Initialize the 2048 game"""
        self.config = Game2048Config(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.state = Game2048State()
        self.initialize_board()
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        
    def initialize_board(self):
        """Initialize the game board with two random tiles"""
        self.state.grid = np.zeros((4, 4), dtype=int)
        self.spawn_new_tile()
        self.spawn_new_tile()
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing 2048, the popular sliding tile game. Your goal is to create the highest value tiles possible by merging smaller ones.

Rules:
1. The game is played on a 4x4 grid
2. Each turn, you can slide all tiles UP, DOWN, LEFT, or RIGHT
3. When two tiles with the same number collide, they merge into one tile with their sum
4. After each valid move, a new tile appears in a random empty cell
   - 90% chance of a 2 tile
   - 10% chance of a 4 tile
5. Score increases by the value of each merge (merging two 4s adds 8 to score)
6. Game ends only when no valid moves are left (grid is full and no merges possible)

To make a move, respond with exactly one of these commands:
MOVE: UP
MOVE: DOWN
MOVE: LEFT
MOVE: RIGHT

Invalid moves (those that don't change the board state) are not allowed.

The game state shown after each move includes:
- Current grid layout
- Current score
- Number of moves made
- Available valid moves

Strategy tips:
- Keep your highest tiles in a corner
- Build a chain of decreasing values
- Plan several moves ahead
- Avoid scattering small tiles

Think carefully about each move to maximize your score!"""
        }
        
    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get the current game state in a tokenizer-friendly format"""
        # Create simple grid representation
        grid_str = ""
        for i in range(4):
            row = []
            for j in range(4):
                value = int(self.state.grid[i][j])
                row.append('_' if value == 0 else str(value))
            grid_str += ' '.join(row) + '\n'
        
        # List valid moves
        valid_moves = self.get_valid_moves()
        moves_str = " ".join([move.value for move in valid_moves]) if valid_moves else "GAME_OVER"
        
        state = (
            f"STATE\n"
            f"{grid_str}"
            f"SCORE {self.state.score}\n"
            f"MOVES {self.state.moves}\n"
            f"HIGHEST {self.state.highest_tile}\n"
            f"VALID {moves_str}"
        )
        
        return state

    def _get_highest_tile_position(self) -> str:
        """Get the position of the highest tile in coordinate notation"""
        max_val = np.max(self.state.grid)
        if max_val == 0:
            return "none"
        pos = np.where(self.state.grid == max_val)
        row, col = pos[0][0], pos[1][0]
        return f"{chr(65 + col)}{row + 1}"  # Convert to A1-D4 notation

    def spawn_new_tile(self) -> None:
        """Spawn a new tile (2 or 4) in a random empty cell"""
        empty_cells = list(zip(*np.where(self.state.grid == 0)))
        if empty_cells:
            i, j = random.choice(empty_cells)
            value = np.random.choice([2, 4], p=[self.config.spawn_rates[2], self.config.spawn_rates[4]])
            self.state.grid[i, j] = value
            self.state.total_spawned[value] += 1
            self.state.update_metrics(self.state.grid)

    def get_valid_moves(self) -> List[Move]:
        """Get list of valid moves in current position"""
        valid_moves = []
        for move in Move:
            # Create a temporary grid for testing the move
            temp_grid = self.state.grid.copy()
            score_gained, grid_changed = self._apply_move(temp_grid, move)
            # Only consider moves that actually change the grid
            if grid_changed:
                valid_moves.append(move)
        return valid_moves

    def _apply_move(self, grid: np.ndarray, move: Move) -> Tuple[int, bool]:
        """Apply move to grid, return (score_gained, grid_changed)"""
        original_grid = grid.copy()
        score_gained = 0
        merges = 0
        largest_merge_this_move = 0
        
        # Rotate grid to handle all moves as LEFT
        if move == Move.UP:
            grid = np.rot90(grid)
        elif move == Move.RIGHT:
            grid = np.rot90(grid, 2)
        elif move == Move.DOWN:
            grid = np.rot90(grid, 3)
            
        def process_row(row: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
            """Process a single row, return (new_row, score_gained, merges, largest_merge)"""
            # Remove all zeros first
            row = row[row != 0]
            if len(row) <= 1:
                return np.pad(row, (0, 4 - len(row))), 0, 0, 0
                
            new_row = []
            score = 0
            merges_in_row = 0
            largest_merge = 0
            i = 0
            
            while i < len(row):
                if i + 1 < len(row) and row[i] == row[i + 1]:
                    # Merge equal tiles
                    new_value = row[i] * 2
                    new_row.append(new_value)
                    score += new_value
                    merges_in_row += 1
                    largest_merge = max(largest_merge, new_value)
                    i += 2
                else:
                    # Keep single tile
                    new_row.append(row[i])
                    i += 1
            
            # Convert to numpy array and pad
            new_row = np.array(new_row)
            new_row = np.pad(new_row, (0, 4 - len(new_row)))
            
            return new_row, score, merges_in_row, largest_merge
            
        # Process each row
        for i in range(4):
            grid[i], row_score, row_merges, row_largest_merge = process_row(grid[i])
            score_gained += row_score
            merges += row_merges
            largest_merge_this_move = max(largest_merge_this_move, row_largest_merge)
            
        # Rotate back
        if move == Move.UP:
            grid = np.rot90(grid, 3)
        elif move == Move.RIGHT:
            grid = np.rot90(grid, 2)
        elif move == Move.DOWN:
            grid = np.rot90(grid)
            
        # Update metrics
        self.state.total_merges += merges
        if merges > 0:
            self.state.moves_since_last_merge = 0
            self.state.largest_merge = max(self.state.largest_merge, largest_merge_this_move)
        else:
            self.state.moves_since_last_merge += 1
            
        # Check if grid changed
        grid_changed = not np.array_equal(original_grid, grid)
        
        return score_gained, grid_changed

    def _parse_move(self, response: str) -> Optional[str]:
        """Extract move from response string, handling both direct commands and natural language
        
        Args:
            response: The full response string from the player
            
        Returns:
            Extracted move string or None if no valid move found
        """
        if not isinstance(response, str):
            return None
            
        # 1. Clean the response
        # Remove code blocks while keeping their content
        response = re.sub(r'```.*?```', lambda m: m.group(0).replace('```', ''), response, flags=re.DOTALL)
        # Remove markdown bold
        response = re.sub(r'\*\*.*?\*\*', '', response)
        # Normalize whitespace while preserving newlines
        response = '\n'.join(line.strip() for line in response.split('\n'))
            
        # 2. Find all potential moves
        valid_moves = [m.value for m in Move]
        moves = []
        
        # Process each line
        for line in response.split('\n'):
            # Look for MOVE: followed by a valid direction and optional context
            move_pattern = r'MOVE:\s*(' + '|'.join(valid_moves) + r')\b(?:\s+(?:because|to|is|as|since|will|for|and|#|[^A-Z])|[.,!?]|$)'
            matches = list(re.finditer(move_pattern, line, re.IGNORECASE))
            
            for match in matches:
                move = match.group(1).upper()
                # Get the rest of the line after this move
                after_text = line[match.end():].strip()
                # Skip if followed by another valid move
                if any(m in after_text.upper().split() for m in valid_moves):
                    continue
                moves.append(move)
                break
                
        # Return the first valid move found
        return moves[0] if moves else None

    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        try:
            move_str = self._parse_move(response)
            if not move_str:
                self.state.consecutive_invalid_moves += 1
                self.state.total_invalid_moves += 1
                return {
                    "valid": False,
                    "message": "ERROR: Could not find valid move in response. Move must contain 'MOVE: X' where X is UP, DOWN, LEFT, or RIGHT",
                    "end_turn": False,
                    "end_game": self.state.consecutive_invalid_moves >= self.config.max_invalid_moves
                }
            
            try:
                move = Move[move_str]
            except KeyError:
                self.state.consecutive_invalid_moves += 1
                self.state.total_invalid_moves += 1
                valid_moves = self.get_valid_moves()
                return {
                    "valid": False,
                    "message": f"ERROR: Invalid move direction! Must be UP, DOWN, LEFT, or RIGHT\nVALID MOVES: MOVE: {', '.join(m.value for m in valid_moves)}",
                    "end_turn": False,
                    "end_game": self.state.consecutive_invalid_moves >= self.config.max_invalid_moves
                }
                
            # Check if move is valid
            valid_moves = self.get_valid_moves()
            if move not in valid_moves:
                self.state.consecutive_invalid_moves += 1
                self.state.total_invalid_moves += 1
                return {
                    "valid": False,
                    "message": f"ERROR: Invalid move! This move wouldn't change the board state\nVALID MOVES: MOVE: {', '.join(m.value for m in valid_moves)}",
                    "end_turn": False,
                    "end_game": self.state.consecutive_invalid_moves >= self.config.max_invalid_moves
                }
                
            # Apply the move
            score_gained, _ = self._apply_move(self.state.grid, move)
            
            # Reset consecutive invalid moves counter on successful move
            self.state.consecutive_invalid_moves = 0
            
            self.state.score += score_gained
            self.state.moves += 1
            
            # Update highest tile and spawn new tile
            self.state.highest_tile = int(np.max(self.state.grid))
            self.spawn_new_tile()
            
            # Record move in history
            self.state.move_history.append({
                "move": move.value,
                "score_gained": score_gained,
                "total_score": self.state.score,
                "highest_tile": self.state.highest_tile
            })
            
            # Check if game is over (no valid moves left)
            game_over = not bool(self.get_valid_moves())
            if game_over:
                self.state.game_over = True
                self.end_time = datetime.now().isoformat()
            
            return {
                "valid": True,
                "message": f"Move accepted: {response}\n\n{self.get_current_state(player_id)}",
                "end_turn": True,
                "end_game": game_over,
                "skip_inference": False
            }
            
        except Exception as e:
            # Catch any unexpected errors
            self.state.consecutive_invalid_moves += 1
            self.state.total_invalid_moves += 1
            valid_moves = self.get_valid_moves()
            return {
                "valid": False,
                "message": f"ERROR: Invalid move direction! Must be UP, DOWN, LEFT, or RIGHT\nVALID MOVES: MOVE: {', '.join(m.value for m in valid_moves)}",
                "end_turn": False,
                "end_game": self.state.consecutive_invalid_moves >= self.config.max_invalid_moves
            }

    def get_game_image(self) -> Optional[str]:
        """Generate base64 encoded PNG of current board state"""
        try:
            # Constants for image generation
            CELL_SIZE = 100
            PADDING = 10
            GRID_SIZE = 4
            FONT_SIZE = 36
            
            # Calculate total size
            total_size = (CELL_SIZE * GRID_SIZE) + (PADDING * (GRID_SIZE + 1))
            
            # Create new image with white background
            img = Image.new('RGB', (total_size, total_size), 'white')
            draw = ImageDraw.Draw(img)
            
            # Color scheme for different tile values
            COLOR_SCHEME = {
                0: ('#CCC0B3', '#776E65'),  # Empty cell
                2: ('#EEE4DA', '#776E65'),
                4: ('#EDE0C8', '#776E65'),
                8: ('#F2B179', '#F9F6F2'),
                16: ('#F59563', '#F9F6F2'),
                32: ('#F67C5F', '#F9F6F2'),
                64: ('#F65E3B', '#F9F6F2'),
                128: ('#EDCF72', '#F9F6F2'),
                256: ('#EDCC61', '#F9F6F2'),
                512: ('#EDC850', '#F9F6F2'),
                1024: ('#EDC53F', '#F9F6F2'),
                2048: ('#EDC22E', '#F9F6F2'),
                4096: ('#EDC11D', '#F9F6F2'),  # Extended colors for higher values
                8192: ('#EDC00C', '#F9F6F2'),
                16384: ('#EDBE00', '#F9F6F2'),
                32768: ('#EDBD00', '#F9F6F2'),
                65536: ('#EDBC00', '#F9F6F2'),
            }
            
            # Get tile color, default to a deep gold color for any value beyond defined scheme
            def get_tile_colors(value):
                if value in COLOR_SCHEME:
                    return COLOR_SCHEME[value]
                return ('#EDBA00', '#F9F6F2')  # Default gold color for very high values
            
            # Try to load font, fall back to default if not found
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
            except:
                font = ImageFont.load_default()
            
            # Draw each cell
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    # Calculate cell position
                    x = j * (CELL_SIZE + PADDING) + PADDING
                    y = i * (CELL_SIZE + PADDING) + PADDING
                    
                    # Get cell value and colors
                    value = int(self.state.grid[i][j])
                    bg_color, text_color = get_tile_colors(value)
                    
                    # Draw cell background
                    draw.rectangle([x, y, x + CELL_SIZE, y + CELL_SIZE], fill=bg_color)
                    
                    # Draw value if cell is not empty
                    if value != 0:
                        # Get text size for centering
                        text = str(value)
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # Calculate text position for centering
                        text_x = x + (CELL_SIZE - text_width) // 2
                        text_y = y + (CELL_SIZE - text_height) // 2
                        
                        # Draw text
                        draw.text(
                            (text_x, text_y),
                            text,
                            fill=text_color,
                            font=font
                        )
            
            # Save the image to the run directory
            image_path = "board.png"
            img.save(image_path)
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            print(f"Warning: Failed to generate board image: {e}")
            return None

    def get_game_result(self) -> Dict:
        """Get the final game result"""
        self.end_time = datetime.now().isoformat()
        self.state.game_over = True
        
        result = {
            "status": "completed" if not self.state.consecutive_invalid_moves >= self.config.max_invalid_moves else "error",
            "moves": self.state.moves,
            "score": self.state.score,
            "highest_tile": self.state.highest_tile,
            "high_score": self.state.score,  # For now, game score is high score
            "message": self.get_final_position(),
            "history": self.state.move_history,
            
            # Additional metrics
            "total_invalid_moves": self.state.total_invalid_moves,
            "consecutive_invalid_moves": self.state.consecutive_invalid_moves,
            "total_merges": self.state.total_merges,
            "largest_merge": self.state.largest_merge,
            "empty_cells": self.state.empty_cells,
            "moves_since_last_merge": self.state.moves_since_last_merge,
            "total_spawned": self.state.total_spawned,
            "duration": (datetime.fromisoformat(self.end_time) - datetime.fromisoformat(self.start_time)).total_seconds(),
            "avg_score_per_move": self.state.score / max(1, self.state.moves),
            "merge_rate": self.state.total_merges / max(1, self.state.moves)
        }
        
        if self.state.consecutive_invalid_moves >= self.config.max_invalid_moves:
            result["message"] = f"Game Over - {self.config.max_invalid_moves} consecutive invalid moves"
            
        return result
        
    def get_final_position(self) -> str:
        """Get string representation of final game state"""
        if self.state.consecutive_invalid_moves >= self.config.max_invalid_moves:
            return f"Game terminated due to {self.config.max_invalid_moves} consecutive invalid moves"
            
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return "Game Over - No valid moves remaining"
            
        if self.state.moves >= self.config.max_turns:
            return f"Game Over - Maximum turns ({self.config.max_turns}) reached"
            
        return "Game completed successfully"

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the game with the provided players"""
        if not players:
            return {"status": "error", "message": "No players provided"}
            
        current_player = players[0]  # Single player game
        print(f"\nStarting 2048 game...\n")
        
        while not self.state.game_over and self.state.moves < self.config.max_turns:
            try:
                print(f"\nMove {self.state.moves + 1}")
                print("=" * 30)
                
                # Get game state and image
                state_message = {
                    "role": "user",
                    "content": self.get_current_state(1)  # Always player 1
                }
                
                # Generate and verify image
                game_image = self.get_game_image()
                if not game_image:
                    print("Warning: Failed to generate game image")
                
                # Get move from player with visual state
                response = current_player.get_response(state_message, game_image)
                print(f"Player response: {response}")
                
                outcome = self.attempt_move(response, 1)
                
                if not outcome["valid"]:
                    error_message = {
                        "role": "assistant",
                        "content": outcome["message"]
                    }
                    current_player.add_message(error_message)
                    print(f"Invalid move: {outcome['message']}")
                    
                    if outcome["end_game"]:
                        print(f"Game Over - {self.config.max_invalid_moves} consecutive invalid moves")
                        self.state.game_over = True
                        return self.get_game_result()
                    continue
                    
                if outcome["end_game"] or not bool(self.get_valid_moves()):
                    self.state.game_over = True
                    return self.get_game_result()
                    
            except Exception as e:
                print(f"Error during turn: {str(e)}")
                error_message = {
                    "role": "system",
                    "content": f"Error during turn: {str(e)}"
                }
                current_player.add_message(error_message)
                continue
                
        # If we exit the loop due to max turns
        self.state.game_over = True
        return self.get_game_result()
