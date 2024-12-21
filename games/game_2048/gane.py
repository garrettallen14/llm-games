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

Moves that don't change the board state are not allowed.
You may think before submitting the move."""
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
        return f"{chr(65 + col)}{row + 1}"

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
