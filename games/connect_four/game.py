# games/connect_four/game.py

import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import base64
import io
from PIL import Image, ImageDraw
import re
from dataclasses import dataclass, field

from base.llm_player import BaseLLMPlayer

@dataclass
class ConnectFourState:
    """Tracks the current state of the Connect Four game"""
    board: np.ndarray = field(default_factory=lambda: np.zeros((6, 7), dtype=int))
    current_player: int = 1  # 1 for first player (red), 2 for second player (yellow)
    move_history: List[int] = field(default_factory=list)

@dataclass
class ConnectFourConfig:
    """Configuration for Connect Four game"""
    run_dir: Path
    max_turns: int = 100
    rows: int = 6
    cols: int = 7
    cell_size: int = 60
    padding: int = 10

class ConnectFourGame:
    def __init__(self, run_dir: Path, max_turns: int = 100):
        self.config = ConnectFourConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.state = ConnectFourState()
        
        # Move validation pattern
        self.move_pattern = re.compile(r"MOVE:\s*([1-7])")
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing Connect Four. The board has 7 columns (numbered 1-7 from left to right).
                Submit your move in the format 'MOVE: X' where X is the column number (1-7).
                
                Rules:
                - You must only make legal moves (columns that aren't full)
                - Play to win by connecting four of your pieces horizontally, vertically, or diagonally
                - Player 1 uses red pieces (R), Player 2 uses yellow pieces (Y)
                - Empty spaces are marked with dots (.)
                
                Before each move:
                1. Analyze the current board state carefully
                2. Consider your opponent's threats
                3. Look for winning opportunities
                4. Choose the best column for your move
                
                Then respond in the EXACT format 'MOVE: X' where X is a column number 1-7."""
        }
        
    def _create_board_image(self) -> Image.Image:
        """Create a PIL Image of the current board state"""
        width = self.config.cols * self.config.cell_size + 2 * self.config.padding
        height = self.config.rows * self.config.cell_size + 2 * self.config.padding
        
        # Create image with white background
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Draw blue board background
        draw.rectangle(
            [(self.config.padding, self.config.padding), 
             (width - self.config.padding, height - self.config.padding)],
            fill='blue'
        )
        
        # Draw cells
        for row in range(self.config.rows):
            for col in range(self.config.cols):
                x = self.config.padding + col * self.config.cell_size
                y = self.config.padding + row * self.config.cell_size
                
                # Determine piece color
                if self.state.board[row, col] == 0:
                    color = 'white'
                elif self.state.board[row, col] == 1:
                    color = 'red'
                else:
                    color = 'yellow'
                
                # Draw circle
                draw.ellipse(
                    [x + 5, y + 5, x + self.config.cell_size - 5, y + self.config.cell_size - 5],
                    fill=color
                )
        
        # Draw column numbers
        for col in range(self.config.cols):
            x = self.config.padding + col * self.config.cell_size + self.config.cell_size // 2
            draw.text(
                (x, 2),
                str(col + 1),
                fill='black',
                anchor='mt'
            )
        
        return image
        
    def get_game_image(self) -> Optional[str]:
        """Generate base64 encoded PNG of current board state"""
        try:
            image = self._create_board_image()
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image.save("board.png")
            # image.save(self.config.run_dir / "board.png")
            return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        except Exception as e:
            print(f"Warning: Failed to generate board image: {e}")
            return None
            
    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get current game state from a player's perspective"""
        # Convert board to string representation
        board_str = ""
        for row in range(self.config.rows):
            for col in range(self.config.cols):
                if self.state.board[row, col] == 0:
                    board_str += ". "
                elif self.state.board[row, col] == 1:
                    board_str += "R "
                else:
                    board_str += "Y "
            board_str += "\n"
            
        # Add column numbers
        board_str += "1 2 3 4 5 6 7\n"
        
        # Build state message
        state = (
            f"Current board:\n{board_str}\n"
            f"Player {self.state.current_player} to move "
            f"({'Red' if self.state.current_player == 1 else 'Yellow'} pieces)\n"
            f"Legal moves (columns): {', '.join(map(str, self._get_legal_moves()))}"
        )
        
        if self.state.move_history:
            state += f"\nMove history: {', '.join(map(str, self.state.move_history))}"
            
        # Add player-specific information
        if player_id:
            piece_color = "Red" if player_id == 1 else "Yellow"
            state += f"\nYou are playing {piece_color} pieces"
            
        return state
        
    def _get_legal_moves(self) -> List[int]:
        """Get list of columns that aren't full"""
        return [col + 1 for col in range(self.config.cols) if self.state.board[0, col] == 0]
        
    def _check_winner(self) -> bool:
        """Check if the current player has won"""
        # Convert current player number to check for
        player = self.state.current_player
        
        # Check horizontal
        for row in range(self.config.rows):
            for col in range(self.config.cols - 3):
                if np.all(self.state.board[row, col:col + 4] == player):
                    return True
        
        # Check vertical
        for row in range(self.config.rows - 3):
            for col in range(self.config.cols):
                if np.all(self.state.board[row:row + 4, col] == player):
                    return True
        
        # Check diagonal (positive slope)
        for row in range(self.config.rows - 3):
            for col in range(self.config.cols - 3):
                if np.all(np.array([self.state.board[row + i, col + i] for i in range(4)]) == player):
                    return True
        
        # Check diagonal (negative slope)
        for row in range(3, self.config.rows):
            for col in range(self.config.cols - 3):
                if np.all(np.array([self.state.board[row - i, col + i] for i in range(4)]) == player):
                    return True
        
        return False
        
    def _is_board_full(self) -> bool:
        """Check if the board is full (draw)"""
        return np.all(self.state.board != 0)
        
    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        print(f"Player {player_id} move: {response}")
        # Verify it's the player's turn
        if player_id != self.state.current_player:
            return {
                "valid": True,
                "message": f"Not your turn. Waiting for Player {self.state.current_player}",
                "end_turn": True,
                "end_game": False,
                "skip_inference": True
            }
            
        # Parse move
        match = self.move_pattern.search(response)
        if not match:
            return {
                "valid": False,
                "message": "Error: Move must be in format 'MOVE: X' where X is a column number 1-7",
                "end_turn": False
            }
            
        try:
            column = int(match.group(1)) - 1  # Convert to 0-based indexing
            
            # Check if column is full
            if self.state.board[0, column] != 0:
                return {
                    "valid": False,
                    "message": f"Column {column + 1} is full! Legal moves: {', '.join(map(str, self._get_legal_moves()))}",
                    "end_turn": False
                }
                
            # Find the lowest empty row in the chosen column
            for row in range(self.config.rows - 1, -1, -1):
                if self.state.board[row, column] == 0:
                    self.state.board[row, column] = self.state.current_player
                    break
                    
            # Record move
            self.state.move_history.append(column + 1)
            
            # Check game end conditions
            game_over = False
            if self._check_winner():
                self.end_time = datetime.now().isoformat()
                game_over = True
            elif self._is_board_full():
                self.end_time = datetime.now().isoformat()
                game_over = True
                
            # Switch players
            self.state.current_player = 3 - self.state.current_player
            
            return {
                "valid": True,
                "message": f"Move to column {column + 1} played.\n\n{self.get_current_state(player_id)}",
                "end_turn": True,
                "end_game": game_over,
                "skip_inference": False
            }
            
        except ValueError:
            return {
                "valid": False,
                "message": "Invalid move format. Please use a number between 1 and 7.",
                "end_turn": False
            }
            
    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the game with the provided players"""
        turn = 0
        while turn < self.config.max_turns:
            current_player = players[self.state.current_player - 1]
            
            # Get game state
            state_message = {
                "role": "user",
                "content": self.get_current_state(self.state.current_player)
            }
            
            # Get player response
            try:
                response = current_player.get_response(state_message, self.get_game_image())
                outcome = self.attempt_move(response, self.state.current_player)
                
                if not outcome["valid"]:
                    continue
                    
                # Notify other player
                other_player_id = 3 - self.state.current_player
                other_player = players[other_player_id - 1]
                other_state = {
                    "role": "user",
                    "content": self.get_current_state(other_player_id)
                }
                
                if not outcome["skip_inference"]:
                    other_player.get_response(other_state, self.get_game_image())
                    
                if outcome["end_game"]:
                    return self.get_game_result()
                    
            except Exception as e:
                print(f"Error during turn: {str(e)}")
                continue
                
            turn += 1
            
        return self.get_game_result()
        
    def get_game_result(self) -> Dict[str, str]:
        """Get the current game result"""
        if self._check_winner():
            # Note: current_player has already been switched, so we check the opposite
            winner = "Red" if self.state.current_player == 2 else "Yellow"
            player_num = 1 if winner == "Red" else 2
            return {
                "status": "complete",
                "winner": winner,
                "player": player_num,
                "final_position": self.get_final_position(),
                "move_history": self.state.move_history
            }
        elif self._is_board_full():
            return {
                "status": "draw",
                "reason": "board_full",
                "final_position": self.get_final_position(),
                "move_history": self.state.move_history
            }
        else:
            return {
                "status": "in_progress",
                "move_history": self.state.move_history
            }
            
    def get_final_position(self) -> str:
        """Get string representation of final board state"""
        board_str = ""
        for row in range(self.config.rows):
            for col in range(self.config.cols):
                if self.state.board[row, col] == 0:
                    board_str += ". "
                elif self.state.board[row, col] == 1:
                    board_str += "R "
                else:
                    board_str += "Y "
            board_str += "\n"
        return board_str