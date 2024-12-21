"""Chess game implementation with LLM players."""

import chess
import chess.svg
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from datetime import datetime
import re
import base64
import cairosvg
from dataclasses import dataclass, field

from base.llm_player import BaseLLMPlayer

@dataclass
class ChessState:
    """Tracks the current state of the Chess game.
    
    Attributes:
        board: Current chess board state
        move_history: List of moves in UCI format
        current_player: Current player (1=White, 2=Black)
        consecutive_failures: Count of consecutive invalid moves
        max_failures: Maximum allowed invalid moves before forfeit
    """
    board: chess.Board = field(default_factory=chess.Board)
    move_history: List[str] = field(default_factory=list)
    current_player: int = 1
    consecutive_failures: int = 0
    max_failures: int = 5

@dataclass
class ChessConfig:
    """Configuration settings for Chess game.
    
    Attributes:
        run_dir: Directory for game artifacts
        max_turns: Maximum allowed game turns
        board_size: Visual board size in pixels
    """
    run_dir: Path
    max_turns: int = 100
    board_size: int = 400

class ChessGame:
    """Chess game manager handling game flow and state.
    
    Attributes:
        config: Game configuration settings
        state: Current game state
        start_time: Game start timestamp
        end_time: Game end timestamp
    """
    
    # Move validation pattern in UCI format - now handles promotions
    MOVE_PATTERN = re.compile(r"MOVE:\s*([a-h][1-8][a-h][1-8][qrbn]?)")
    
    def __init__(self, run_dir: Path, max_turns: int = 100) -> None:
        """Initialize chess game with configuration.

        Args:
            run_dir: Directory for game artifacts
            max_turns: Maximum allowed game turns
        """
        self.config = ChessConfig(run_dir=run_dir, max_turns=max_turns)
        self.state = ChessState()
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None

    @staticmethod
    def get_system_prompt() -> Dict[str, str]:
        """Return the system prompt for LLM players."""
        return {
            "role": "system",
            "content": """You are playing chess. Your goal is to win the game through skillful play.

To make a move, you must respond in this exact format:
MOVE: <square_from><square_to>

The move format uses UCI (Universal Chess Interface) notation:
- Each square is specified by its file (a-h) and rank (1-8)
- First two characters are the starting square
- Last two characters are the destination square
- No spaces or extra characters in the move itself

After each move:
- You will see the current board position
- You will be given a list of all legal moves
- You must choose one of these legal moves
- Respond with ONLY the move in the specified format

Important:
- You must ONLY make legal moves from the provided list
- Think carefully about each move
- Play to win, but always follow the rules
- Moves must be exact - any deviation will be rejected
- If you fail 5 times in a row, you will forfeit the game
"""
        }

    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Generate formatted string of current game state.

        Args:
            player_id: Optional ID of player requesting state

        Returns:
            Formatted game state string
        """
        fen = self.state.board.fen()
        turn = "White" if self.state.board.turn else "Black"
        legal_moves = [move.uci() for move in self.state.board.legal_moves]
        
        state_parts = [
            f"Current position (FEN): {fen}",
            f"ASCII Board:\n{self.state.board}",
            f"It is {turn} to move.",
            f"Legal moves: {', '.join(legal_moves)}"
        ]
        
        if self.state.move_history:
            state_parts.append(f"Move history: {', '.join(self.state.move_history)}")
            
        if player_id:
            player_color = "White" if player_id == 1 else "Black"
            state_parts.append(f"You are playing {player_color}")
            
        return "\n".join(state_parts)

    def get_game_image(self) -> Optional[str]:
        """Generate base64 encoded PNG of current board state.

        Returns:
            Base64 encoded PNG data URI or None if generation fails
        """
        try:
            svg_content = chess.svg.board(
                board=self.state.board,
                size=self.config.board_size,
                coordinates=True,
                lastmove=self.state.board.peek() if self.state.board.move_stack else None,
                flipped=not self.state.board.turn
            )
            
            png_data = cairosvg.svg2png(bytestring=svg_content)
            with open("board.png", "wb") as f:
                f.write(png_data)
            
            return f"data:image/png;base64,{base64.b64encode(png_data).decode()}"
            
        except Exception as e:
            print(f"Warning: Failed to generate board image: {e}")
            return None

    def validate_response(self, response: str, player_id: int) -> Dict[str, Any]:
        """Validate player's move response.

        Args:
            response: Player's move response
            player_id: ID of player making move

        Returns:
            Validation result dictionary
        """

        # Verify player turn
        if player_id != self.state.current_player:
            return {
                "valid": False,
                "end_turn": True
            }
        
        # Validate move format
        match = self.MOVE_PATTERN.search(response)
        if not match:
            return {
                "valid": False,
                "message": "Error: Move must be in format 'MOVE: <move>'",
                "end_turn": False
            }
        
        return {
            "valid": True,
            "move": match.group(1)
        }

    def attempt_move(self, response: str, player_id: int) -> Dict[str, Any]:
        """Process and execute a player's move attempt.

        Args:
            response: Player's move response
            player_id: ID of player making move

        Returns:
            Move attempt result dictionary
        """
        print(f"Player {player_id} move: {response}")
        
        validation = self.validate_response(response, player_id)
        if not validation["valid"]:
            return validation
        
        move_str = validation["move"]
        try:
            move = chess.Move.from_uci(move_str)
            if move not in self.state.board.legal_moves:
                legal_moves = [m.uci() for m in self.state.board.legal_moves]
                return {
                    "valid": False,
                    "message": f"Illegal move: {move_str}! You MUST select one of the legal moves: {', '.join(legal_moves)}",
                    "end_turn": False
                }
            
            # Execute move
            self.state.board.push(move)
            self.state.move_history.append(move_str)
            self.state.current_player = 3 - self.state.current_player
            
            # Check game end
            game_over = self.state.board.is_game_over()
            if game_over:
                self.end_time = datetime.now().isoformat()
            
            return {
                "valid": True,
                "message": f"Move {move_str} played.\n\n{self.get_current_state(player_id)}",
                "end_turn": True,
                "end_game": game_over,
                "skip_inference": False
            }
            
        except ValueError:
            return {
                "valid": False,
                "message": f"Invalid move format: {move_str}",
                "end_turn": False
            }

    def run(self, players: List[BaseLLMPlayer]) -> Dict[str, Any]:
        """Run the chess game with provided players.

        Args:
            players: List of LLM players

        Returns:
            Game result dictionary
        """
        turn = 0
        while turn < self.config.max_turns:
            current_player = players[self.state.current_player - 1]
            
            # Get initial state for current player's turn
            state_message = {
                "role": "user",
                "content": self.get_current_state(self.state.current_player)
            }
            
            try:
                # Keep trying until we get a valid move
                while True:
                    response = current_player.get_response(state_message, self.get_game_image())
                    outcome = self.attempt_move(response, self.state.current_player)
                    
                    if outcome["valid"]:
                        self.state.consecutive_failures = 0
                        break
                    
                    # Handle invalid move
                    self.state.consecutive_failures += 1
                    current_player.add_message({
                        "role": "system",
                        "content": outcome["message"]
                    })
                    
                    # Check for too many failures
                    if self.state.consecutive_failures >= self.state.max_failures:
                        return {
                            "status": "forfeit",
                            "winner": f"Player {3 - self.state.current_player}",
                            "player": 3 - self.state.current_player,
                            "reason": f"Player {self.state.current_player} forfeited after {self.state.max_failures} consecutive invalid moves",
                            "final_position": self.state.board.fen(),
                            "move_history": self.state.move_history
                        }
                
                # Check if game is over after the move
                if outcome["end_game"]:
                    return self.get_game_result()
                    
            except Exception as e:
                print(f"Error during turn: {str(e)}")
                continue
            
            turn += 1
        
        return self.get_game_result()

    def get_game_result(self) -> Dict[str, Any]:
        """Get the current game result.

        Returns:
            Game result dictionary
        """
        if not self.state.board.is_game_over():
            return {
                "status": "in_progress",
                "move_history": self.state.move_history
            }
        
        outcome = self.state.board.outcome()
        
        if outcome.winner is not None:
            winner = 1 if outcome.winner else 2
            return {
                "status": "complete",
                "winner": f"Player {winner}",
                "player": winner,
                "reason": "checkmate",
                "final_position": self.state.board.fen(),
                "move_history": self.state.move_history
            }
        
        return {
            "status": "draw",
            "reason": outcome.termination.name.lower(),
            "final_position": self.state.board.fen(),
            "move_history": self.state.move_history
        }

    def get_final_position(self) -> str:
        """Get string representation of final game state.

        Returns:
            Formatted final game state string
        """
        return (
            f"Final position (FEN): {self.state.board.fen()}\n"
            f"Move history: {', '.join(self.state.move_history)}\n"
            f"Final board:\n{self.state.board}"
        )