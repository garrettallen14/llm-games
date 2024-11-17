# games/chess/game.py

import chess
import chess.svg
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
import re
import base64
import cairosvg
from dataclasses import dataclass, field

from base.llm_player import BaseLLMPlayer

@dataclass
class ChessState:
    """Tracks the current state of the Chess game"""
    board: chess.Board = field(default_factory=chess.Board)
    move_history: List[str] = field(default_factory=list)
    current_player: int = 1  # 1 for White, 2 for Black

@dataclass
class ChessConfig:
    """Configuration for Chess game"""
    run_dir: Path
    max_turns: int = 100
    board_size: int = 400

class ChessGame:
    def __init__(self, run_dir: Path, max_turns: int = 100):
        self.config = ChessConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        self.state = ChessState()
        
        # Move validation pattern
        self.move_pattern = re.compile(r"MOVE:\s*([a-h][1-8][a-h][1-8])")
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing chess. Submit moves in format 'MOVE: X' where X is a move in UCI format.
                UCI format is a four-character string where the first two characters represent the square 
                the piece is moving from and the last two characters represent the destination square.
                
                Example moves:
                - 'MOVE: e2e4' (King's pawn opening)
                - 'MOVE: g8f6' (Knight to f6)
                
                You must only make legal moves. You are playing to win. After each move,
                I will show you the current board state and list of legal moves.
                Think carefully to find the best move, then respond in the EXACT format 'MOVE: X'.
                
                Player 1 plays White, Player 2 plays Black."""
        }
        
    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get the current game state from a player's perspective"""
        fen = self.state.board.fen()
        turn = "White" if self.state.board.turn else "Black"
        legal_moves = [move.uci() for move in self.state.board.legal_moves]
        
        state = (
            f"Current position (FEN): {fen}\n"
            f"ASCII Board:\n{self.state.board}\n"
            f"It is {turn} to move.\n"
            f"Legal moves: {', '.join(legal_moves)}"
        )
        
        if self.state.move_history:
            state += f"\nMove history: {', '.join(self.state.move_history)}"
            
        # Add player-specific information
        if player_id:
            player_color = "White" if player_id == 1 else "Black"
            state += f"\nYou are playing {player_color}"
            
        return state
        
    def get_game_image(self) -> Optional[str]:
        """Generate base64 encoded PNG of current board state"""
        try:
            svg_content = chess.svg.board(
                board=self.state.board,
                size=self.config.board_size,
                coordinates=True,
                lastmove=self.state.board.peek() if self.state.board.move_stack else None,
                flipped=not self.state.board.turn
            )
            
            png_data = cairosvg.svg2png(bytestring=svg_content)
            # Save to disk
            with open("board.png", "wb") as f:
                f.write(png_data)
            # with open(self.config.run_dir / "board.png", "wb") as f:
            #     f.write(png_data)
            return f"data:image/png;base64,{base64.b64encode(png_data).decode()}"
            
        except Exception as e:
            print(f"Warning: Failed to generate board image: {e}")
            return None
            
    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        print(f"Player {player_id} move: {response}")
        # Verify it's the player's turn
        if player_id != self.state.current_player:
            return {
                "valid": True,
                "message": f"Not your turn. Waiting for {'White' if self.state.current_player == 1 else 'Black'}",
                "end_turn": True,
                "end_game": False,
                "skip_inference": True
            }
            
        # Parse move
        match = self.move_pattern.search(response)
        if not match:
            return {
                "valid": False,
                "message": "Error: Move must be in format 'MOVE: e2e4'",
                "end_turn": False
            }
            
        move_str = match.group(1)
        try:
            move = chess.Move.from_uci(move_str)
            if move not in self.state.board.legal_moves:
                legal_moves = [m.uci() for m in self.state.board.legal_moves]
                return {
                    "valid": False,
                    "message": f"Illegal move: {move_str}! Legal moves: {', '.join(legal_moves)}",
                    "end_turn": False
                }
                
            # Make the move
            self.state.board.push(move)
            self.state.move_history.append(move_str)
            
            # Switch players
            self.state.current_player = 3 - self.state.current_player
            
            # Check game end conditions
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
        if not self.state.board.is_game_over():
            return {
                "status": "in_progress",
                "move_history": self.state.move_history
            }
            
        outcome = self.state.board.outcome()
        
        if outcome.winner is not None:
            # True for White win, False for Black win
            winner = 1 if outcome.winner else 2
            return {
                "status": "complete",
                "winner": f"Player {winner}",
                "player": winner,
                "reason": "checkmate",
                "final_position": self.state.board.fen(),
                "move_history": self.state.move_history
            }
        else:
            # Draw
            return {
                "status": "draw",
                "reason": outcome.termination.name.lower(),
                "final_position": self.state.board.fen(),
                "move_history": self.state.move_history
            }
            
    def get_final_position(self) -> str:
        """Get string representation of final game state"""
        return (
            f"Final position (FEN): {self.state.board.fen()}\n"
            f"Move history: {', '.join(self.state.move_history)}\n"
            f"Final board:\n{self.state.board}"
        )