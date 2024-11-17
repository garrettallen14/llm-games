from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import random
import re
from datetime import datetime
from dataclasses import dataclass, field

from base.llm_player import BaseLLMPlayer
from games.scrabble.constants import *
from games.scrabble.utils import (
    ScrabbleWords, BoardManager, TileBag, MoveValidator,
    Visualizer, Move, Player, Tile
)

@dataclass
class ScrabbleConfig:
    """Game configuration"""
    run_dir: Path
    max_turns: int = 100
    num_players: int = 2

@dataclass
class ScrabbleState:
    """Complete game state"""
    board: np.ndarray = field(default_factory=lambda: np.full((BOARD_SIZE, BOARD_SIZE), '', dtype=str))
    board_manager: BoardManager = field(default_factory=BoardManager)
    current_player: int = 1
    players: Dict[int, Player] = field(default_factory=dict)
    tile_bag: TileBag = field(default_factory=TileBag)
    move_history: List[Move] = field(default_factory=list)
    consecutive_passes: int = 0
    last_move: Optional[Move] = None

class ScrabbleGame:
    """Main game implementation"""
    
    def __init__(self, run_dir: Path, max_turns: int = 100):
        """Initialize game with all components"""
        self.config = ScrabbleConfig(
            run_dir=run_dir,
            max_turns=max_turns
        )
        
        # Initialize core components
        self.words = ScrabbleWords()
        self.state = ScrabbleState()
        self.validator = MoveValidator(self.words)
        self.visualizer = Visualizer()
        
        # Move parsing patterns
        self.move_pattern = re.compile(r"PLAY:\s*([A-Z*]+)\s+([A-O]\d{1,2})\s+(ACROSS|DOWN)")
        self.pass_pattern = re.compile(r"PASS")
        self.exchange_pattern = re.compile(r"EXCHANGE:\s*([A-Z*]+)")
        
        # Game metadata
        self.start_time = datetime.now().isoformat()
        self.end_time = None

    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        # Get word examples
        example_words = []
        lengths = [2, 3, 4, 5, 6, 7]
        sample_words = sorted(list(self.words.words))
        
        for length in lengths:
            matching = [w for w in sample_words if len(w) == length]
            if matching:
                example_words.extend(random.sample(matching, min(3, len(matching))))

        return {
            "role": "system",
            "content": f"""You are playing Scrabble with {self.config.num_players} players.

The dictionary contains {len(self.words.words):,} valid English words.
Example valid words include: {', '.join(example_words[:15])}... and many more.

Use these formats for moves:
1. Play a word: 'PLAY: WORD POSITION DIRECTION'
   - WORD: The word to play (use * for blank tiles)
   - POSITION: Board coordinate (e.g., H8, A1)
   - DIRECTION: ACROSS or DOWN
   Example: 'PLAY: HELLO H8 ACROSS'

2. Pass your turn: 'PASS'

3. Exchange tiles: 'EXCHANGE: LETTERS'
   Example: 'EXCHANGE: ABC'

Rules:
- First word must cross center square (H8)
- Words must connect to existing words
- All formed words must be valid
- Use * to represent blank tiles
- {PASS_LIMIT} consecutive passes ends the game
- Bingo bonus of {BINGO_BONUS} points for using all 7 tiles
- Premium squares multiply word or letter scores

Your rack will be shown before each turn.
Think carefully about scoring opportunities and premium squares.
Make sure all words formed (including crosswords) are valid."""
        }

    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get formatted game state"""
        # Format board
        board_str = "   " + " ".join(COLUMNS) + "\n"
        for i, row in enumerate(self.state.board):
            board_str += f"{i+1:2d} " + " ".join(cell if cell else "." for cell in row) + "\n"
            
        # Base state info
        state = [
            board_str,
            f"Your rack: {' '.join(t.letter for t in self.state.players[player_id].rack)}" if player_id else "",
            "Scores:",
        ]
        
        # Add scores
        for pid, player in self.state.players.items():
            state.append(f"Player {pid}: {player.score}")
            
        # Add last move info
        if self.state.last_move:
            state.append(f"Last move: {self.state.last_move.word} for {self.state.last_move.points} points")
        else:
            state.append("No moves played yet")
            
        return "\n".join(state)

    def _parse_move(self, response: str) -> Optional[Move]:
        """Parse a move string into a Move object"""
        match = self.move_pattern.match(response)
        if not match:
            return None
            
        word, position, direction = match.groups()
        col = ord(position[0]) - ord('A')
        row = int(position[1:]) - 1
        
        return Move(
            word=word.upper(),
            start_position=(row, col),
            direction=Direction[direction]
        )

    def _handle_pass(self, player_id: int) -> Dict:
        """Process a pass move"""
        self.state.consecutive_passes += 1
        self.state.players[player_id].pass_count += 1
        self._switch_player()
        
        return {
            "valid": True,
            "message": f"Player {player_id} passes",
            "end_turn": True,
            "end_game": self.state.consecutive_passes >= PASS_LIMIT
        }

    def _handle_exchange(self, player_id: int, letters: str) -> Dict:
        """Process a tile exchange"""
        if len(letters) > len(self.state.tile_bag.tiles):
            return {
                "valid": False,
                "message": "Not enough tiles in bag for exchange",
                "end_turn": False
            }
            
        player = self.state.players[player_id]
        exchange_tiles = []
        rack_copy = player.rack.copy()
        
        # Find requested tiles
        for letter in letters.upper():
            tile = next((t for t in rack_copy if t.letter == letter), None)
            if not tile:
                return {
                    "valid": False,
                    "message": f"Don't have tile {letter} to exchange",
                    "end_turn": False
                }
            rack_copy.remove(tile)
            exchange_tiles.append(tile)
            
        # Remove exchanged tiles
        for tile in exchange_tiles:
            player.rack.remove(tile)
            
        # Get new tiles
        new_tiles = self.state.tile_bag.exchange_tiles(exchange_tiles)
        player.rack.extend(new_tiles)
        
        self.state.consecutive_passes = 0
        self._switch_player()
        
        return {
            "valid": True,
            "message": "Tiles exchanged successfully",
            "end_turn": True,
            "end_game": False
        }

    def _switch_player(self) -> None:
        """Switch to next player"""
        self.state.current_player = self.state.current_player % self.config.num_players + 1

    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt with improved error handling"""
        print(f"Player {player_id} move: {response}")
        
        # Verify turn
        if player_id != self.state.current_player:
            return {
                "valid": False,
                "message": f"Not your turn. Waiting for Player {self.state.current_player}",
                "end_turn": False
            }

        # Handle pass
        if self.pass_pattern.match(response):
            return self._handle_pass(player_id)

        # Handle exchange
        if exchange_match := self.exchange_pattern.match(response):
            return self._handle_exchange(player_id, exchange_match.group(1))

        # Parse play move
        move = self._parse_move(response)
        if not move:
            return {
                "valid": False,
                "message": "Invalid move format. Use 'PLAY: WORD H8 ACROSS'",
                "end_turn": False
            }

        # Validate move
        is_first = not bool(self.state.move_history)
        valid, error_msg, score_info = self.validator.validate_move(
            move,
            self.state.board,
            self.state.players[player_id].rack,
            is_first
        )

        if not valid:
            return {
                "valid": False,
                "message": error_msg,  # Now properly formatted
                "end_turn": False
            }
            
        # Apply valid move
        self.state.board_manager.place_move(move)
        player = self.state.players[player_id]
        
        # Update scores and state
        move.points = score_info['total_score']
        player.score += move.points
        player.moves.append(move)
        
        # Update tile rack
        for tile in move.tiles_used:
            player.rack.remove(tile)
        new_tiles = self.state.tile_bag.draw_tiles(RACK_SIZE - len(player.rack))
        player.rack.extend(new_tiles)
        
        # Update game state
        self.state.last_move = move
        self.state.move_history.append(move)
        self.state.consecutive_passes = 0
        
        # Switch players
        self._switch_player()
        
        return {
            "valid": True,
            "message": (f"Move {move.word} played for {move.points} points\n"
                       f"Words formed: {', '.join(score_info['words_formed'])}"),
            "end_turn": True,
            "end_game": self._check_game_over()
        }

    def _check_game_over(self) -> bool:
        """Check if game is over"""
        return (
            self.state.consecutive_passes >= PASS_LIMIT or
            (not self.state.tile_bag.tiles and 
             any(not p.rack for p in self.state.players.values()))
        )

    def get_game_result(self) -> Dict:
        """Get final game result"""
        self.end_time = datetime.now().isoformat()
        
        # Calculate final penalties
        final_scores = {}
        for player_id, player in self.state.players.items():
            penalty = sum(tile.points for tile in player.rack)
            final_scores[player_id] = player.score - penalty
            
        # Find winner
        winner = max(final_scores.items(), key=lambda x: x[1])
        
        return {
            "status": "complete",
            "winner": f"Player {winner[0]}",
            "final_scores": final_scores,
            "moves_played": len(self.state.move_history),
            "duration": self.end_time,
            "final_position": self.get_current_state()
        }

    def get_game_image(self) -> Optional[str]:
        """Generate game visualization"""
        try:
            return self.visualizer.create_game_image(
                self.state.board,
                self.state.last_move,
                self.state.players[self.state.current_player].rack
                if self.state.current_player in self.state.players
                else None
            )
        except Exception as e:
            print(f"Failed to generate game image: {e}")
            return None

    def initialize_players(self, num_players: int = 2) -> None:
        """Set up initial player states"""
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"Invalid number of players: {num_players}")
            
        self.config.num_players = num_players
        for player_id in range(1, num_players + 1):
            self.state.players[player_id] = Player(
                id=player_id,
                rack=self.state.tile_bag.draw_tiles(RACK_SIZE)
            )

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run complete game"""
        # Initialize players
        self.initialize_players(len(players))
        
        # Initialize players with system prompt
        for player in players:
            player.initialize_with_prompt(self.get_system_prompt())
            
        turn = 0
        while turn < self.config.max_turns:
            current_player = players[self.state.current_player - 1]
            
            # Get game state
            state_message = {
                "role": "user",
                "content": self.get_current_state(self.state.current_player)
            }
            
            try:
                # Get player's move
                response = current_player.get_response(state_message, self.get_game_image())
                outcome = self.attempt_move(response, self.state.current_player)
                
                if not outcome["valid"]:
                    print(f"Invalid move: {outcome['message']}")
                    continue
                    
                print(f"Turn {turn + 1}: Player {self.state.current_player} - {response}")
                print(f"Outcome: {outcome['message']}")
                
                # Notify other players
                for pid, player in enumerate(players, 1):
                    if pid != self.state.current_player:
                        player.get_response(
                            {
                                "role": "user",
                                "content": f"Player {self.state.current_player} played: {response}\n\n{self.get_current_state(pid)}"
                            },
                            self.get_game_image()
                        )
                
                if outcome["end_game"]:
                    return self.get_game_result()
                    
            except Exception as e:
                print(f"Error during turn: {str(e)}")
                continue
                
            turn += 1
        
        return self.get_game_result()