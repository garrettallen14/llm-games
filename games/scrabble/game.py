# games/scrabble/game.py

from enum import Enum
from typing import Dict, Optional, List, Set, Tuple
from pathlib import Path
from datetime import datetime
import re
import numpy as np
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from random import shuffle

from base.llm_player import BaseLLMPlayer

# Constants
LETTER_VALUES = {
    'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4,
    'I': 1, 'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8,
    'Y': 4, 'Z': 10, '#': 0  # # represents blank tile
}

INITIAL_DISTRIBUTION = {
    'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 2,
    'I': 9, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8, 'P': 2,
    'Q': 1, 'R': 6, 'S': 4, 'T': 6, 'U': 4, 'V': 2, 'W': 2, 'X': 1,
    'Y': 2, 'Z': 1, '#': 2
}

PREMIUM_SQUARES = {
    'TRIPLE_WORD': [(0,0), (7,0), (14,0), (0,7), (14,7), (0,14), (7,14), (14,14)],
    'DOUBLE_WORD': [(1,1), (2,2), (3,3), (4,4), (1,13), (2,12), (3,11), (4,10),
                    (13,1), (12,2), (11,3), (10,4), (13,13), (12,12), (11,11), (10,10)],
    'TRIPLE_LETTER': [(1,5), (1,9), (5,1), (5,5), (5,9), (5,13), (9,1), (9,5), 
                      (9,9), (9,13), (13,5), (13,9)],
    'DOUBLE_LETTER': [(0,3), (0,11), (2,6), (2,8), (3,0), (3,7), (3,14), (6,2),
                      (6,6), (6,8), (6,12), (7,3), (7,11), (8,2), (8,6), (8,8),
                      (8,12), (11,0), (11,7), (11,14), (12,6), (12,8), (14,3), (14,11)]
}

BOARD_SIZE = 15
RACK_SIZE = 7
DEFAULT_FONT_SIZE = 20

MOVE_REGEX = re.compile(r'^PLAY:\s*([A-Z]+)\s+([A-O]\d{1,2}|[1-9]\d?[A-O])\s+(RIGHT|DOWN)$', re.IGNORECASE)
EXCHANGE_REGEX = re.compile(r'^EXCHANGE:\s*([A-Z#]+)$', re.IGNORECASE)

@dataclass
class Move:
    """Represents a move in the game"""
    word: str
    start_position: Tuple[int, int]
    direction: str  # "RIGHT" or "DOWN"
    player_id: int
    score: int = 0
    positions: List[Tuple[int, int]] = field(default_factory=list)
    tiles_used: List['Tile'] = field(default_factory=list)
    
class Tile:
    """Represents a single letter tile"""
    def __init__(self, letter: str, letter_values: Dict[str, int]):
        self.letter = letter.upper()
        self.is_blank = (letter == '#')
        self.points = letter_values[self.letter]
        self.display_letter = None if self.is_blank else self.letter

    def get_letter(self) -> str:
        return self.display_letter or self.letter
    
    def get_score(self) -> int:
        return 0 if self.is_blank else self.points

    def set_blank_letter(self, letter: str) -> None:
        if self.is_blank:
            self.display_letter = letter.upper()

class Bag:
    """Manages the pool of available tiles"""
    def __init__(self):
        self.bag = []
        self.initialize_bag()

    def initialize_bag(self) -> None:
        for letter, count in INITIAL_DISTRIBUTION.items():
            for _ in range(count):
                self.bag.append(Tile(letter, LETTER_VALUES))
        shuffle(self.bag)

    def draw_tile(self) -> Optional[Tile]:
        return self.bag.pop() if self.bag else None

    def get_remaining_tiles(self) -> int:
        return len(self.bag)

@dataclass
class ScrabbleState:
    """Tracks the current state of the Scrabble game"""
    board: np.ndarray = field(default_factory=lambda: np.full((BOARD_SIZE, BOARD_SIZE), None))
    current_player: int = 1
    player_scores: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0})
    player_racks: Dict[int, List[Tile]] = field(default_factory=lambda: {1: [], 2: []})
    move_history: List[Move] = field(default_factory=list)
    last_move: Optional[Move] = None
    bag: Bag = field(default_factory=Bag)
    consecutive_passes: int = 0

@dataclass
class ScrabbleConfig:
    """Configuration for Scrabble game"""
    run_dir: Path
    max_turns: int = 100
    dictionary_path: Path = Path("games/scrabble/dic.txt")

class ScrabbleGame:
    def __init__(self, run_dir: Path, max_turns: int = 100):
        """Initialize the game"""
        self.config = ScrabbleConfig(run_dir=run_dir, max_turns=max_turns)
        self.state = ScrabbleState()
        
        # Load dictionary
        with open(self.config.dictionary_path) as f:
            self.dictionary = set(word.strip().upper() for word in f)
        
        # Initialize regex patterns
        self.move_pattern = re.compile(r"PLAY:\s*(\w+)\s+([A-O](?:1[0-5]|[1-9])|(?:1[0-5]|[1-9])[A-O])\s+(RIGHT|DOWN)")
        self.exchange_pattern = re.compile(r"EXCHANGE:\s*([A-Z#]+)")
        
        # Initialize fonts
        try:
            self.font = ImageFont.truetype("Arial", DEFAULT_FONT_SIZE)
            self.small_font = ImageFont.truetype("Arial", DEFAULT_FONT_SIZE // 2)
        except OSError:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
        
        # Initial tile deal
        self._deal_initial_tiles()

    def get_system_prompt(self) -> Dict[str, str]:
        """Return the system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are playing Scrabble. Submit moves in one of these formats:

1. Play a word: 'PLAY: WORD POSITION DIRECTION'
   Example: 'PLAY: HELLO H8 RIGHT' or 'PLAY: WORLD 8H DOWN'
   
2. Exchange tiles: 'EXCHANGE: ABC'
   Example: 'EXCHANGE: XYZ' to swap those letters
   
3. Pass turn: 'PASS'

Rules:
- First move must include center square (H8)
- Words must connect to existing tiles (after first move)
- All words formed must be valid dictionary words
- Premium squares: TW (Triple Word), DW (Double Word), etc.
- Blank tiles (#) can represent any letter
- Game ends when bag is empty and a player uses all tiles

Try to win the game!"""
        }

    def get_current_state(self, player_id: Optional[int] = None) -> str:
        """Get the current game state from a player's perspective"""
        state = [
            f"Current board:",
            self._board_to_string(),
            f"\nScores - Player 1: {self.state.player_scores[1]}, Player 2: {self.state.player_scores[2]}",
            f"Tiles in bag: {self.state.bag.get_remaining_tiles()}",
            f"Player {self.state.current_player}'s turn"
        ]
        
        if player_id == self.state.current_player:
            rack = self.state.player_racks[player_id]
            state.append(f"\nYour rack: {' '.join(tile.get_letter() for tile in rack)}")
            
        if self.state.last_move:
            state.append(f"\nLast move: {self.state.last_move.word} "
                        f"for {self.state.last_move.score} points")
            
        return "\n".join(state)

    def get_game_image(self) -> Optional[str]:
        """Generate visualization of the current game state"""
        try:
            img = self._create_board_image()
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img.save("board.png")
            return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        except Exception as e:
            print(f"Failed to generate game image: {e}")
            return None


    def attempt_move(self, response: str, player_id: int) -> Dict:
        """Process a move attempt and return the outcome"""
        print(f"Player {player_id} move: {response}")
        
        # 1. Basic input validation
        if not response or not isinstance(response, str):
            return {
                "valid": False,
                "message": "Invalid move format. Must be 'PLAY: WORD POSITION DIRECTION', 'EXCHANGE: ABC', or 'PASS'",
                "end_turn": False
            }
        
        response = response.strip().upper()
        
        # 2. Verify player turn
        if player_id != self.state.current_player:
            return {
                "valid": False,
                "message": f"Not your turn. Waiting for Player {self.state.current_player}",
                "end_turn": False
            }
        
        # 3. Handle PASS
        if response == "PASS":
            self.state.consecutive_passes += 1
            self._next_player()
            return {
                "valid": True,
                "message": "Turn passed",
                "end_turn": True,
                "end_game": self.state.consecutive_passes >= 6,
                "skip_inference": False
            }
        
        # 4. Handle EXCHANGE
        if exchange_match := re.match(r"^EXCHANGE:\s*([A-Z#]+)$", response):
            letters = exchange_match.group(1)
            rack_letters = [tile.get_letter() for tile in self.state.player_racks[player_id]]
            
            # Validate letters to exchange
            for letter in letters:
                if letter not in rack_letters:
                    return {
                        "valid": False,
                        "message": f"Cannot exchange letter '{letter}' - not in your rack: {' '.join(rack_letters)}",
                        "end_turn": False
                    }
            
            # Check enough tiles in bag
            if len(letters) > self.state.bag.get_remaining_tiles():
                return {
                    "valid": False,
                    "message": f"Cannot exchange {len(letters)} tiles - only {self.state.bag.get_remaining_tiles()} tiles left in bag",
                    "end_turn": False
                }
            
            return self._handle_exchange(letters, player_id)
        
        # 5. Handle word play - parse format
        play_match = re.match(r"^PLAY:\s*([A-Z]+)\s+([A-O]\d{1,2}|\d{1,2}[A-O])\s+(RIGHT|DOWN)$", response)
        if not play_match:
            return {
                "valid": False,
                "message": "Invalid play format. Must be 'PLAY: WORD POSITION DIRECTION' (e.g., 'PLAY: HELLO H8 RIGHT')",
                "end_turn": False
            }
        
        word = play_match.group(1)
        position_str = play_match.group(2)
        direction = play_match.group(3)
        
        # 6. Validate dictionary word
        if word not in self.dictionary:
            return {
                "valid": False,
                "message": f"'{word}' is not in the dictionary",
                "end_turn": False
            }
        
        # 7. Parse and validate position
        try:
            row, col = self._parse_position(position_str)
            if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
                return {
                    "valid": False,
                    "message": f"Position {position_str} is outside the board boundaries",
                    "end_turn": False
                }
        except ValueError:
            return {
                "valid": False,
                "message": f"Invalid position format: {position_str}. Use format like 'H8' or '8H'",
                "end_turn": False
            }
        
        # 8. Create move object
        move = Move(
            word=word,
            start_position=(row, col),
            direction=direction,
            player_id=player_id
        )
        
        # 9. Check word fits on board
        if direction == "RIGHT" and col + len(word) > BOARD_SIZE:
            return {
                "valid": False,
                "message": f"Word '{word}' would extend beyond right edge of board from {position_str}",
                "end_turn": False
            }
        if direction == "DOWN" and row + len(word) > BOARD_SIZE:
            return {
                "valid": False,
                "message": f"Word '{word}' would extend beyond bottom edge of board from {position_str}",
                "end_turn": False
            }
        
        # 10. Validate tile availability
        rack_tiles = self.state.player_racks[player_id]
        available_letters = [tile.get_letter() for tile in rack_tiles]
        required_letters = list(word)
        
        for letter in required_letters:
            if letter not in available_letters:
                return {
                    "valid": False,
                    "message": f"Cannot form '{word}' - missing letter '{letter}'. Your rack: {' '.join(available_letters)}",
                    "end_turn": False
                }
            available_letters.remove(letter)
        
        # 11. Check first move covers center
        if not self.state.move_history:
            center = BOARD_SIZE // 2
            covers_center = False
            
            if direction == "RIGHT":
                if row == center and col <= center < col + len(word):
                    covers_center = True
            else:  # DOWN
                if col == center and row <= center < row + len(word):
                    covers_center = True
                    
            if not covers_center:
                return {
                    "valid": False,
                    "message": "First move must cover the center square (H8)",
                    "end_turn": False
                }
        
        # 12. Check connectivity (after first move)
        if self.state.move_history and not self._check_connectivity(move):
            return {
                "valid": False,
                "message": "Word must connect to existing tiles on the board",
                "end_turn": False
            }
        
        # 13. Check for invalid crosswords
        invalid_words = self._find_invalid_words(move)
        if invalid_words:
            return {
                "valid": False,
                "message": f"Would create invalid words: {', '.join(invalid_words)}",
                "end_turn": False
            }
        
        # 14. All validation passed - apply move
        move.score = self._calculate_score(move)
        self._apply_move(move)
        self.state.consecutive_passes = 0
        
        return {
            "valid": True,
            "message": f"Played {word} for {move.score} points",
            "end_turn": True,
            "end_game": self._check_game_over(),
            "skip_inference": False
        }



    def _handle_exchange(self, letters: str, player_id: int) -> Dict:
        """Handle tile exchange move"""
        letters = letters.upper()
        rack = self.state.player_racks[player_id]
        
        # Verify player has these letters
        rack_letters = [t.get_letter() for t in rack]
        for letter in letters:
            if letter not in rack_letters:
                return {
                    "valid": False,
                    "message": f"You don't have letter {letter}",
                    "end_turn": False
                }

        # Remove letters from rack
        for letter in letters:
            for tile in rack:
                if tile.get_letter() == letter:
                    rack.remove(tile)
                    self.state.bag.bag.append(tile)
                    break

        # Draw new tiles
        shuffle(self.state.bag.bag)
        while len(rack) < RACK_SIZE and self.state.bag.bag:
            rack.append(self.state.bag.draw_tile())

        self._next_player()
        return {
            "valid": True,
            "message": f"Exchanged tiles: {letters}",
            "end_turn": True,
            "end_game": False,
            "skip_inference": False
        }

    def _validate_move(self, move: Move) -> Tuple[bool, str]:
        """Validate a move"""
        # Check dictionary
        if move.word not in self.dictionary:
            return False, f"'{move.word}' is not in the dictionary"

        # Check board bounds
        if not self._check_bounds(move):
            return False, "Word placement would go off board"

        # Check first move covers center
        if not self.state.move_history and not self._covers_center(move):
            return False, "First move must cover center square (H8)"

        # Check connectivity
        if self.state.move_history and not self._check_connectivity(move):
            return False, "Word must connect to existing tiles"

        # Check tile availability
        if not self._check_tiles_available(move):
            return False, "You don't have the required tiles"

        # Validate all created words
        invalid_words = self._find_invalid_words(move)
        if invalid_words:
            return False, f"Would create invalid words: {', '.join(invalid_words)}"

        return True, ""
    
    def _board_to_string(self) -> str:
        """Convert board to string representation"""
        # Column headers
        result = ["   " + " ".join(chr(65 + i) for i in range(BOARD_SIZE))]
        
        # Add horizontal line
        result.append("   " + "-" * (BOARD_SIZE * 2 - 1))
        
        # Board rows
        for row in range(BOARD_SIZE):
            # Row number (padded for alignment)
            row_str = f"{row + 1:2d}|"
            
            # Row contents
            for col in range(BOARD_SIZE):
                tile = self.state.board[row, col]
                if tile is None:
                    # Show premium square indicators
                    pos = (row, col)
                    if pos in PREMIUM_SQUARES['TRIPLE_WORD']:
                        row_str += "TW"
                    elif pos in PREMIUM_SQUARES['DOUBLE_WORD']:
                        row_str += "DW"
                    elif pos in PREMIUM_SQUARES['TRIPLE_LETTER']:
                        row_str += "TL"
                    elif pos in PREMIUM_SQUARES['DOUBLE_LETTER']:
                        row_str += "DL"
                    elif pos == (BOARD_SIZE//2, BOARD_SIZE//2):
                        row_str += "★ "  # Center square
                    else:
                        row_str += ". "
                else:
                    # Show letter (and score as subscript if space allows)
                    row_str += f"{tile.get_letter()} "
                    
            result.append(row_str)
        
        return "\n".join(result)

    def _create_board_image(self) -> Image.Image:
        """Create visualization of current board state"""
        cell_size = 40
        padding = 20
        width = BOARD_SIZE * cell_size + 2 * padding
        height = width
        
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw grid and premium squares
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                x = padding + col * cell_size
                y = padding + row * cell_size
                
                # Draw cell
                draw.rectangle(
                    [x, y, x + cell_size, y + cell_size],
                    outline='black'
                )
                
                # Color premium squares
                pos = (row, col)
                if pos in PREMIUM_SQUARES['TRIPLE_WORD']:
                    draw.rectangle([x+1, y+1, x+cell_size-1, y+cell_size-1],
                                 fill='red')
                elif pos in PREMIUM_SQUARES['DOUBLE_WORD']:
                    draw.rectangle([x+1, y+1, x+cell_size-1, y+cell_size-1],
                                 fill='pink')
                elif pos in PREMIUM_SQUARES['TRIPLE_LETTER']:
                    draw.rectangle([x+1, y+1, x+cell_size-1, y+cell_size-1],
                                 fill='blue')
                elif pos in PREMIUM_SQUARES['DOUBLE_LETTER']:
                    draw.rectangle([x+1, y+1, x+cell_size-1, y+cell_size-1],
                                 fill='lightblue')
                
                # Draw tiles
                tile = self.state.board[row, col]
                if tile:
                    # Tile background
                    draw.rectangle([x+2, y+2, x+cell_size-2, y+cell_size-2],
                                 fill='tan', outline='brown')
                    
                    # Letter
                    letter = tile.get_letter()
                    draw.text((x + cell_size//2, y + cell_size//2),
                            letter, fill='black', font=self.font, anchor="mm")
                    
                    # Score
                    if not tile.is_blank:
                        draw.text((x + cell_size - 5, y + cell_size - 5),
                                str(tile.points), fill='black',
                                font=self.small_font, anchor="rb")
        
        # Draw coordinates
        for i in range(BOARD_SIZE):
            # Column letters
            draw.text((padding + i * cell_size + cell_size//2, padding//2),
                     chr(65 + i), fill='black',
                     font=self.small_font, anchor="mm")
            # Row numbers
            draw.text((padding//2, padding + i * cell_size + cell_size//2),
                     str(i + 1), fill='black',
                     font=self.small_font, anchor="mm")
        
        # Draw current player's rack if we have one
        if rack := self.state.player_racks.get(self.state.current_player):
            rack_y = height + 10
            rack_height = cell_size + 20
            rack_img = Image.new('RGB', (width, rack_height), 'white')
            rack_draw = ImageDraw.Draw(rack_img)
            
            for i, tile in enumerate(rack):
                x = padding + i * cell_size
                y = 10
                
                # Tile background
                rack_draw.rectangle([x, y, x + cell_size - 2, y + cell_size - 2],
                                  fill='tan', outline='brown')
                
                # Letter
                letter = tile.get_letter()
                rack_draw.text((x + cell_size//2, y + cell_size//2),
                             letter, fill='black', font=self.font, anchor="mm")
                
                # Score
                if not tile.is_blank:
                    rack_draw.text((x + cell_size - 5, y + cell_size - 5),
                                 str(tile.points), fill='black',
                                 font=self.small_font, anchor="rb")
            
            # Combine board and rack images
            combined = Image.new('RGB', (width, height + rack_height), 'white')
            combined.paste(img, (0, 0))
            combined.paste(rack_img, (0, height))
            return combined
            
        return img

    def _deal_initial_tiles(self) -> None:
        """Deal starting hands to both players"""
        for player_id in [1, 2]:
            rack = []
            for _ in range(RACK_SIZE):
                if tile := self.state.bag.draw_tile():
                    rack.append(tile)
            self.state.player_racks[player_id] = rack

    def _parse_position(self, pos_str: str) -> Tuple[int, int]:
        """Convert position string (e.g. 'H8' or '8H') to (row, col) tuple"""
        letter = next(c for c in pos_str if c.isalpha())
        number = int(''.join(c for c in pos_str if c.isdigit()))
        col = ord(letter.upper()) - ord('A')
        row = number - 1
        return (row, col)

    def _check_bounds(self, move: Move) -> bool:
        """Check if move stays within board boundaries"""
        row, col = move.start_position
        if row < 0 or col < 0 or row >= BOARD_SIZE or col >= BOARD_SIZE:
            return False
            
        length = len(move.word)
        if move.direction == "RIGHT":
            return col + length <= BOARD_SIZE
        else:  # DOWN
            return row + length <= BOARD_SIZE

    def _covers_center(self, move: Move) -> bool:
        """Check if move covers center square (H8)"""
        row, col = move.start_position
        length = len(move.word)
        center = BOARD_SIZE // 2
        
        if move.direction == "RIGHT":
            return (row == center and 
                   col <= center < col + length)
        else:  # DOWN
            return (col == center and 
                   row <= center < row + length)

    def _check_connectivity(self, move: Move) -> bool:
        """Check if move connects to existing tiles"""
        row, col = move.start_position
        for i, letter in enumerate(move.word):
            if move.direction == "RIGHT":
                curr_row, curr_col = row, col + i
            else:  # DOWN
                curr_row, curr_col = row + i, col
                
            # Check adjacent squares
            adjacents = [
                (curr_row-1, curr_col), (curr_row+1, curr_col),
                (curr_row, curr_col-1), (curr_row, curr_col+1)
            ]
            
            for adj_row, adj_col in adjacents:
                if (0 <= adj_row < BOARD_SIZE and 
                    0 <= adj_col < BOARD_SIZE and
                    self.state.board[adj_row, adj_col] is not None):
                    return True
                    
        return False

    def _check_tiles_available(self, move: Move) -> bool:
        """Check if player has required tiles for move"""
        rack = self.state.player_racks[move.player_id].copy()
        for letter in move.word:
            found = False
            for tile in rack:
                if tile.get_letter() == letter or tile.is_blank:
                    rack.remove(tile)
                    found = True
                    break
            if not found:
                return False
        return True

    def _find_invalid_words(self, move: Move) -> List[str]:
        """Find any invalid words that would be created by move"""
        invalid_words = []
        test_board = self.state.board.copy()
        
        # Place the main word temporarily
        row, col = move.start_position
        for i, letter in enumerate(move.word):
            if move.direction == "RIGHT":
                test_board[row, col + i] = Tile(letter, LETTER_VALUES)
            else:  # DOWN
                test_board[row + i, col] = Tile(letter, LETTER_VALUES)
        
        # Check all words formed
        words = self._find_all_words(test_board)
        for word in words:
            if word not in self.dictionary:
                invalid_words.append(word)
                
        return invalid_words

    def _find_all_words(self, board: np.ndarray) -> List[str]:
        """Find all words on the board (horizontal and vertical)"""
        words = []
        
        # Horizontal words
        for row in range(BOARD_SIZE):
            word = ''
            for col in range(BOARD_SIZE):
                if tile := board[row, col]:
                    word += tile.get_letter()
                elif word:
                    if len(word) > 1:
                        words.append(word)
                    word = ''
            if len(word) > 1:
                words.append(word)
        
        # Vertical words
        for col in range(BOARD_SIZE):
            word = ''
            for row in range(BOARD_SIZE):
                if tile := board[row, col]:
                    word += tile.get_letter()
                elif word:
                    if len(word) > 1:
                        words.append(word)
                    word = ''
            if len(word) > 1:
                words.append(word)
                
        return words

    def _calculate_score(self, move: Move) -> int:
        """Calculate score for move including premium squares and crosswords"""
        score = 0
        word_multiplier = 1
        row, col = move.start_position
        
        # Score main word
        for i, letter in enumerate(move.word):
            curr_row = row + (i if move.direction == "DOWN" else 0)
            curr_col = col + (i if move.direction == "RIGHT" else 0)
            letter_multiplier = 1
            pos = (curr_row, curr_col)
            
            if pos in PREMIUM_SQUARES['TRIPLE_WORD']:
                word_multiplier *= 3
            elif pos in PREMIUM_SQUARES['DOUBLE_WORD']:
                word_multiplier *= 2
            elif pos in PREMIUM_SQUARES['TRIPLE_LETTER']:
                letter_multiplier = 3
            elif pos in PREMIUM_SQUARES['DOUBLE_LETTER']:
                letter_multiplier = 2
                
            score += LETTER_VALUES[letter] * letter_multiplier
            
        score *= word_multiplier
        
        # Add bonus for using all tiles
        if len(move.word) == RACK_SIZE:
            score += 50
            
        return score

    def _apply_move(self, move: Move) -> None:
        """Apply move to game state"""
        row, col = move.start_position
        rack = self.state.player_racks[move.player_id]
        
        # Place tiles
        for i, letter in enumerate(move.word):
            if move.direction == "RIGHT":
                curr_row, curr_col = row, col + i
            else:  # DOWN
                curr_row, curr_col = row + i, col
                
            # Find tile to use (prefer exact match over blank)
            tile = None
            for t in rack:
                if t.get_letter() == letter:
                    tile = t
                    break
            if not tile:
                for t in rack:
                    if t.is_blank:
                        tile = t
                        tile.set_blank_letter(letter)
                        break
                        
            rack.remove(tile)
            self.state.board[curr_row, curr_col] = tile
            
        # Calculate score
        move.score = self._calculate_score(move)
        self.state.player_scores[move.player_id] += move.score
        
        # Draw new tiles
        while len(rack) < RACK_SIZE and self.state.bag.bag:
            rack.append(self.state.bag.draw_tile())
            
        # Update game state
        self.state.move_history.append(move)
        self.state.last_move = move
        self._next_player()

    def _next_player(self) -> None:
        """Switch to next player"""
        self.state.current_player = 3 - self.state.current_player

    def _check_game_over(self) -> bool:
        """Check if game should end"""
        # Game ends if:
        # 1. No tiles in bag and a player is out of tiles
        # 2. Six consecutive passes
        if self.state.consecutive_passes >= 6:
            return True
            
        if not self.state.bag.bag:
            for rack in self.state.player_racks.values():
                if not rack:
                    return True
                    
        return False

    def get_game_result(self) -> Dict[str, str]:
        """Get the final game result"""
        if not self._check_game_over():
            return {
                "status": "in_progress",
                "move_history": [m.word for m in self.state.move_history]
            }
            
        # Calculate final scores (subtract remaining tiles)
        final_scores = self.state.player_scores.copy()
        for player_id, rack in self.state.player_racks.items():
            penalty = sum(tile.points for tile in rack)
            final_scores[player_id] -= penalty
            
        # Determine winner
        if final_scores[1] > final_scores[2]:
            winner = 1
        elif final_scores[2] > final_scores[1]:
            winner = 2
        else:
            return {
                "status": "draw",
                "final_scores": final_scores,
                "move_history": [m.word for m in self.state.move_history]
            }
            
        return {
            "status": "complete",
            "winner": f"Player {winner}",
            "player": winner,
            "score": f"{final_scores[1]}-{final_scores[2]}",
            "move_history": [m.word for m in self.state.move_history]
        }

    def run(self, players: List[BaseLLMPlayer]) -> Dict:
        """Run the game with provided players"""
        turn = 0
        while turn < self.config.max_turns:
            current_player = players[self.state.current_player - 1]
            
            # Get game state with error messages if any
            state_msg = self.get_current_state(self.state.current_player)
            
            # Get player response
            try:
                response = current_player.get_response(
                    {"role": "user", "content": state_msg}, 
                    self.get_game_image()
                )
                
                # Attempt move and get outcome
                outcome = self.attempt_move(response, self.state.current_player)
                
                if not outcome["valid"]:
                    # Send error message back to same player
                    error_msg = f"***ERROR:*** {outcome['message']}"
                    current_player.log_message({"role": "system", "content": error_msg})
                    current_player.log_message({"role": "system", "content": error_msg})
                    current_player.log_message({"role": "system", "content": error_msg})
                    current_player.messages.append({"role": "system", "content": error_msg})
                    current_player.messages.append({"role": "system", "content": error_msg})
                    current_player.messages.append({"role": "system", "content": error_msg})
                    continue
                    
                # Valid move - notify other player
                other_player_id = 3 - self.state.current_player
                other_player = players[other_player_id - 1]
                other_state = self.get_current_state(other_player_id)
                
                if not outcome["skip_inference"]:
                    other_player.get_response(
                        {"role": "user", "content": other_state},
                        self.get_game_image()
                    )
                
                if outcome["end_game"]:
                    return self.get_game_result()
                    
            except Exception as e:
                print(f"Error during turn: {str(e)}")
                continue
                
            turn += 1
            
        return self.get_game_result()