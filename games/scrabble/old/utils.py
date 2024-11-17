```python
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import random
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from wordfreq import word_frequency, get_frequency_dict

from games.scrabble.constants import *

@dataclass
class Tile:
    """Represents a single Scrabble tile"""
    letter: str
    points: int
    is_blank: bool = False
    assigned_letter: Optional[str] = None

    @classmethod
    def from_letter(cls, letter: str) -> 'Tile':
        """Create a tile from a letter"""
        if letter == '*':
            return cls(letter='*', points=0, is_blank=True)
        data = TILE_DISTRIBUTION[letter]
        return cls(letter=letter, points=data['points'], is_blank=False)

    def assign_blank(self, letter: str) -> None:
        """Assign a letter to a blank tile"""
        if not self.is_blank:
            raise ValueError("Cannot assign letter to non-blank tile")
        self.assigned_letter = letter.upper()

    def get_display_letter(self) -> str:
        """Get the letter to display"""
        return self.assigned_letter if self.is_blank else self.letter

@dataclass
class Move:
    """Represents a complete move with all relevant information"""
    word: str
    start_position: Tuple[int, int]
    direction: Direction
    tiles_used: List[Tile] = field(default_factory=list)
    points: int = 0
    positions: List[Tuple[int, int]] = field(default_factory=list)
    blank_assignments: Dict[int, str] = field(default_factory=dict)
    crosswords_formed: List[str] = field(default_factory=list)
    is_bingo: bool = False
    rack: Optional[List[Tile]] = None

@dataclass
class Player:
    """Tracks player state"""
    id: int
    rack: List[Tile] = field(default_factory=list)
    score: int = 0
    pass_count: int = 0
    moves: List[Move] = field(default_factory=list)

class ScrabbleWords:
    """Dictionary management with frequency-based filtering"""
    
    def __init__(self, min_frequency: float = MIN_WORD_FREQUENCY):
        self.min_frequency = min_frequency
        self.words: Set[str] = set()
        self.cache: Dict[str, bool] = {}
        self.stats = {
            'total_words': 0,
            'by_length': {},
            'frequency_ranges': {
                'very_common': 0,    # > 1e-3
                'common': 0,         # > 1e-4
                'uncommon': 0,       # > 1e-5
                'rare': 0,           # > 1e-6
                'very_rare': 0       # <= 1e-6
            }
        }
        self._load_dictionary()

    def _load_dictionary(self) -> None:
        """Load and filter English words from wordfreq"""
        print("Loading word frequency dictionary...")
        frequency_dict = get_frequency_dict('en')
        
        for word, freq in frequency_dict.items():
            if (MIN_WORD_LENGTH <= len(word) <= BOARD_SIZE and 
                freq >= self.min_frequency and 
                word.isalpha()):
                
                self.words.add(word.upper())
                
                # Update statistics
                length = len(word)
                self.stats['by_length'][length] = self.stats['by_length'].get(length, 0) + 1
                
                if freq > FREQUENCY_RANGES['VERY_COMMON']:
                    self.stats['frequency_ranges']['very_common'] += 1
                elif freq > FREQUENCY_RANGES['COMMON']:
                    self.stats['frequency_ranges']['common'] += 1
                elif freq > FREQUENCY_RANGES['UNCOMMON']:
                    self.stats['frequency_ranges']['uncommon'] += 1
                elif freq > FREQUENCY_RANGES['RARE']:
                    self.stats['frequency_ranges']['rare'] += 1
                else:
                    self.stats['frequency_ranges']['very_rare'] += 1

        self.stats['total_words'] = len(self.words)
        self._print_stats()

    def _print_stats(self) -> None:
        """Print detailed dictionary statistics"""
        print("\n=== Scrabble Dictionary Statistics ===")
        print(f"Total words loaded: {self.stats['total_words']:,}")
        
        print("\nWord count by length:")
        for length in sorted(self.stats['by_length'].keys()):
            count = self.stats['by_length'][length]
            percentage = (count / self.stats['total_words']) * 100
            print(f"{length} letters: {count:,} ({percentage:.1f}%)")
        
        print("\nWord count by frequency:")
        for category, count in self.stats['frequency_ranges'].items():
            percentage = (count / self.stats['total_words']) * 100
            print(f"{category.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
        print("\nDictionary ready for use.")

    def is_valid(self, word: str) -> bool:
        """Check if a word is valid, with caching"""
        word = word.upper()
        if word in self.cache:
            return self.cache[word]
        
        is_valid = word in self.words
        self.cache[word] = is_valid
        return is_valid

class TileBag:
    """Manages tile distribution and drawing"""
    
    def __init__(self):
        self.tiles: List[Tile] = []
        self.remaining: Dict[str, int] = {}
        self._initialize_tiles()

    def _initialize_tiles(self) -> None:
        """Create initial tile distribution"""
        for letter, data in TILE_DISTRIBUTION.items():
            for _ in range(data['count']):
                self.tiles.append(Tile.from_letter(letter))
                self.remaining[letter] = data['count']
        random.shuffle(self.tiles)

    def draw_tiles(self, count: int) -> List[Tile]:
        """Draw specified number of tiles"""
        drawn = []
        for _ in range(min(count, len(self.tiles))):
            tile = self.tiles.pop()
            drawn.append(tile)
            self.remaining[tile.letter] -= 1
        return drawn

    def return_tiles(self, tiles: List[Tile]) -> None:
        """Return tiles to the bag"""
        self.tiles.extend(tiles)
        for tile in tiles:
            self.remaining[tile.letter] += 1
        random.shuffle(self.tiles)

    def exchange_tiles(self, tiles: List[Tile]) -> List[Tile]:
        """Exchange tiles if enough remain in bag"""
        if len(tiles) > len(self.tiles):
            return tiles
        self.return_tiles(tiles)
        return self.draw_tiles(len(tiles))

    def tiles_remaining(self) -> int:
        """Get count of remaining tiles"""
        return len(self.tiles)

class BoardManager:
    """Manages board state and operations"""
    
    def __init__(self):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), '', dtype=str)
        self.occupied_positions: Set[Tuple[int, int]] = set()
        self.anchor_points: Set[Tuple[int, int]] = {CENTER_SQUARE}

    def place_move(self, move: Move) -> None:
        """Place a move on the board and update state"""
        positions = self._get_word_positions(move)
        for pos, letter in zip(positions, move.word):
            if pos not in self.occupied_positions:
                row, col = pos
                self.board[row][col] = letter
                self.occupied_positions.add(pos)
                self._update_anchor_points(pos)

    def _get_word_positions(self, move: Move) -> List[Tuple[int, int]]:
        """Get all positions for a word"""
        positions = []
        row, col = move.start_position
        dr, dc = move.direction.value
        
        for i in range(len(move.word)):
            positions.append((row + i * dr, col + i * dc))
        
        return positions

    def _update_anchor_points(self, pos: Tuple[int, int]) -> None:
        """Update valid starting positions after a tile placement"""
        row, col = pos
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < BOARD_SIZE and 
                0 <= new_col < BOARD_SIZE and 
                (new_row, new_col) not in self.occupied_positions):
                self.anchor_points.add((new_row, new_col))

class MoveValidator:
    """Validates all aspects of move legality"""
    
    def __init__(self, dictionary: ScrabbleWords):
        self.dictionary = dictionary

    def validate_move(self, move: Move, board: np.ndarray, rack: List[Tile],
                     is_first_move: bool) -> Tuple[bool, str, Optional[Dict]]:
        """Complete move validation"""
        try:
            # Set rack for tile assignment
            move.rack = rack
            positions = self._get_word_positions(move)
            
            # Basic validations
            if not self._check_boundaries(positions):
                return False, ERROR_MESSAGES['OUT_OF_BOUNDS'], None
                
            if is_first_move and CENTER_SQUARE not in positions:
                return False, ERROR_MESSAGES['NO_CENTER'], None

            # Word validations
            if not self.dictionary.is_valid(move.word):
                return False, ERROR_MESSAGES['INVALID_WORD'].format(word=move.word), None

            # Tile availability
            missing = self._check_tiles_needed(move, board, positions)
            if missing:
                return False, ERROR_MESSAGES['INSUFFICIENT_TILES'].format(
                    tiles=', '.join(missing)
                ), None

            # Connection validation (except first move)
            if not is_first_move and not self._check_connection(positions, board):
                return False, ERROR_MESSAGES['NO_CONNECTION'], None

            # Crossword validation
            crosswords = self._get_crosswords(move, board)
            for word, pos in crosswords:
                if not self.dictionary.is_valid(word):
                    return False, ERROR_MESSAGES['INVALID_CROSSWORD'].format(
                        word=word,
                        pos=f"{chr(pos[1] + ord('A'))}{pos[0] + 1}"
                    ), None

            # Calculate score
            score_info = self._calculate_score(move, board, positions, crosswords)
            move.points = score_info['total_score']
            move.positions = positions
            move.crosswords_formed = [word for word, _ in crosswords]
            
            return True, "", score_info

        except Exception as e:
            return False, f"Invalid move: {str(e)}", None


[Previous constants.py implementation remains exactly the same]

```python
# games/scrabble/utils.py continued from above:

    def _check_boundaries(self, positions: List[Tuple[int, int]]) -> bool:
        """Check if all positions are within board boundaries"""
        return all(0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE 
                  for row, col in positions)

    def _check_tiles_needed(self, move: Move, board: np.ndarray,
                          positions: List[Tuple[int, int]]) -> List[str]:
        """Determine required tiles for move"""
        needed = {}
        for letter, pos in zip(move.word, positions):
            row, col = pos
            if board[row][col] == '':
                needed[letter] = needed.get(letter, 0) + 1

        available = {}
        blank_count = 0
        for tile in move.rack:
            if tile.is_blank:
                blank_count += 1
            else:
                available[tile.letter] = available.get(tile.letter, 0) + 1

        missing = []
        for letter, count in needed.items():
            still_needed = count
            if letter in available:
                used = min(available[letter], still_needed)
                still_needed -= used

            if still_needed > 0 and blank_count > 0:
                used = min(blank_count, still_needed)
                blank_count -= used
                still_needed -= used

            if still_needed > 0:
                missing.append(letter)

        return missing

    def _check_connection(self, positions: List[Tuple[int, int]], 
                         board: np.ndarray) -> bool:
        """Check if move connects to existing tiles"""
        for row, col in positions:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                adj_row, adj_col = row + dr, col + dc
                if (0 <= adj_row < BOARD_SIZE and 
                    0 <= adj_col < BOARD_SIZE and
                    board[adj_row][adj_col] and 
                    (adj_row, adj_col) not in positions):
                    return True
        return False

    def _get_crosswords(self, move: Move, board: np.ndarray
                       ) -> List[Tuple[str, Tuple[int, int]]]:
        """Find all crosswords formed by move"""
        crosswords = []
        temp_board = board.copy()
        row, col = move.start_position
        
        # Place word temporarily
        for i, letter in enumerate(move.word):
            curr_row = row + i * move.direction.value[0]
            curr_col = col + i * move.direction.value[1]
            temp_board[curr_row][curr_col] = letter
        
        # Check perpendicular words
        for i, letter in enumerate(move.word):
            curr_row = row + i * move.direction.value[0]
            curr_col = col + i * move.direction.value[1]
            
            if board[curr_row][curr_col] == '':  # Only check new placements
                perp_word = self._extract_word(
                    temp_board,
                    (curr_row, curr_col),
                    move.direction.perpendicular()
                )
                if perp_word and len(perp_word) > 1:
                    crosswords.append((perp_word, (curr_row, curr_col)))
        
        return crosswords

    def _extract_word(self, board: np.ndarray, pos: Tuple[int, int],
                     direction: Direction) -> Optional[str]:
        """Extract complete word in given direction"""
        row, col = pos
        dr, dc = direction.value
        
        # Find word start
        start_row, start_col = row, col
        while (start_row > 0 and start_col > 0 and 
               board[start_row - dr][start_col - dc]):
            start_row -= dr
            start_col -= dc
        
        # Build word
        word = []
        curr_row, curr_col = start_row, start_col
        while (curr_row < BOARD_SIZE and curr_col < BOARD_SIZE and 
               board[curr_row][curr_col]):
            word.append(board[curr_row][curr_col])
            curr_row += dr
            curr_col += dc
        
        return ''.join(word) if len(word) > 1 else None

    def _calculate_score(self, move: Move, board: np.ndarray,
                        positions: List[Tuple[int, int]],
                        crosswords: List[Tuple[str, Tuple[int, int]]]
                        ) -> Dict:
        """Calculate complete move score"""
        score_info = {
            'word_score': 0,
            'crossword_scores': [],
            'bingo_bonus': 0,
            'total_score': 0,
            'words_formed': [move.word]
        }

        # Main word score
        word_multiplier = 1
        word_score = 0
        
        for i, pos in enumerate(positions):
            row, col = pos
            if board[row][col] == '':  # Only score new tiles
                letter = move.word[i]
                letter_multiplier = 1
                letter_score = 0 if letter in move.blank_assignments else TILE_DISTRIBUTION[letter]['points']

                # Apply premium squares
                if pos in PREMIUM_SQUARES['TRIPLE_LETTER']:
                    letter_multiplier = 3
                elif pos in PREMIUM_SQUARES['DOUBLE_LETTER']:
                    letter_multiplier = 2

                if pos in PREMIUM_SQUARES['TRIPLE_WORD']:
                    word_multiplier *= 3
                elif pos in PREMIUM_SQUARES['DOUBLE_WORD']:
                    word_multiplier *= 2

                word_score += letter_score * letter_multiplier

        score_info['word_score'] = word_score * word_multiplier

        # Crossword scores
        for word, pos in crosswords:
            crossword_score = self._calculate_crossword_score(word, pos, board)
            score_info['crossword_scores'].append(crossword_score)
            score_info['words_formed'].append(word)

        # Bingo bonus
        new_tiles = sum(1 for pos in positions if board[pos[0]][pos[1]] == '')
        if new_tiles == RACK_SIZE:
            score_info['bingo_bonus'] = BINGO_BONUS
            move.is_bingo = True

        # Total score
        score_info['total_score'] = (
            score_info['word_score'] +
            sum(score_info['crossword_scores']) +
            score_info['bingo_bonus']
        )

        return score_info

    def _calculate_crossword_score(self, word: str, pos: Tuple[int, int],
                                 board: np.ndarray) -> int:
        """Calculate score for a crossword"""
        score = 0
        row, col = pos
        
        for i, letter in enumerate(word):
            curr_pos = (row + i, col)
            if board[curr_pos[0]][curr_pos[1]] == '':
                letter_score = TILE_DISTRIBUTION[letter]['points']
                
                if curr_pos in PREMIUM_SQUARES['TRIPLE_LETTER']:
                    letter_score *= 3
                elif curr_pos in PREMIUM_SQUARES['DOUBLE_LETTER']:
                    letter_score *= 2
                    
                score += letter_score
                
        return score

class Visualizer:
    """Handles board and game state visualization"""
    
    def __init__(self):
        self.fonts = self._initialize_fonts()

    def _initialize_fonts(self) -> Dict[str, ImageFont.FreeTypeFont]:
        """Initialize fonts with fallback to default"""
        try:
            return {
                'LETTER': ImageFont.truetype("Arial", DIMENSIONS['FONT_LETTER']),
                'SCORE': ImageFont.truetype("Arial", DIMENSIONS['FONT_SCORE']),
                'COORDINATE': ImageFont.truetype("Arial", DIMENSIONS['FONT_COORDINATE']),
                'PREMIUM': ImageFont.truetype("Arial", DIMENSIONS['FONT_PREMIUM'])
            }
        except OSError:
            default = ImageFont.load_default()
            return {k: default for k in ['LETTER', 'SCORE', 'COORDINATE', 'PREMIUM']}

    def create_game_image(self, board: np.ndarray, last_move: Optional[Move] = None,
                         rack: Optional[List[Tile]] = None) -> str:
        """Create complete game visualization"""
        # Create board image
        board_img = self._create_board_image(board, last_move)
        
        # Create rack image if provided
        if rack is not None:
            rack_img = self._create_rack_image(rack)
            
            # Combine images
            width = max(board_img.width, rack_img.width)
            height = board_img.height + rack_img.height + 20
            
            combined = Image.new('RGB', (width, height), COLORS['BOARD']['BACKGROUND'])
            combined.paste(board_img, (0, 0))
            combined.paste(
                rack_img,
                ((width - rack_img.width) // 2, board_img.height + 10)
            )
        else:
            combined = board_img

        # Convert to base64
        buffer = io.BytesIO()
        combined.save(buffer, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def _create_board_image(self, board: np.ndarray, 
                          last_move: Optional[Move] = None) -> Image.Image:
        """Create board visualization"""
        cell_size = DIMENSIONS['CELL_SIZE']
        padding = DIMENSIONS['PADDING']
        width = BOARD_SIZE * cell_size + 2 * padding
        height = width
        
        img = Image.new('RGB', (width, height), COLORS['BOARD']['BACKGROUND'])
        draw = ImageDraw.Draw(img)
        
        # Draw grid and premium squares
        self._draw_grid(draw, cell_size, padding)
        self._draw_premium_squares(draw, cell_size, padding)
        
        # Draw tiles
        recent_positions = set(last_move.positions) if last_move else set()
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if letter := board[row][col]:
                    x = padding + col * cell_size
                    y = padding + row * cell_size
                    self._draw_tile(
                        draw, x, y, letter,
                        is_recent=(row, col) in recent_positions
                    )
        
        # Draw coordinates
        self._draw_coordinates(draw, cell_size, padding)
        
        return img

    def _create_rack_image(self, rack: List[Tile]) -> Image.Image:
        """Create rack visualization"""
        tile_size = DIMENSIONS['RACK_TILE_SIZE']
        padding = DIMENSIONS['RACK_PADDING']
        width = len(rack) * tile_size + (len(rack) + 1) * padding
        height = tile_size + 2 * padding
        
        img = Image.new('RGB', (width, height), COLORS['BOARD']['BACKGROUND'])
        draw = ImageDraw.Draw(img)
        
        for i, tile in enumerate(rack):
            x = padding + i * (tile_size + padding)
            y = padding
            self._draw_rack_tile(draw, x, y, tile)
        
        return img

    def _draw_grid(self, draw: ImageDraw.Draw, cell_size: int, padding: int) -> None:
        """Draw board grid"""
        for i in range(BOARD_SIZE + 1):
            x = padding + i * cell_size
            y = padding + i * cell_size
            
            draw.line([(x, padding), (x, padding + BOARD_SIZE * cell_size)],
                     fill=COLORS['BOARD']['GRID'],
                     width=DIMENSIONS['GRID_LINE'])
            draw.line([(padding, y), (padding + BOARD_SIZE * cell_size, y)],
                     fill=COLORS['BOARD']['GRID'],
                     width=DIMENSIONS['GRID_LINE'])

    def _draw_premium_squares(self, draw: ImageDraw.Draw, cell_size: int,
                            padding: int) -> None:
        """Draw premium square indicators"""
        for square_type, positions in PREMIUM_SQUARES.items():
            color = COLORS['BOARD'][square_type]
            label = square_type.replace('_', ' ')[:2]
            
            for row, col in positions:
                x = padding + col * cell_size
                y = padding + row * cell_size
                
                draw.rectangle([x, y, x + cell_size, y + cell_size],
                             fill=color)
                draw.text((x + cell_size // 2, y + cell_size // 2),
                         label,
                         fill=COLORS['BOARD']['GRID'],
                         font=self.fonts['PREMIUM'],
                         anchor="mm")

    def _draw_tile(self, draw: ImageDraw.Draw, x: int, y: int, letter: str,
                   is_recent: bool = False) -> None:
        """Draw a letter tile"""
        cell_size = DIMENSIONS['CELL_SIZE']
        padding = 2
        
        # Tile background
        draw.rectangle(
            [x + padding, y + padding,
             x + cell_size - padding, y + cell_size - padding],
            fill=COLORS['TILE']['BACKGROUND'],
            outline=COLORS['TILE']['RECENT'] if is_recent else COLORS['BOARD']['GRID'],
            width=DIMENSIONS['TILE_BORDER'] if is_recent else 1
        )
        
        # Letter
        draw.text((x + cell_size // 2, y + cell_size // 2),
                 letter,
                 fill=COLORS['TILE']['TEXT'],
                 font=self.fonts['LETTER'],
                 anchor="mm")
        
        # Score
        if letter != '*':
            score = TILE_DISTRIBUTION[letter]['points']
            draw.text((x + cell_size - 8, y + cell_size - 8),
                     str(score),
                     fill=COLORS['TILE']['SCORE'],
                     font=self.fonts['SCORE'],
                     anchor="rb")

    def _draw_rack_tile(self, draw: ImageDraw.Draw, x: int, y: int,
                       tile: Tile) -> None:
        """Draw a rack tile"""
        tile_size = DIMENSIONS['RACK_TILE_SIZE']
        
        # Background
        draw.rectangle(
            [x, y, x + tile_size, y + tile_size],
            fill=COLORS['TILE']['BLANK'] if tile.is_blank else COLORS['TILE']['BACKGROUND'],
            outline=COLORS['TILE']['TEXT'],
            width=1
        )
        
        # Letter
        draw.text((x + tile_size // 2, y + tile_size // 2),
                 tile.get_display_letter(),
                 fill=COLORS['TILE']['TEXT'],
                 font=self.fonts['LETTER'],
                 anchor="mm")
        
        # Score (not for blanks)
        if not tile.is_blank:
            draw.text((x + tile_size - 5, y + tile_size - 5),
                     str(tile.points),
                     fill=COLORS['TILE']['SCORE'],
                     font=self.fonts['SCORE'],
                    anchor="rb")

   def _draw_coordinates(self, draw: ImageDraw.Draw, cell_size: int, padding: int) -> None:
       """Draw board coordinates"""
       for i in range(BOARD_SIZE):
           # Column labels (A-O)
           x = padding + i * cell_size + cell_size // 2
           draw.text((x, padding // 2),
                    COLUMNS[i],
                    fill=COLORS['BOARD']['GRID'],
                    font=self.fonts['COORDINATE'],
                    anchor="mm")
           
           # Row labels (1-15)
           y = padding + i * cell_size + cell_size // 2
           draw.text((padding // 2, y),
                    ROWS[i],
                    fill=COLORS['BOARD']['GRID'],
                    font=self.fonts['COORDINATE'],
                    anchor="mm")

```python
# games/scrabble/game.py

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
from datetime import datetime
import numpy as np
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
       """Initialize game"""
       self.config = ScrabbleConfig(
           run_dir=run_dir,
           max_turns=max_turns
       )
       
       # Initialize core components
       self.words = ScrabbleWords()
       self.state = ScrabbleState()
       self.validator = MoveValidator(self.words)
       self.visualizer = Visualizer()
       
       # Move patterns
       self.move_pattern = re.compile(r"PLAY:\s*([A-Z*]+)\s+([A-O]\d{1,2})\s+(ACROSS|DOWN)")
       self.pass_pattern = re.compile(r"PASS")
       self.exchange_pattern = re.compile(r"EXCHANGE:\s*([A-Z*]+)")
       
       # Game metadata
       self.start_time = datetime.now().isoformat()
       self.end_time = None

   def get_system_prompt(self) -> Dict[str, str]:
       """Generate system prompt for LLM players"""
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
- Game ends when no more legal plays are possible"""
       }

   def get_current_state(self, player_id: Optional[int] = None) -> str:
       """Get formatted game state"""
       # Format board
       board_str = "   " + " ".join(COLUMNS) + "\n"
       for i, row in enumerate(self.state.board):
           board_str += f"{i+1:2d} " + " ".join(cell if cell else "." for cell in row) + "\n"
           
       # Base state info
       state = [board_str]
       
       if player_id is not None:
           state.append(f"Your rack: {' '.join(t.get_display_letter() for t in self.state.players[player_id].rack)}")
       
       state.extend([
           "Scores:",
           *[f"Player {pid}: {p.score}" for pid, p in self.state.players.items()]
       ])
       
       if self.state.last_move:
           state.append(f"Last move: {self.state.last_move.word} for {self.state.last_move.points} points")
       else:
           state.append("No moves played yet")
           
       return "\n".join(state)

   def attempt_move(self, response: str, player_id: int) -> Dict:
       """Process a move attempt"""
       print(f"Player {player_id} move: {response}")
       
       # Verify turn
       if player_id != self.state.current_player:
           return {
               "valid": False,
               "message": ERROR_MESSAGES['NOT_YOUR_TURN'].format(
                   player=self.state.current_player
               ),
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
               "message": ERROR_MESSAGES['INVALID_MOVE_FORMAT'],
               "end_turn": False
           }
           
       # Validate move
       is_first = not bool(self.state.move_history)
       valid, error, score_info = self.validator.validate_move(
           move,
           self.state.board,
           self.state.players[player_id].rack,
           is_first
       )
       
       if not valid:
           return {
               "valid": False,
               "message": error,
               "end_turn": False
           }
           
       # Apply valid move
       self._apply_move(move, player_id, score_info)
       
       message = f"Move {move.word} played for {move.points} points"
       if move.crosswords_formed:
           message += f"\nCrosswords formed: {', '.join(move.crosswords_formed)}"
       if move.is_bingo:
           message += f"\nBINGO! +{BINGO_BONUS} points"
           
       return {
           "valid": True,
           "message": message,
           "end_turn": True,
           "end_game": self._check_game_over()
       }

   def _parse_move(self, response: str) -> Optional[Move]:
       """Parse move string into Move object"""
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
       """Process pass move"""
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
       """Process tile exchange"""
       if len(letters) > len(self.state.tile_bag.tiles):
           return {
               "valid": False,
               "message": ERROR_MESSAGES['NO_TILES_EXCHANGE'],
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
           
       # Perform exchange
       for tile in exchange_tiles:
           player.rack.remove(tile)
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

   def _apply_move(self, move: Move, player_id: int, score_info: Dict) -> None:
       """Apply validated move to game state"""
       # Update board
       self.state.board_manager.place_move(move)
       
       # Update player state
       player = self.state.players[player_id]
       player.score += score_info['total_score']
       player.moves.append(move)
       
       # Update tiles
       for tile in move.tiles_used:
           player.rack.remove(tile)
       new_tiles = self.state.tile_bag.draw_tiles(RACK_SIZE - len(player.rack))
       player.rack.extend(new_tiles)
       
       # Update game state
       self.state.move_history.append(move)
       self.state.last_move = move
       self.state.consecutive_passes = 0
       
       # Switch players
       self._switch_player()

   def _switch_player(self) -> None:
       """Switch to next player"""
       self.state.current_player = self.state.current_player % self.config.num_players + 1

   def _check_game_over(self) -> bool:
       """Check if game should end"""
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