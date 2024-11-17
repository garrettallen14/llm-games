# games/scrabble/constants.py
from enum import Enum
from typing import Dict, Set, Tuple, Final, List

class Direction(Enum):
    """Move directions on the board"""
    ACROSS = (0, 1)
    DOWN = (1, 0)

    def perpendicular(self) -> 'Direction':
        """Get perpendicular direction"""
        return Direction.DOWN if self == Direction.ACROSS else Direction.ACROSS

# Board Configuration
BOARD_SIZE: Final = 15
CENTER_SQUARE: Final = (7, 7)
RACK_SIZE: Final = 7
MIN_WORD_LENGTH: Final = 2
BINGO_BONUS: Final = 50
MIN_PLAYERS: Final = 2
MAX_PLAYERS: Final = 4
PASS_LIMIT: Final = 6

# Word Frequencies
MIN_WORD_FREQUENCY: Final = 4e-7
FREQUENCY_RANGES = {
    'VERY_COMMON': 1e-3,
    'COMMON': 1e-4,
    'UNCOMMON': 1e-5,
    'RARE': 1e-6
}

# Tile Distribution and Points
TILE_DISTRIBUTION: Final[Dict[str, Dict[str, int]]] = {
    'A': {'count': 9, 'points': 1},
    'B': {'count': 2, 'points': 3},
    'C': {'count': 2, 'points': 3},
    'D': {'count': 4, 'points': 2},
    'E': {'count': 12, 'points': 1},
    'F': {'count': 2, 'points': 4},
    'G': {'count': 3, 'points': 2},
    'H': {'count': 2, 'points': 4},
    'I': {'count': 9, 'points': 1},
    'J': {'count': 1, 'points': 8},
    'K': {'count': 1, 'points': 5},
    'L': {'count': 4, 'points': 1},
    'M': {'count': 2, 'points': 3},
    'N': {'count': 6, 'points': 1},
    'O': {'count': 8, 'points': 1},
    'P': {'count': 2, 'points': 3},
    'Q': {'count': 1, 'points': 10},
    'R': {'count': 6, 'points': 1},
    'S': {'count': 4, 'points': 1},
    'T': {'count': 6, 'points': 1},
    'U': {'count': 4, 'points': 1},
    'V': {'count': 2, 'points': 4},
    'W': {'count': 2, 'points': 4},
    'X': {'count': 1, 'points': 8},
    'Y': {'count': 2, 'points': 4},
    'Z': {'count': 1, 'points': 10},
    '*': {'count': 2, 'points': 0}  # Blank tiles
}

# Premium Square Positions
PREMIUM_SQUARES: Final[Dict[str, Set[Tuple[int, int]]]] = {
    'TRIPLE_WORD': {
        (0, 0), (0, 7), (0, 14),
        (7, 0), (7, 14),
        (14, 0), (14, 7), (14, 14)
    },
    'DOUBLE_WORD': {
        (1, 1), (1, 13), (2, 2), (2, 12),
        (3, 3), (3, 11), (4, 4), (4, 10),
        (10, 4), (10, 10), (11, 3), (11, 11),
        (12, 2), (12, 12), (13, 1), (13, 13)
    },
    'TRIPLE_LETTER': {
        (1, 5), (1, 9), (5, 1), (5, 5),
        (5, 9), (5, 13), (9, 1), (9, 5),
        (9, 9), (9, 13), (13, 5), (13, 9)
    },
    'DOUBLE_LETTER': {
        (0, 3), (0, 11), (2, 6), (2, 8),
        (3, 0), (3, 7), (3, 14), (6, 2),
        (6, 6), (6, 8), (6, 12), (7, 3),
        (7, 11), (8, 2), (8, 6), (8, 8),
        (8, 12), (11, 0), (11, 7), (11, 14),
        (12, 6), (12, 8), (14, 3), (14, 11)
    }
}

# Visual Constants
COLORS: Final[Dict[str, Dict[str, str]]] = {
    'BOARD': {
        'BACKGROUND': '#F5F5F5',
        'GRID': '#333333',
        'REGULAR': '#E8DCC4',
        'TRIPLE_WORD': '#FF6B6B',
        'DOUBLE_WORD': '#FFB4B4',
        'TRIPLE_LETTER': '#6B9AFF',
        'DOUBLE_LETTER': '#B4D4FF'
    },
    'TILE': {
        'BACKGROUND': '#F7D8A5',
        'BLANK': '#E8E8E8',
        'TEXT': '#000000',
        'SCORE': '#666666',
        'RECENT': '#FFD700'
    }
}

# Drawing Constants
DIMENSIONS: Final[Dict[str, int]] = {
    'CELL_SIZE': 50,
    'PADDING': 30,
    'GRID_LINE': 2,
    'TILE_BORDER': 3,
    'RACK_TILE_SIZE': 40,
    'RACK_PADDING': 10,
    'FONT_LETTER': 28,
    'FONT_SCORE': 14,
    'FONT_COORDINATE': 16,
    'FONT_PREMIUM': 12
}

# Error Messages
ERROR_MESSAGES: Final[Dict[str, str]] = {
    'INVALID_WORD': "Word '{word}' is not in the dictionary",
    'INVALID_WORDS': "Invalid word(s) formed: {words}",
    'INVALID_CROSSWORD': "Forms invalid crossword '{word}' at position {pos}",
    'NO_CENTER': "First word must cross the center square (H8)",
    'NO_CONNECTION': "Word must connect to existing words on the board",
    'OUT_OF_BOUNDS': "Word placement would go out of bounds",
    'INSUFFICIENT_TILES': "Missing required tiles: {tiles}",
    'TILE_CONFLICT': "Position {pos} already contains different letter",
    'INVALID_MOVE_FORMAT': "Invalid move format. Use 'PLAY: WORD H8 ACROSS'",
    'NOT_YOUR_TURN': "Not your turn. Waiting for Player {player}",
    'NO_TILES_EXCHANGE': "Not enough tiles in bag for exchange",
    'INVALID_DIRECTION': "Direction must be ACROSS or DOWN"
}

# Coordinate System
COLUMNS: Final[List[str]] = [chr(i) for i in range(ord('A'), ord('A') + BOARD_SIZE)]
ROWS: Final[List[str]] = [str(i) for i in range(1, BOARD_SIZE + 1)]