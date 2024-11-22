"""Type definitions for the Color Chat game"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Set
import random

@dataclass
class Position:
    """A position in the game world"""
    x: int
    y: int
    
    def distance_to(self, other: 'Position') -> int:
        """Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)

@dataclass
class Agent:
    """An agent in the game world"""
    id: int
    position: Position
    color: Tuple[int, int, int] = (255, 255, 255)  # Default white
    last_message: str = ""

@dataclass
class WorldConfig:
    """World configuration"""
    size: int = 10  # Fixed 10x10 grid
    communication_radius: int = 2  # Can hear messages within 2 blocks
