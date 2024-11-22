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
    size: int = 50  # Default 50x50 grid
    communication_radius: int = 5  # Default communication radius
    
    def __post_init__(self):
        """Validate configuration"""
        if self.size < 5:
            raise ValueError("Grid size must be at least 5x5")
        if self.size > 100:
            raise ValueError("Grid size cannot exceed 100x100 for performance reasons")
        if self.communication_radius < 1:
            raise ValueError("Communication radius must be at least 1")
        if self.communication_radius > self.size // 2:
            raise ValueError(f"Communication radius cannot exceed half the grid size ({self.size // 2})")
