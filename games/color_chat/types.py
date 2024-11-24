"""Type definitions for the Color Chat game"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Set
import random
from games.color_chat.constants import MAX_WOOD_PER_TILE

class TerrainType(Enum):
    """Types of terrain in the game world"""
    EMPTY = "empty"
    WATER = "water"
    TREE = "tree"
    SHELTER = "shelter"

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
    """An agent in the game world with energy levels and inventory.
    
    Agents start with 100 energy. Energy decreases by 1 for each move,
    but increases by 1 when near other agents. Can carry up to 1 wood."""
    id: int
    position: Position
    color: Tuple[int, int, int] = (255, 255, 255)  # Default white
    last_message: str = ""
    energy: int = 100  # Default starting energy
    wood_inventory: int = 0  # Amount of wood being carried (max 1)

@dataclass
class Shelter:
    """Represents a shelter in the game world"""
    name: str
    owner_id: int
    position: Position
    wood_storage: int = 0

@dataclass
class WorldConfig:
    """World configuration"""
    size: int = 50  # Default 50x50 grid
    communication_radius: int = 5  # Default communication radius
    water_tiles: Set[Tuple[int, int]] = field(default_factory=set)  # Positions of water tiles
    tree_tiles: Set[Tuple[int, int]] = field(default_factory=set)  # Positions of tree tiles
    wood_storage: Dict[Tuple[int, int], int] = field(default_factory=dict)  # Wood stored at each position
    
    def __post_init__(self):
        """Validate configuration and initialize resources"""
        if self.size < 5:
            raise ValueError("Grid size must be at least 5x5")
        if self.size > 100:
            raise ValueError("Grid size cannot exceed 100x100 for performance reasons")
        if self.communication_radius < 1:
            raise ValueError("Communication radius must be at least 1")
        if self.communication_radius > self.size // 2:
            raise ValueError(f"Communication radius cannot exceed half the grid size ({self.size // 2})")
        
        # Initialize wood on tree tiles
        for tree_pos in self.tree_tiles:
            if tree_pos not in self.wood_storage:
                self.wood_storage[tree_pos] = MAX_WOOD_PER_TILE
