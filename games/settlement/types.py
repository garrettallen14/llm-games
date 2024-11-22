"""Type definitions for the Settlement game"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from collections import defaultdict

class TerrainType(Enum):
    """Terrain types and their movement costs"""
    WATER = 0      # Impassable
    MOUNTAIN = 2   # 2x energy cost
    FOREST = 1.5   # 1.5x energy cost
    PLAINS = 1     # 1x energy cost
    FERTILE = 1    # 1x energy cost

class ResourceType(Enum):
    """Resource types and their collection costs"""
    WOOD = (5, 10)    # (energy_cost, units_per_collection)
    STONE = (8, 10)
    FOOD = (3, 10)
    WATER = (2, 10)

class BuildingType(Enum):
    """Building types and their properties"""
    HOUSE = auto()    # +5 energy regen
    FARM = auto()     # Food collection
    WELL = auto()     # Water collection
    STORAGE = auto()  # Resource storage
    MARKET = auto()   # Trading

class ActionType(Enum):
    """Available action types"""
    MOVE = auto()
    GATHER = auto()
    TRADE = auto()
    BUILD = auto()
    SAY = auto()
    ENTER_BUILDING = auto()

class Position:
    """A position in the game world"""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        return f"({self.x},{self.y})"
    
    def __repr__(self):
        return f"Position({self.x}, {self.y})"
    
    def distance_to(self, other: 'Position') -> int:
        """Manhattan distance to another position"""
        return abs(self.x - other.x) + abs(self.y - other.y)

@dataclass
class Resource:
    """Resource deposit in the world"""
    type: ResourceType
    amount: int
    max_amount: int = 200
    regen_rate: float = 0.1  # 10% daily regeneration

    def collect(self, amount: int) -> int:
        """Collect resources, returns actual amount collected"""
        collected = min(amount, self.amount)
        self.amount -= collected
        return collected

    def regenerate(self):
        """Regenerate resources"""
        regen_amount = int(self.max_amount * self.regen_rate)
        self.amount = min(self.max_amount, self.amount + regen_amount)

@dataclass
class Building:
    """Building in the world"""
    type: BuildingType
    position: Position
    owner_id: int
    inventory: Dict[ResourceType, int] = field(default_factory=dict)
    max_storage: int = 1000

@dataclass
class Agent:
    """An agent in the game world"""
    id: int
    name: str
    position: Position
    energy: int = 100
    max_energy: int = 100
    inventory: Dict[ResourceType, int] = field(default_factory=lambda: defaultdict(int))
    in_building: Optional['Building'] = None
    vision_radius: int = 10  # Reduced to 5 at night

    def can_act(self, cost: int) -> bool:
        """Check if agent has enough energy for action"""
        return self.energy >= cost

    def spend_energy(self, amount: int):
        """Spend energy for action"""
        self.energy = max(0, self.energy - amount)

    def regenerate_energy(self, amount: int):
        """Regenerate energy"""
        self.energy = min(self.max_energy, self.energy + amount)

    def has_resources_for_building(self) -> bool:
        """Check if agent has enough resources to build any type of building"""
        # Basic building costs
        BUILDING_COSTS = {
            BuildingType.HOUSE: {ResourceType.WOOD: 5, ResourceType.STONE: 3},
            BuildingType.FARM: {ResourceType.WOOD: 3, ResourceType.STONE: 2},
            BuildingType.WELL: {ResourceType.STONE: 5},
            BuildingType.STORAGE: {ResourceType.WOOD: 4, ResourceType.STONE: 4},
            BuildingType.MARKET: {ResourceType.WOOD: 6, ResourceType.STONE: 6}
        }

        # Check if we have enough resources for any building type
        for building_costs in BUILDING_COSTS.values():
            has_resources = True
            for resource_type, amount in building_costs.items():
                if self.inventory.get(resource_type, 0) < amount:
                    has_resources = False
                    break
            if has_resources:
                return True
        return False

@dataclass
class WorldConfig:
    """World configuration"""
    size: int = 500
    water_body_ratio: float = 0.2
    mountain_ratio: float = 0.1
    forest_ratio: float = 0.3
    resource_density: float = 0.1
    max_resource_amount: int = 200
