"""World implementation for Settlement game"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
import random
from dataclasses import dataclass
import json
from collections import defaultdict
import heapq

from games.settlement.types import (
    TerrainType, ResourceType, BuildingType, ActionType,
    Position, Resource, Building, Agent, WorldConfig
)

class World:
    """Main world class handling game state and mechanics"""
    
    def __init__(self, config: WorldConfig):
        """Initialize the world with given configuration"""
        self.config = config
        self.size = config.size
        
        # Initialize layers
        self.terrain = np.full((self.size, self.size), TerrainType.PLAINS.value, dtype=int)
        self.resources: Dict[Position, Resource] = {}
        self.buildings: Dict[Position, Building] = {}
        self.agents: Dict[int, Agent] = {}
        self.agent_positions: Dict[Position, int] = {}
        
        # Time system
        self.current_turn = 0
        self.current_day = 0
        
        # Message log
        self.messages: List[Dict[str, Any]] = []
        
        # Generate initial world
        self._generate_terrain()
        self._generate_resources()
        self._generate_spawn_points()
        
    def _generate_terrain(self):
        """Generate world terrain using noise-based approach"""
        # Generate base noise for terrain
        noise = np.random.random((self.size, self.size))
        
        # Water bodies (lowest 20%)
        water_mask = noise < self.config.water_body_ratio
        self.terrain[water_mask] = TerrainType.WATER.value
        
        # Mountains (highest 10%)
        mountain_threshold = 1.0 - self.config.mountain_ratio
        mountain_mask = noise > mountain_threshold
        self.terrain[mountain_mask] = TerrainType.MOUNTAIN.value
        
        # Forests (30% of remaining)
        forest_mask = (noise > self.config.water_body_ratio) & (noise < mountain_threshold)
        forest_cells = np.random.choice(
            [0, 1], 
            size=forest_mask.sum(), 
            p=[0.7, 0.3]
        )
        self.terrain[forest_mask] = np.where(
            forest_cells, 
            TerrainType.FOREST.value,
            TerrainType.PLAINS.value
        )
        
        # Fertile land (random 10% of plains)
        plains_mask = self.terrain == TerrainType.PLAINS.value
        fertile_cells = np.random.choice(
            [0, 1], 
            size=plains_mask.sum(), 
            p=[0.9, 0.1]
        )
        self.terrain[plains_mask] = np.where(
            fertile_cells,
            TerrainType.FERTILE.value,
            TerrainType.PLAINS.value
        )
    
    def _generate_resources(self):
        """Generate initial resource deposits"""
        resource_probs = {
            TerrainType.WATER: {ResourceType.WATER: 0.8},
            TerrainType.MOUNTAIN: {ResourceType.STONE: 0.6},
            TerrainType.FOREST: {ResourceType.WOOD: 0.7, ResourceType.FOOD: 0.3},
            TerrainType.PLAINS: {ResourceType.FOOD: 0.4},
            TerrainType.FERTILE: {ResourceType.FOOD: 0.8}
        }
        
        for x in range(self.size):
            for y in range(self.size):
                terrain = TerrainType(self.terrain[x, y])
                if terrain not in resource_probs:
                    continue
                    
                for res_type, prob in resource_probs[terrain].items():
                    if random.random() < prob * self.config.resource_density:
                        pos = Position(x, y)
                        amount = random.randint(50, self.config.max_resource_amount)
                        self.resources[pos] = Resource(
                            type=res_type,
                            amount=amount,
                            max_amount=self.config.max_resource_amount
                        )
    
    def _generate_spawn_points(self):
        """Generate valid spawn points for agents"""
        self.spawn_points = []
        
        # Find all valid spawn locations (plains or fertile, not near water)
        for x in range(self.size):
            for y in range(self.size):
                if self.terrain[x, y] in [TerrainType.PLAINS.value, TerrainType.FERTILE.value]:
                    # Check surrounding tiles for water
                    valid = True
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.size and 
                                0 <= ny < self.size and 
                                self.terrain[nx, ny] == TerrainType.WATER.value):
                                valid = False
                                break
                    if valid:
                        self.spawn_points.append(Position(x, y))
        
        # Shuffle spawn points
        random.shuffle(self.spawn_points)
    
    def update_time(self):
        """Update game time and handle time-based events"""
        self.current_turn = (self.current_turn + 1) % 24
        if self.current_turn == 0:
            self.current_day += 1
            self._daily_update()
    
    def _daily_update(self):
        """Handle daily updates"""
        # Regenerate resources
        for resource in self.resources.values():
            resource.regenerate()
            
        # Generate new resource deposits
        self._spawn_new_resources()
    
    def _spawn_new_resources(self):
        """Spawn new resource deposits near existing ones"""
        new_resources = {}
        for pos, resource in self.resources.items():
            if random.random() < 0.2:  # 20% chance for new deposits
                # Try to spawn in adjacent tiles
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        new_x = pos.x + dx
                        new_y = pos.y + dy
                        
                        if (0 <= new_x < self.size and 
                            0 <= new_y < self.size and
                            self.terrain[new_x, new_y] != TerrainType.WATER.value):
                            
                            new_pos = Position(new_x, new_y)
                            if new_pos not in self.resources and new_pos not in new_resources:
                                amount = random.randint(50, self.config.max_resource_amount)
                                new_resources[new_pos] = Resource(
                                    type=resource.type,
                                    amount=amount,
                                    max_amount=self.config.max_resource_amount
                                )
        
        # Add new resources to world
        self.resources.update(new_resources)
    
    def is_night(self) -> bool:
        """Check if it's currently night time"""
        return 12 <= self.current_turn < 24
    
    def get_movement_cost(self, pos: Position) -> float:
        """Get movement cost for a position"""
        if not (0 <= pos.x < self.size and 0 <= pos.y < self.size):
            return float('inf')
        
        terrain_type = TerrainType(self.terrain[pos.x, pos.y])
        if terrain_type == TerrainType.WATER:
            return float('inf')
            
        base_cost = terrain_type.value
        return base_cost * (2 if self.is_night() else 1)
    
    def get_vision_radius(self) -> int:
        """Get current vision radius based on time of day"""
        return 5 if self.is_night() else 10

    def add_agent(self, name: str) -> int:
        """Add a new agent to the world"""
        if not self.spawn_points:
            raise ValueError("No spawn points available")
            
        agent_id = len(self.agents) + 1
        spawn_pos = self.spawn_points.pop()
        
        agent = Agent(
            id=agent_id,
            name=name,
            position=spawn_pos,
            energy=100,
            max_energy=100
        )
        
        self.agents[agent_id] = agent
        self.agent_positions[spawn_pos] = agent_id
        return agent_id
    
    def move_agent(self, agent_id: int, direction: str) -> Tuple[bool, str]:
        """Move agent in cardinal direction"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False, "Invalid agent ID"
            
        # Get new position
        dx, dy = {
            "north": (0, -1),
            "south": (0, 1),
            "east": (1, 0),
            "west": (-1, 0)
        }.get(direction.lower(), (0, 0))
        
        if dx == 0 and dy == 0:
            return False, "Invalid direction"
            
        new_x = agent.position.x + dx
        new_y = agent.position.y + dy
        new_pos = Position(new_x, new_y)
        
        # Check bounds and terrain
        cost = self.get_movement_cost(new_pos)
        if cost == float('inf'):
            return False, "Invalid move - impassable terrain or out of bounds"
            
        # Check energy
        if not agent.can_act(cost):
            return False, "Insufficient energy"
            
        # Check if position is occupied
        if new_pos in self.agent_positions:
            return False, "Position occupied by another agent"
            
        # Move agent
        del self.agent_positions[agent.position]
        agent.position = new_pos
        self.agent_positions[new_pos] = agent_id
        agent.spend_energy(cost)
        
        return True, "Move successful"
    
    def gather_resource(self, agent_id: int, resource_pos: Position) -> Tuple[bool, str]:
        """Gather resources from adjacent tile"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False, "Invalid agent ID"
            
        # Check if night
        if self.is_night():
            return False, "Cannot gather resources at night"
            
        # Check if resource exists
        resource = self.resources.get(resource_pos)
        if not resource:
            return False, "No resource at position"
            
        # Check adjacency
        if abs(agent.position.x - resource_pos.x) > 1 or abs(agent.position.y - resource_pos.y) > 1:
            return False, "Resource not adjacent"
            
        # Calculate collection amount and energy cost
        energy_cost, units_per_collection = resource.type.value
        max_collection = 50
        collection_amount = min(max_collection, resource.amount)
        total_energy_cost = (collection_amount // units_per_collection) * energy_cost
        
        # Check energy
        if not agent.can_act(total_energy_cost):
            return False, "Insufficient energy"
            
        # Collect resource
        collected = resource.collect(collection_amount)
        if resource.amount <= 0:
            del self.resources[resource_pos]
            
        # Add to inventory
        if resource.type not in agent.inventory:
            agent.inventory[resource.type] = 0
        agent.inventory[resource.type] += collected
        
        # Spend energy
        agent.spend_energy(total_energy_cost)
        
        return True, f"Gathered {collected} units of {resource.type.name}"
    
    def build(self, agent_id: int, building_type: BuildingType, position: Position) -> Tuple[bool, str]:
        """Build a new building"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False, "Invalid agent ID"
            
        # Check position validity
        if not (0 <= position.x < self.size and 0 <= position.y < self.size):
            return False, "Invalid position"
            
        # Check terrain
        terrain_type = TerrainType(self.terrain[position.x, position.y])
        if terrain_type in [TerrainType.WATER, TerrainType.MOUNTAIN]:
            return False, "Cannot build on this terrain"
            
        # Check if position is occupied
        if position in self.buildings or position in self.agent_positions:
            return False, "Position occupied"
            
        # Check energy
        build_cost = 20
        if not agent.can_act(build_cost):
            return False, "Insufficient energy"
            
        # Create building
        building = Building(
            type=building_type,
            position=position,
            owner_id=agent_id
        )
        self.buildings[position] = building
        
        # Spend energy
        agent.spend_energy(build_cost)
        
        return True, f"Built {building_type.name}"
    
    def trade(self, agent_id: int, target_id: int, offer: Dict[ResourceType, int], request: Dict[ResourceType, int]) -> Tuple[bool, str]:
        """Trade resources between agents"""
        agent = self.agents.get(agent_id)
        target = self.agents.get(target_id)
        if not agent or not target:
            return False, "Invalid agent ID"
            
        # Check adjacency
        if abs(agent.position.x - target.position.x) > 1 or abs(agent.position.y - target.position.y) > 1:
            return False, "Agents not adjacent"
            
        # Check energy
        trade_cost = 1
        if not agent.can_act(trade_cost):
            return False, "Insufficient energy"
            
        # Validate offer
        for res_type, amount in offer.items():
            if res_type not in agent.inventory or agent.inventory[res_type] < amount:
                return False, "Insufficient resources for offer"
                
        # Validate request
        for res_type, amount in request.items():
            if res_type not in target.inventory or target.inventory[res_type] < amount:
                return False, "Target has insufficient resources"
                
        # Execute trade
        for res_type, amount in offer.items():
            agent.inventory[res_type] -= amount
            if res_type not in target.inventory:
                target.inventory[res_type] = 0
            target.inventory[res_type] += amount
            
        for res_type, amount in request.items():
            target.inventory[res_type] -= amount
            if res_type not in agent.inventory:
                agent.inventory[res_type] = 0
            agent.inventory[res_type] += amount
            
        # Spend energy
        agent.spend_energy(trade_cost)
        
        return True, "Trade successful"
    
    def enter_building(self, agent_id: int, building_pos: Position) -> Tuple[bool, str]:
        """Enter a building"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False, "Invalid agent ID"
            
        # Check if building exists
        building = self.buildings.get(building_pos)
        if not building:
            return False, "No building at position"
            
        # Check adjacency
        if abs(agent.position.x - building_pos.x) > 1 or abs(agent.position.y - building_pos.y) > 1:
            return False, "Building not adjacent"
            
        # Check energy
        enter_cost = 1
        if not agent.can_act(enter_cost):
            return False, "Insufficient energy"
            
        # Enter building
        agent.in_building = building_pos
        agent.spend_energy(enter_cost)
        
        return True, f"Entered {building.type.name}"
    
    def say(self, agent_id: int, message: str) -> Tuple[bool, str]:
        """Broadcast message to nearby agents"""
        agent = self.agents.get(agent_id)
        if not agent:
            return False, "Invalid agent ID"
            
        # Add message to log
        self.messages.append({
            "turn": self.current_turn,
            "agent_id": agent_id,
            "message": message,
            "position": (agent.position.x, agent.position.y)
        })
        
        return True, "Message broadcast"
    
    def get_agent_state(self, agent_id: int) -> Dict[str, Any]:
        """Get complete state for an agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError("Invalid agent ID")
            
        # Get local view
        vision_radius = self.get_vision_radius()
        local_view = self._generate_local_view(agent.position, vision_radius)
        
        # Get visible agents and their messages
        visible_agents = self._get_visible_agents(agent_id)
        recent_messages = self._get_recent_messages(agent.position)
        
        # Get valid actions
        valid_actions = self.get_valid_actions(agent_id)
        
        return {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "position": {"x": agent.position.x, "y": agent.position.y},
                "energy": agent.energy,
                "inventory": {k.name: v for k, v in agent.inventory.items()},
                "in_building": agent.in_building._asdict() if agent.in_building else None
            },
            "world": {
                "time": {
                    "turn": self.current_turn,
                    "day": self.current_day,
                    "is_night": self.is_night()
                },
                "local_view": local_view,
                "visible_agents": visible_agents,
                "recent_messages": recent_messages
            },
            "valid_actions": valid_actions
        }
    
    def _generate_local_view(self, center: Position, radius: int) -> str:
        """Generate detailed ASCII representation of local area with coordinates and legend"""
        # First create coordinate header
        view = ['    ' + ''.join(f'{(center.x - radius + i):3d}' for i in range(2*radius + 1))]
        view.append('    ' + '---' * (2*radius + 1))
        
        for y in range(center.y - radius, center.y + radius + 1):
            row = [f'{y:3d}|']  # Add y-coordinate
            for x in range(center.x - radius, center.x + radius + 1):
                if not (0 <= x < self.size and 0 <= y < self.size):
                    row.append('OOB')  # Out of bounds
                else:
                    pos = Position(x, y)
                    if pos in self.agent_positions:
                        agent_id = self.agent_positions[pos]
                        row.append(f'A{agent_id:02d}')  # Agent with ID, zero-padded
                    elif pos in self.buildings:
                        building = self.buildings[pos]
                        building_codes = {
                            BuildingType.HOUSE: 'HSE',
                            BuildingType.FARM: 'FRM',
                            BuildingType.WELL: 'WEL',
                            BuildingType.STORAGE: 'STR',
                            BuildingType.MARKET: 'MKT'
                        }
                        row.append(building_codes[building.type])
                    elif pos in self.resources:
                        resource = self.resources[pos]
                        resource_codes = {
                            ResourceType.WOOD: 'WOD',
                            ResourceType.STONE: 'STN',
                            ResourceType.FOOD: 'FOD',
                            ResourceType.WATER: 'WTR'
                        }
                        row.append(resource_codes[resource.type])
                    else:
                        terrain = TerrainType(self.terrain[x, y])
                        terrain_codes = {
                            TerrainType.WATER: 'WAT',
                            TerrainType.MOUNTAIN: 'MTN',
                            TerrainType.FOREST: 'FOR',
                            TerrainType.PLAINS: 'PLN',
                            TerrainType.FERTILE: 'FRT'
                        }
                        row.append(terrain_codes[terrain])
            view.append(''.join(row))
        
        # Add legend at the bottom
        view.extend([
            '',
            'Legend:',
            'Terrain: WAT=Water MTN=Mountain FOR=Forest PLN=Plains FRT=Fertile',
            'Resources: WOD=Wood STN=Stone FOD=Food WTR=Water',
            'Buildings: HSE=House FRM=Farm WEL=Well STR=Storage MKT=Market',
            'Agents: A##=Agent(ID)',
            'OOB=Out of Bounds'
        ])
        
        return '\n'.join(view)
    
    def _get_visible_agents(self, agent_id: int) -> List[Dict[str, Any]]:
        """Get information about visible agents"""
        agent = self.agents[agent_id]
        vision_radius = self.get_vision_radius()
        visible = []
        
        for other_id, other in self.agents.items():
            if other_id == agent_id:
                continue
                
            dx = abs(other.position.x - agent.position.x)
            dy = abs(other.position.y - agent.position.y)
            if dx <= vision_radius and dy <= vision_radius:
                visible.append({
                    "id": other.id,
                    "name": other.name,
                    "position": {"x": other.position.x, "y": other.position.y}
                })
                
        return visible
    
    def _get_recent_messages(self, position: Position) -> List[Dict[str, Any]]:
        """Get recent messages visible from position"""
        vision_radius = self.get_vision_radius()
        recent_messages = []
        
        for msg in reversed(self.messages[-50:]):  # Last 50 messages
            msg_x, msg_y = msg["position"]
            if (abs(msg_x - position.x) <= vision_radius and 
                abs(msg_y - position.y) <= vision_radius):
                recent_messages.append(msg)
                
        return recent_messages
    
    def get_valid_actions(self, agent_id: int) -> List[str]:
        """Get list of valid actions for an agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return []
        
        actions = []
        
        # Movement is always valid if enough energy
        min_move_cost = min(terrain.value for terrain in TerrainType if terrain != TerrainType.WATER)
        if agent.can_act(min_move_cost):
            actions.extend(["north", "south", "east", "west"])
        
        # Check for nearby resources to gather
        if not self.is_night():  # Can only gather during day
            for pos, resource in self.resources.items():
                if pos.distance_to(agent.position) <= 1:
                    actions.append(f"({pos.x},{pos.y})")
        
        # Check for valid build locations
        if agent.has_resources_for_building() and agent.can_act(20):  # Building costs 20 energy
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:  # Can't build on self
                        continue
                    build_pos = Position(agent.position.x + dx, agent.position.y + dy)
                    if self.is_valid_build_location(build_pos):
                        actions.extend([
                            f"house ({build_pos.x},{build_pos.y})",
                            f"farm ({build_pos.x},{build_pos.y})",
                            f"well ({build_pos.x},{build_pos.y})",
                            f"storage ({build_pos.x},{build_pos.y})",
                            f"market ({build_pos.x},{build_pos.y})"
                        ])
        
        # Check for nearby agents to trade with
        for other_id, other_agent in self.agents.items():
            if other_id != agent_id and other_agent.position.distance_to(agent.position) <= 1:
                actions.append(f"trade {other_id}")
        
        # Check for nearby buildings to enter
        if agent.can_act(5):  # Entering costs 5 energy
            for pos, building in self.buildings.items():
                if pos.distance_to(agent.position) <= 1:
                    actions.append(f"enter ({pos.x},{pos.y})")
        
        # Can always say something
        actions.append("say <message>")
        
        return actions
