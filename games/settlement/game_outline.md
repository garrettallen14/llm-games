# Settlement Game Implementation Guide

## Implementation Steps

### Step 1: Core Data Structures
```python
# games/settlement/game.py

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from enum import Enum
import numpy as np
import random
from pathlib import Path
import json
from PIL import Image, ImageDraw
import io
import base64

# Implement enums and dataclasses as specified in Game-Specific Data section
```

### Step 2: World Class Implementation
```python
class SettlementGame:
    def __init__(self, run_dir: Path, max_turns: int = 100):
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.world = World(size=500)
        self.current_turn = 0
        self.game_over = False
        self.logs = []
    
    def get_system_prompt(self) -> Dict[str, str]:
        """Return the game's system prompt for LLM players"""
        return {
            "role": "system",
            "content": """You are an agent in a settlement-building simulation. Your goal is to survive and thrive by:
1. Gathering resources (wood, stone, food, water)
2. Building structures (houses, farms, wells, storage, markets)
3. Trading with other agents
4. Managing your energy levels
5. Developing relationships

Available actions:
[Detailed action list with formats from Action Specifications section]

You must respond with a JSON object containing:
{
    "action_type": "<action_type>",
    "parameters": {<action_specific_parameters>},
    "reasoning": "Brief explanation of your decision"
}"""
        }
    
    def run(self, players: List['BaseLLMPlayer']) -> Dict:
        """Main game loop"""
        # Initialize world with players
        for idx, player in enumerate(players):
            self.world.add_agent(name=f"Agent_{player.player_id}")
        
        while not self.game_over and self.current_turn < self.max_turns:
            # Process each player's turn
            for player in players:
                # Generate game state
                state = self.world.get_agent_state(player.player_id)
                valid_actions = self.world.get_valid_actions(player.player_id)
                
                # Generate world visualization
                game_image = self._generate_game_image()
                
                # Get player's action
                message = {
                    "role": "user",
                    "content": f"Current Game State:\n{json.dumps(state, indent=2)}\n\nValid Actions:\n{json.dumps(valid_actions, indent=2)}\n\nWhat is your next action?"
                }
                
                response = player.get_response(message, game_image)
                
                try:
                    # Parse and execute action
                    action = json.loads(response)
                    success, message = self.world.execute_action(player.player_id, action)
                    
                    # Log action result
                    self.logs.append({
                        "turn": self.current_turn,
                        "player": player.player_id,
                        "action": action,
                        "success": success,
                        "message": message
                    })
                    
                except Exception as e:
                    print(f"Error processing action for player {player.player_id}: {str(e)}")
            
            # Update world state
            self.world.update_time()
            self.current_turn += 1
            
            # Check win conditions
            self.game_over = self._check_game_over()
        
        return self._generate_game_result()
    
    def _generate_game_image(self) -> str:
        """Generate a visual representation of the game state"""
        # Create image
        img = Image.new('RGB', (4000, 4000), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw grid
        cell_size = 8  # 4000/500
        for i in range(501):
            draw.line([(i*cell_size, 0), (i*cell_size, 4000)], fill='gray')
            draw.line([(0, i*cell_size), (4000, i*cell_size)], fill='gray')
        
        # Draw game elements
        # [Implementation of drawing resources, buildings, and agents]
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    
    def _check_game_over(self) -> bool:
        """Check if game ending conditions are met"""
        # [Implementation of win/loss conditions]
        return False
    
    def _generate_game_result(self) -> Dict:
        """Generate final game results"""
        return {
            "turns_played": self.current_turn,
            "game_over": self.game_over,
            "logs": self.logs,
            # Add other relevant statistics
        }
```

### Step 3: World Class Implementation
```python
class World:
    """Implement as specified in the game outline"""
    def __init__(self, config: WorldConfig):
        self.config = config
        self.size = config.size
        
        # Initialize layers
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.terrain_layer = np.zeros((self.size, self.size), dtype=int)
        self.resource_layer: Dict[Tuple[int, int], Tuple[ResourceType, int]] = {}
        self.building_layer: Dict[Tuple[int, int], Building] = {}
        self.agent_layer: Dict[Tuple[int, int], Agent] = {}
        
        # Game state
        self.time = 0
        self.day = 0
        self.agents: Dict[int, Agent] = {}
        self.messages: List[Dict] = []
        
        # Generate world
        self._generate_terrain()
        self._generate_resources()
        self._setup_spawn_points()

    def _generate_terrain(self):
        """Generate base terrain using Perlin noise"""
        scale = 50.0
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        
        # Generate base noise
        noise = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                noise[i][j] = snoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity)
        
        # Normalize to 0-1
        noise = (noise - noise.min()) / (noise.max() - noise.min())
        
        # Assign terrain types based on noise values
        self.terrain_layer = np.zeros_like(noise, dtype=int)
        
        # Water bodies (lowest areas)
        water_mask = noise < self.config.water_body_ratio
        self.terrain_layer[water_mask] = TerrainType.WATER.value
        
        # Mountains (highest areas)
        mountain_threshold = 1 - self.config.mountain_ratio
        mountain_mask = noise > mountain_threshold
        self.terrain_layer[mountain_mask] = TerrainType.MOUNTAIN.value
        
        # Forests (medium-high areas)
        forest_threshold = mountain_threshold - self.config.forest_ratio
        forest_mask = (noise > forest_threshold) & (noise <= mountain_threshold)
        self.terrain_layer[forest_mask] = TerrainType.FOREST.value
        
        # Fertile land (medium-low areas)
        fertile_threshold = self.config.water_body_ratio + self.config.fertile_ratio
        fertile_mask = (noise >= self.config.water_body_ratio) & (noise < fertile_threshold)
        self.terrain_layer[fertile_mask] = TerrainType.FERTILE.value
        
        # Rest is plains
        plains_mask = ~(water_mask | mountain_mask | forest_mask | fertile_mask)
        self.terrain_layer[plains_mask] = TerrainType.PLAINS.value

    def _generate_resources(self):
        """Generate resources based on terrain"""
        # Resource generation probabilities per terrain type
        resource_probs = {
            TerrainType.WATER: {ResourceType.WATER: 0.8},
            TerrainType.MOUNTAIN: {ResourceType.STONE: 0.7},
            TerrainType.FOREST: {ResourceType.WOOD: 0.6},
            TerrainType.FERTILE: {ResourceType.FOOD: 0.5},
            TerrainType.PLAINS: {
                ResourceType.WOOD: 0.2,
                ResourceType.STONE: 0.2,
                ResourceType.FOOD: 0.3
            }
        }
        
        def generate_cluster(x: int, y: int, resource: ResourceType):
            """Generate a cluster of resources around a point"""
            cluster_size = self.config.resource_cluster_size
            for dx in range(-cluster_size, cluster_size + 1):
                for dy in range(-cluster_size, cluster_size + 1):
                    # Probability decreases with distance from center
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > cluster_size:
                        continue
                        
                    prob = 1 - (dist / cluster_size)
                    if random.random() > prob:
                        continue
                        
                    new_x, new_y = x + dx, y + dy
                    if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                        continue
                        
                    pos = (new_x, new_y)
                    if pos not in self.resource_layer:
                        # Amount varies with distance from center
                        base_amount = random.randint(50, 200)
                        amount = int(base_amount * (1 - dist/cluster_size))
                        self.resource_layer[pos] = (resource, amount)
        
        # Generate resources based on terrain
        for x in range(self.size):
            for y in range(self.size):
                terrain = TerrainType(self.terrain_layer[x, y])
                probs = resource_probs[terrain]
                
                for resource_type, prob in probs.items():
                    if random.random() < prob:
                        generate_cluster(x, y, resource_type)

    def _setup_spawn_points(self):
        """Setup valid spawn points for agents"""
        self.spawn_points = []
        
        # Find suitable spawn locations (plains, not too close to resources)
        min_distance = 5  # Minimum distance between spawn points
        
        def is_valid_spawn(x: int, y: int) -> bool:
            # Check if location is plains and not too close to resources or other spawns
            if self.terrain_layer[x, y] != TerrainType.PLAINS.value:
                return False
                
            # Check distance to resources
            for rx, ry in self.resource_layer.keys():
                if abs(x - rx) + abs(y - ry) < min_distance:
                    return False
                    
            # Check distance to other spawn points
            for sx, sy in self.spawn_points:
                if abs(x - sx) + abs(y - sy) < min_distance:
                    return False
                    
            return True
        
        # Find spawn points using spiral pattern
        def spiral_coords():
            x = y = 0
            dx = 0
            dy = -1
            for i in range(max(self.size, self.size)**2):
                if (-self.size/2 < x <= self.size/2) and (-self.size/2 < y <= self.size/2):
                    yield (x + self.size//2, y + self.size//2)
                if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
                    dx, dy = -dy, dx
                x, y = x+dx, y+dy
        
        # Find first 100 valid spawn points
        for x, y in spiral_coords():
            if len(self.spawn_points) >= 100:  # Maximum number of spawn points
                break
            if 0 <= x < self.size and 0 <= y < self.size and is_valid_spawn(x, y):
                self.spawn_points.append((x, y))

    def add_agent(self, name: str = None) -> int:
        """Add a new agent to the world at a spawn point"""
        agent_id = len(self.agents) + 1
        
        # Get random spawn point and remove it
        if not self.spawn_points:
            raise ValueError("No more spawn points available")
        spawn_pos = self.spawn_points.pop(random.randrange(len(self.spawn_points)))
        
        # Create agent
        agent = Agent(
            id=agent_id,
            name=name or f"Agent_{agent_id}",
            position=spawn_pos,
            inventory={resource: 0 for resource in ResourceType},
            energy=100
        )
        
        # Add to world
        self.agents[agent_id] = agent
        self.agent_layer[spawn_pos] = agent
        self.grid[spawn_pos[0], spawn_pos[1]] = agent_id
        
        return agent_id
```

### Step 4: Testing Setup
```python
# games/settlement/test_game.py

def test_basic_game_flow():
    """Test basic game initialization and flow"""
    
def test_action_execution():
    """Test each action type"""
    
def test_state_generation():
    """Test state representation"""
    
def test_win_conditions():
    """Test game ending conditions"""
```

### Step 5: Required Dependencies
```
# games/settlement/requirements.txt
numpy
Pillow
```

### Integration Notes:
1. The game class name must be "SettlementGame" to match runner.py naming convention
2. All actions must return (success: bool, message: str)
3. State representations must match the LLM input format
4. Image generation is optional but enhances LLM understanding
5. Logging should be comprehensive for debugging

### Error Handling:
1. Validate all LLM responses
2. Provide clear error messages
3. Handle invalid actions gracefully
4. Maintain game state consistency
5. Log all errors for debugging

### Performance Considerations:
1. Cache frequently accessed data
2. Optimize image generation
3. Limit state representation size
4. Efficient pathfinding implementation
5. Memory management for long games

# Settlement Game Design Document

## Game Overview
A multi-agent simulation where LLM-powered agents build and manage a settlement in a resource-rich world. Agents must gather resources, construct buildings, trade with each other, and maintain relationships while managing their energy levels and inventories.

## Core Game Components

### 1. World Environment
- Grid Size: 500x500 cells
- Time System: Day/night cycle affecting agent activities
- Cell Types:
  - Empty (0)
  - Resource Deposit
  - Building
  - Agent

### 2. Resources
Types:
- WOOD: Used for basic construction
- STONE: Used for advanced buildings
- FOOD: Required for agent energy
- WATER: Required for agent survival

Resource Mechanics:
- Randomly distributed across map
- Varying quantities (50-200 units per deposit)
- Renewable over time
- Can be stored in buildings or agent inventory

### 3. Buildings
Types:
- HOUSE: Provides rest and energy recovery
- FARM: Produces food over time
- WELL: Provides water
- STORAGE: Increases resource storage capacity
- MARKET: Enables trading between agents

Building Properties:
- Type
- Position (x, y)
- Owner ID
- Durability (100 max, decreases over time)
- Storage capacity
- Construction requirements (resource costs)

### 4. Agents
Properties:
- Unique ID
- Name
- Position (x, y)
- Energy (100 max)
- Inventory (resource storage)
- Relationships (scores with other agents)
- Memory (last 50 events/interactions)
- Profession (specialization)
- Home location

### 5. Actions
Movement:
- Move in any direction within grid
- Energy cost per movement
- Pathfinding around obstacles

Resource Collection:
- Gather from deposits
- Energy cost per collection
- Inventory capacity limits
- Collection speed based on profession

Building:
- Place buildings on empty cells
- Require specific resources
- Energy cost for construction
- Limited by proximity rules

Trading:
- Offer resources to other agents
- Request resources in return
- Affected by relationship scores
- Market buildings boost trade efficiency

Social:
- Send messages to other agents
- Build/damage relationships
- Form alliances
- Share information

## Core Game Interface
Required interfaces for runner.py integration:
```python
get_system_prompt():
    - Return comprehensive game rules
    - Include action descriptions
    - Define resource and building mechanics
    - Specify win conditions
    - Format: Dict[str, str] with role and content

run(players: List[BaseLLMPlayer]):
    - Initialize game world
    - Main game loop implementation
    - Turn management
    - State updates
    - Win condition checking
    - Return: Dict with game results

get_state(player_id: int):
    - Generate complete game state
    - Include visible environment
    - Format agent-specific view
    - Include recent memories
    - Return: Dict matching State Input Format

execute_action(player_id: int, action: Dict):
    - Validate action possibility
    - Check resource requirements
    - Process action effects
    - Update game state
    - Return: Tuple[bool, str] success and message

get_valid_actions(player_id: int):
    - Generate available actions
    - Consider agent state
    - Check resource requirements
    - Validate building possibilities
    - Return: Dict[str, List] of action types
```

### Multimodal Integration
Game state visualization support:
- World grid representation (500x500)
- Resource deposit indicators
- Building placement visualization
- Agent position markers
- Time of day indication

Image handling:
- Base64 encoded PNG format
- 4000x4000 pixel resolution
- Color coding for different elements:
  - Resources: Unique colors per type
  - Buildings: Distinct shapes and colors
  - Agents: Highlighted positions
  - Grid: Light background with grid lines

### Game-Specific Data
Custom data structures:
```python
ResourceType(Enum):
    WOOD = "wood"
    STONE = "stone"
    FOOD = "food"
    WATER = "water"

BuildingType(Enum):
    HOUSE = "house"
    FARM = "farm"
    WELL = "well"
    STORAGE = "storage"
    MARKET = "market"

@dataclass
class Building:
    type: BuildingType
    position: Tuple[int, int]
    owner_id: int
    durability: int
    storage: Dict[ResourceType, int]

@dataclass
class Agent:
    id: int
    name: str
    position: Tuple[int, int]
    inventory: Dict[ResourceType, int]
    relationships: Dict[int, int]
    memory: List[str]
    profession: str
    home: Tuple[int, int]
    energy: int
```

## Core Functions

### World Management
```pseudocode
update_time():
    - Progress day/night cycle
    - Update resource regeneration
    - Process building decay
    - Manage agent energy cycles

spawn_resources():
    - Randomly add new resource deposits
    - Balance resource distribution
    - Scale quantities based on agent population

update_building_states():
    - Reduce building durability over time
    - Process resource production (farms, wells)
    - Check for building collapse
    - Update storage contents
```

### Agent Actions
```pseudocode
move_agent(agent_id, direction):
    - Validate movement possibility
    - Update agent position
    - Consume energy
    - Update agent memory

gather_resource(agent_id, resource_pos):
    - Check resource availability
    - Verify agent proximity
    - Transfer resources to inventory
    - Update resource deposit quantity
    - Consume energy

build_structure(agent_id, building_type, position):
    - Validate building requirements
    - Check resource availability
    - Consume resources
    - Place building
    - Update agent memory

trade_resources(agent_id, other_id, offer, request):
    - Verify both agents' inventories
    - Check relationship score
    - Transfer resources
    - Update relationship scores
    - Log transaction in memory

send_message(agent_id, other_id, message):
    - Deliver message to recipient
    - Update relationship scores
    - Add to both agents' memories
```

### State Management
```pseudocode
get_nearby_entities(position, radius):
    - Return all agents, buildings, and resources
    - Include distance and direction
    - Filter by visibility rules

calculate_path(start_pos, end_pos):
    - Find optimal route avoiding obstacles
    - Consider energy efficiency
    - Return waypoints

update_relationships(agent_id, other_id, delta):
    - Modify relationship score
    - Apply event-based modifiers
    - Update both agents' memories

manage_inventory(agent_id, resource_type, amount):
    - Add/remove resources
    - Check capacity limits
    - Trigger inventory-full notifications
```

### Game Logic
```pseudocode
validate_action(agent_id, action):
    - Check action possibility
    - Verify resource/energy requirements
    - Consider time of day
    - Return success/failure with reason

process_turn(agent_id):
    - Get agent's current state
    - Request action from LLM
    - Validate and execute action
    - Update world state
    - Generate memory entry

check_win_conditions():
    - Evaluate settlement prosperity
    - Check resource sustainability
    - Assess social harmony
    - Monitor building stability

generate_state_description(agent_id):
    - Compile visible environment
    - Include inventory status
    - List nearby entities
    - Summarize recent memories
```

## Action Specifications

#### 1. Movement Actions
```json
{
    "action_type": "move",
    "parameters": {
        "direction": "<north|south|east|west>"
    }
}
```
- Energy cost: 5
- Validation:
  - Target cell must be empty
  - Must be within grid bounds
  - Agent must have sufficient energy

#### 2. Resource Collection Actions
```json
{
    "action_type": "gather",
    "parameters": {
        "position": [x, y],
        "resource_type": "<wood|stone|food|water>"
    }
}
```
- Energy cost: 15
- Validation:
  - Resource must exist at position
  - Agent must be adjacent to resource
  - Agent must have inventory space
- Collection amount: 10 units per action

#### 3. Building Actions
```json
{
    "action_type": "build",
    "parameters": {
        "building_type": "<house|farm|well|storage|market>",
        "position": [x, y]
    }
}
```
- Energy cost: 20
- Resource costs:
  - House: 5 wood, 5 stone
  - Farm: 8 wood, 3 stone
  - Well: 3 wood, 8 stone
  - Storage: 10 wood, 10 stone
  - Market: 15 wood, 15 stone
- Validation:
  - Position must be empty
  - Agent must have required resources
  - Must be within build range (3 cells)

#### 4. Trading Actions
```json
{
    "action_type": "trade",
    "parameters": {
        "with_id": agent_id,
        "offer": {
            "resource_type": amount,
            ...
        },
        "request": {
            "resource_type": amount,
            ...
        }
    }
}
```
- Energy cost: 5
- Validation:
  - Both agents must be within trade range (3 cells)
  - Offering agent must have offered resources
  - Requesting agent must have requested resources
  - Trade must be balanced (market buildings provide 20% discount)

#### 5. Communication Actions
```json
{
    "action_type": "communicate",
    "parameters": {
        "to_id": agent_id,
        "message": "message_content",
        "type": "<propose_trade|share_info|request_help|social>"
    }
}
```
- Energy cost: 2
- Message types:
  - propose_trade: Initiate trade negotiation
  - share_info: Share resource/building locations
  - request_help: Ask for resources or assistance
  - social: General relationship building
- Validation:
  - Target agent must be within communication range (5 cells)
  - Message length limit: 200 characters

#### 6. Building Interaction Actions
```json
{
    "action_type": "interact_building",
    "parameters": {
        "building_position": [x, y],
        "interaction_type": "<store|retrieve|rest|maintain>",
        "resources": {
            "resource_type": amount,
            ...
        }
    }
}
```
- Energy costs:
  - store/retrieve: 5
  - rest: 0 (recovers energy)
  - maintain: 10
- Validation:
  - Must be adjacent to building
  - Must have permission (owner or public building)
  - Storage capacity limits apply
- Effects:
  - rest: Recover 20 energy per turn
  - maintain: Repair 10 durability points

#### 7. Profession Actions
```json
{
    "action_type": "profession",
    "parameters": {
        "action": "<specialize|train|collaborate>",
        "profession": "<builder|farmer|trader|explorer>",
        "target_id": agent_id  // for collaborate action
    }
}
```
- Energy cost: 10
- Profession bonuses:
  - Builder: -25% building cost
  - Farmer: +50% food gathering
  - Trader: +20% trade efficiency
  - Explorer: +2 visibility range
- Validation:
  - Can only specialize once per day
  - Training requires a mentor (agent with same profession)
  - Collaboration requires compatible professions

#### 8. Time-Specific Actions
```json
{
    "action_type": "time_action",
    "parameters": {
        "action": "<rest|plan|scout>"
    }
}
```
- Energy impacts:
  - rest: Recover all energy (night only)
  - plan: No cost (strategic bonus for next day)
  - scout: 10 energy (increased visibility at night)
- Time restrictions:
  - rest: Night only (time 18-06)
  - plan: Any time
  - scout: Night only (time 18-06)

## LLM Integration

### System Prompt Components
1. World rules and mechanics
2. Available actions and their requirements
3. Resource management guidelines
4. Building system explanation
5. Social interaction framework
6. Win conditions and objectives

### Expected LLM Output Format
```json
{
    "action_type": "<move|gather|build|trade|message>",
    "parameters": {
        // Action-specific parameters
    },
    "reasoning": "Explanation for the action choice"
}
```

### State Input Format
```json
{
    "agent": {
        "id": "unique_id",
        "position": [x, y],
        "inventory": {"resource_type": amount},
        "energy": current_energy,
        "relationships": {"other_id": score}
    },
    "visible_environment": {
        "nearby_agents": [...],
        "nearby_buildings": [...],
        "nearby_resources": [...],
        "time_of_day": current_time
    },
    "recent_memories": [
        "memory_1",
        "memory_2",
        ...
    ]
}
```

## Visualization and State Representation

### 1. Map Structure
The game world is represented as a 500x500 2D grid where each cell can contain:
- Empty space (0)
- Resource deposit (R)
- Building (B)
- Agent (A)

Example of internal grid representation:
```python
self.grid = np.zeros((500, 500), dtype=int)  # Base grid for entities
self.resource_layer = {}  # Dict[(x,y), (ResourceType, amount)]
self.building_layer = {}  # Dict[(x,y), Building]
self.agent_layer = {}    # Dict[(x,y), Agent]
```

### 2. Agent Local View
Each agent receives a 10x10 ASCII representation of their immediate surroundings in their state description:

```python
def get_agent_local_view(self, agent_id: int) -> str:
    """Generate ASCII representation of agent's surroundings"""
    agent = self.agents[agent_id]
    x, y = agent.position
    view_size = 5  # Results in 11x11 view (5 cells in each direction + current cell)
    
    # Initialize empty view
    view = []
    for dy in range(-view_size, view_size + 1):
        row = []
        for dx in range(-view_size, view_size + 1):
            new_x, new_y = x + dx, y + dy
            
            # Check bounds
            if not (0 <= new_x < self.size and 0 <= new_y < self.size):
                row.append('?')  # Out of bounds
                continue
                
            pos = (new_x, new_y)
            
            # Layer priority: Agent > Building > Resource > Empty
            if pos in self.agent_layer:
                agent = self.agent_layer[pos]
                row.append(f'A{agent.id}')  # A1, A2, etc.
            elif pos in self.building_layer:
                building = self.building_layer[pos]
                row.append(building.type.value[0].upper())  # H,F,W,S,M
            elif pos in self.resource_layer:
                resource_type, _ = self.resource_layer[pos]
                row.append(resource_type.value[0].upper())  # W,S,F,W
            else:
                row.append('.')  # Empty space
        
        view.append(' '.join(row))
    
    # Add compass rose and legend
    view.insert(0, "    N    ")
    view.insert(1, "  W + E  ")
    view.insert(2, "    S    ")
    view.append("")
    view.append("Legend:")
    view.append("A# = Agent (#=id)")
    view.append("H = House")
    view.append("F = Farm")
    view.append("W = Well/Wood")
    view.append("S = Storage/Stone")
    view.append("M = Market")
    view.append(". = Empty")
    view.append("? = Out of bounds")
    
    return "\n".join(view)
```

Example ASCII view:
```
    N    
  W + E  
    S    
? ? ? ? ? ? ? ? ? ? ?
? . . W . . . S . . ?
? . A2 . . . . . . ?
? . . H . . F . . . ?
? W . . . A1 . . . ?
? . . . . . . M . . ?
? . S . . . . . . . ?
? . . . W . . . . . ?
? . . . . . . . F . ?
? . . . . . . . . . ?
? ? ? ? ? ? ? ? ? ? ?

Legend:
A# = Agent (#=id)
H = House
F = Farm
W = Well/Wood
S = Storage/Stone
M = Market
. = Empty
? = Out of bounds
```

### 3. Visual Representation
For multimodal LLMs, we generate a PNG image representation of the entire game world:

```python
def generate_game_image(self) -> str:
    """Generate visual representation of game state"""
    # Constants
    CELL_SIZE = 8  # Each grid cell is 8x8 pixels
    GRID_SIZE = 500
    IMG_SIZE = CELL_SIZE * GRID_SIZE
    
    # Colors
    COLORS = {
        'empty': (255, 255, 255),  # White
        'grid': (200, 200, 200),   # Light gray
        'wood': (139, 69, 19),     # Brown
        'stone': (128, 128, 128),  # Gray
        'food': (0, 255, 0),       # Green
        'water': (0, 0, 255),      # Blue
        'house': (255, 0, 0),      # Red
        'farm': (173, 255, 47),    # Green-yellow
        'well': (0, 191, 255),     # Deep sky blue
        'storage': (255, 140, 0),  # Dark orange
        'market': (255, 215, 0),   # Gold
        'agent': (255, 192, 203)   # Pink
    }
    
    # Create image
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=COLORS['empty'])
    draw = ImageDraw.Draw(img)
    
    # Draw grid
    for i in range(GRID_SIZE + 1):
        line_pos = i * CELL_SIZE
        draw.line([(line_pos, 0), (line_pos, IMG_SIZE)], fill=COLORS['grid'])
        draw.line([(0, line_pos), (IMG_SIZE, line_pos)], fill=COLORS['grid'])
    
    # Draw resources
    for pos, (r_type, amount) in self.resource_layer.items():
        x, y = pos
        color = COLORS[r_type.value]
        rect = [
            x * CELL_SIZE,
            y * CELL_SIZE,
            (x + 1) * CELL_SIZE - 1,
            (y + 1) * CELL_SIZE - 1
        ]
        draw.rectangle(rect, fill=color)
    
    # Draw buildings
    for pos, building in self.building_layer.items():
        x, y = pos
        color = COLORS[building.type.value]
        rect = [
            x * CELL_SIZE,
            y * CELL_SIZE,
            (x + 1) * CELL_SIZE - 1,
            (y + 1) * CELL_SIZE - 1
        ]
        draw.rectangle(rect, fill=color)
        # Add building owner ID
        draw.text(
            (x * CELL_SIZE + 2, y * CELL_SIZE + 2),
            str(building.owner_id),
            fill=(0, 0, 0)
        )
    
    # Draw agents
    for pos, agent in self.agent_layer.items():
        x, y = pos
        center = (
            x * CELL_SIZE + CELL_SIZE // 2,
            y * CELL_SIZE + CELL_SIZE // 2
        )
        radius = CELL_SIZE // 3
        draw.ellipse(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius
            ],
            fill=COLORS['agent']
        )
        # Add agent ID
        draw.text(
            (x * CELL_SIZE + 2, y * CELL_SIZE + 2),
            str(agent.id),
            fill=(0, 0, 0)
        )
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
```

### 4. Memory System
Agents maintain a memory of their observations and interactions:

```python
@dataclass
class Memory:
    timestamp: int  # Game turn when memory was created
    type: str      # Observation, Interaction, or Event
    content: str   # Description of what was observed/happened
    location: Optional[Tuple[int, int]]  # Where it happened (if applicable)

def add_memory(self, agent: Agent, memory_type: str, content: str, location: Optional[Tuple[int, int]] = None):
    """Add a memory to agent's memory list"""
    memory = Memory(
        timestamp=self.current_turn,
        type=memory_type,
        content=content,
        location=location
    )
    agent.memory.append(memory)
    
    # Keep only last 50 memories
    if len(agent.memory) > 50:
        agent.memory.pop(0)

# Example memory entries:
# - "Observed wood deposit at (10, 15)"
# - "Built house at (5, 5)"
# - "Traded 5 wood for 3 stone with Agent_2"
# - "Found fertile area with multiple food sources"
```

The memory system helps agents make informed decisions by:
1. Remembering resource locations
2. Tracking building locations and ownership
3. Recording trade history
4. Maintaining relationship context
5. Planning future actions

Each memory entry includes the local view at the time of the observation, allowing agents to reference historical spatial information when making decisions.