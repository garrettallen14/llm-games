# Settlement Game Implementation Guide

## Core Game Components

### 1. World Environment
- Grid Size: 500x500 cells
- Terrain Types and Movement Costs:
  - WATER: Impassable
  - MOUNTAIN: 2x energy cost
  - FOREST: 1.5x energy cost
  - PLAINS: 1x energy cost
  - FERTILE: 1x energy cost

### 2. Resources
Types and Collection Costs:
- WOOD: 5 energy per 10 units
- STONE: 8 energy per 10 units
- FOOD: 3 energy per 10 units
- WATER: 2 energy per 10 units

Resource Mechanics:
- Maximum 50 units per collection
- Must be adjacent to resource
- Collection is instant
- No collection during night
- Resources regenerate 10% daily
- 20% chance for new deposits near existing ones
- Maximum 200 units per deposit

### 3. Buildings
Types:
- HOUSE: Provides +5 energy regeneration per turn
- FARM: Collection point for food
- WELL: Collection point for water
- STORAGE: Store resources
- MARKET: Enable trading

Building Properties:
- Type
- Position (x, y)
- Owner ID
- 16x16 internal room when entered
- Exit from any edge
- 1 energy cost to enter/exit

### 4. Agents
Properties:
- Unique ID
- Name
- Position (x, y)
- Energy (100 max)
- Inventory (resource storage)
- Vision radius: 10 tiles (5 at night)

### 5. Time System
Day/Night Cycle:
- 24 turns = 1 day
- Night (turns 12-23):
  - Double movement energy cost
  - No resource collection
  - Reduced vision (5 tiles)
  - No energy regeneration
- Day (turns 0-11):
  - Normal movement cost
  - Resource collection enabled
  - Normal vision (10 tiles)
  - Normal energy regeneration

### 6. Energy System
Base Values:
- Maximum: 100
- Minimum for actions: 5

Regeneration:
- In house: +5 per turn
- Normal: +2 per turn
- Night: 0 per turn

Consumption:
- Movement: 1 * terrain multiplier
- Resource collection: varies by resource
- Building: 20
- Trading: 1
- Speaking: 0

### 7. Actions
Movement and Pathfinding:
```python
def move_agent(agent_id: int, direction: str) -> Tuple[bool, str]:
    """
    Move agent in cardinal direction (north, south, east, west)
    - Uses A* pathfinding
    - Avoids water tiles
    - Considers terrain movement costs
    - Maximum path length: 50 tiles
    """
    pass

def get_path_cost(start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """
    Calculate total energy cost of path
    - Water: Impassable
    - Mountain: 2x cost
    - Forest: 1.5x cost
    - Plains/Fertile: 1x cost
    """
    pass
```

Resource Collection:
```python
def gather_resource(agent_id: int, resource_pos: Tuple[int, int]) -> Tuple[bool, str]:
    """
    Gather resources from adjacent tile
    - Check adjacency
    - Verify sufficient energy
    - Transfer up to 50 units
    - Consume energy based on resource type
    """
    pass
```

Trading:
```python
def trade(agent_id: int, target_id: int, offer: Dict, request: Dict) -> Tuple[bool, str]:
    """
    Simple trading system
    - Must be adjacent
    - Direct offer/request
    - No negotiation
    - Costs 1 energy
    """
    pass
```

Building Interaction:
```python
def enter_building(agent_id: int, building_pos: Tuple[int, int]) -> Tuple[bool, str]:
    """
    Enter 16x16 building room
    - Empty room with walls
    - Exit from any edge
    - Costs 1 energy
    """
    pass
```

Social:
```python
def say(agent_id: int, message: str) -> Tuple[bool, str]:
    """
    Broadcast message to nearby agents
    - 10 tile radius (5 at night)
    - No energy cost
    - No persistent memory
    """
    pass
```

### 8. State Representation
```python
@dataclass
class GameState:
    """State representation for LLM consumption"""
    inventory: Dict[str, int]
    energy: int
    position: Tuple[int, int]
    time: Dict[str, Any]  # turn, is_night, hour
    local_view: str       # ASCII representation
    visible_agents: List[Dict]  # nearby agents and their messages
    valid_actions: List[str]
```

### 9. Action Parsing
```python
def parse_llm_action(action_str: str) -> Tuple[bool, Optional[Dict], str]:
    """
    Parse and validate LLM action response
    - Validate JSON format
    - Check action possibility
    - Return success, parsed action, error message
    
    On failure:
    - Add error to agent.messages
    - Retry with same state
    """
    try:
        action = json.loads(action_str)
        if not validate_action_format(action):
            return False, None, "Invalid action format"
        
        if not validate_action_possible(action):
            return False, None, "Action not possible"
            
        return True, action, ""
    except Exception as e:
        return False, None, f"Error parsing action: {str(e)}"
```

### Required Action Format
```python
{
    "action_type": str,  # move, gather, trade, build, say, enter_building
    "parameters": {
        # Move
        "direction": str,  # north, south, east, west
        
        # Gather
        "resource_position": [int, int],
        
        # Trade
        "target_id": int,
        "offer": Dict[str, int],
        "request": Dict[str, int],
        
        # Build
        "building_type": str,
        "position": [int, int],
        
        # Say
        "message": str,
        
        # Enter Building
        "building_position": [int, int]
    },
    "reasoning": str  # Brief explanation of decision
}
```
