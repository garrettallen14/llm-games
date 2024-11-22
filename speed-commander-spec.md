# Speed Commander
Real-time Strategy Game Specification v1.0

## Table of Contents
1. Game Overview
2. Core Systems
3. Visual Implementation
4. Game Logic
5. State Management
6. AI Interface
7. Technical Requirements

## 1. Game Overview

### 1.1 Core Concept
Speed Commander is a real-time strategy game designed for AI agent evaluation, testing both strategic thinking and action speed. Two players (RED vs BLUE) compete for resource control on a symmetric grid.

### 1.2 Basic Parameters
- Game Duration: 180 seconds
- Board Size: 12x12 grid
- Update Frequency:
  * Game Tick: 2 seconds
  * Resource Node Spawn: 10 seconds
  * Action Processing: Real-time
- Victory Conditions:
  * Control 75% of territory
  * Eliminate all enemy units
  * Accumulate 1000 resources
  * Highest score after time limit

### 1.3 Scoring System
```
Final Score = (
    (Resources Collected × 0.4) +
    (Territory Controlled × 0.3) +
    (Resource Efficiency × 0.2) +
    (Actions Per Minute × 0.1)
)
```

## 2. Core Systems

### 2.1 Game Elements

#### Units
```
Worker:
- Cost: 50 resources
- Health: 50
- Attack: 0
- Range: 0
- Speed: 1 cell/action
- Special: Can capture nodes, build structures

Fighter:
- Cost: 100 resources
- Health: 100
- Attack: 20
- Range: 1
- Speed: 1 cell/action
- Special: None

Speedster:
- Cost: 75 resources
- Health: 50
- Attack: 10
- Range: 1
- Speed: 2 cells/action
- Special: Can move two cells per action
```

#### Structures
```
Collector:
- Cost: 100 resources
- Health: 100
- Effect: +50% resource collection to adjacent nodes
- Special: None

Turret:
- Cost: 150 resources
- Health: 100
- Attack: 15
- Range: 2
- Special: Auto-attacks nearest enemy

Shield:
- Cost: 100 resources
- Health: 150
- Effect: +50% defense to adjacent friendly units
- Special: Defense bonus stacks multiplicatively
```

### 2.2 Resource System
- Starting Amount: 100 resources per player
- Base Collection: +10 per owned node per tick
- Collection Bonuses:
  * Adjacent Collector: +5 per tick
  * Multiple Collectors: Additive bonus
- Collection Timing: Every 2 seconds (game tick)
- Node Spawning: Every 10 seconds at random valid position

## 3. Visual Implementation

### 3.1 PNG Renderer Requirements
- Resolution: 800x800 pixels
- Grid: 12x12 with clear borders
- Cell Size: 66x66 pixels (800/12)
- Border Width: 2 pixels

### 3.2 Visual Elements
```
Units:
- Worker: Circle
- Fighter: Triangle
- Speedster: Diamond

Structures:
- Collector: Square
- Turret: Pentagon
- Shield: Hexagon

Other Elements:
- Resource Node: Star/Asterisk
- Territory: Background tint (Red: rgba(255,0,0,0.1), Blue: rgba(0,0,255,0.1))
- Grid Lines: Light gray (#CCCCCC)
- Health Bars: 20x4 pixels above units
```

### 3.3 Color Scheme
```
Primary Colors:
- Red Team: #FF4444
- Blue Team: #4444FF
- Neutral: #CCCCCC
- Resource Node: #FFD700
- Grid Background: #FFFFFF
- Grid Lines: #CCCCCC

UI Elements:
- Health Bar (Full): #00FF00
- Health Bar (Empty): #FF0000
- Territory Control: Team color at 10% opacity
- Unit IDs: #000000 (small text)
```

### 3.4 State Representation
Each rendered frame must include:
- Complete board state
- Resource counts for both players
- Time remaining
- Current scores
- Unit IDs
- Health indicators
- Territory control visualization

## 4. Game Logic

### 4.1 Turn Processing
```python
def process_game_tick():
    1. Collect resources from owned nodes
    2. Apply collector bonuses
    3. Process automatic actions (turrets)
    4. Update territory control
    5. Check victory conditions
    6. Update scores
    7. Generate new game state image

def process_player_action(action):
    1. Validate action legality
    2. Check resource requirements
    3. Apply action effects
    4. Update unit states
    5. Check combat results
    6. Update game state
    7. Generate new game state image
```

### 4.2 Action Validation Rules
```
BUILD:
- Sufficient resources available
- Valid build location (empty cell)
- In owned or neutral territory
- Not adjacent to resource node
- Worker unit available if structure

MOVE:
- Unit exists and is owned by player
- Target cell is empty
- Target within movement range
- Path is clear (for Speedster)

ATTACK:
- Unit has attack capability
- Target in range
- Target is enemy unit/structure
- Unit hasn't attacked this tick

CAPTURE:
- Must be Worker unit
- Must be on resource node
- Node must be neutral or enemy-owned
```

### 4.3 Combat Resolution
```
Damage Calculation:
base_damage = attacker.attack
if target.adjacent_to_shield:
    damage_reduction = 0.5 per shield
final_damage = base_damage * (1 - damage_reduction)

Death Processing:
- Unit/structure removed when health <= 0
- Resource nodes become neutral
- Update territory control
```

## 5. State Management

### 5.1 Game State Structure
```python
GameState = {
    "turn_number": int,
    "time_remaining": float,
    "board": numpy.array(12,12),
    "players": {
        "red": PlayerState,
        "blue": PlayerState
    },
    "nodes": List[NodeState],
    "units": List[UnitState],
    "structures": List[StructureState],
    "territory": numpy.array(12,12)
}

PlayerState = {
    "resources": int,
    "score": float,
    "actions_this_turn": int
}

UnitState = {
    "id": str,
    "type": str,
    "owner": str,
    "position": (x,y),
    "health": int,
    "actions_remaining": int
}
```

### 5.2 State Updates
- State updates are atomic
- All state changes generate new game state image
- State history maintained for replay capability
- Validation before state change

## 6. AI Interface

### 6.1 Input Format
```python
{
    "player": "RED" | "BLUE",
    "game_state": GameState,
    "valid_actions": List[Action],
    "action_history": List[Action]
}
```

### 6.2 Output Format
```python
{
    "actions": [
        {
            "action_type": str,
            "unit_id": str,
            "target": [x,y],
            "build_type": str,
            "reasoning": str
        },
        ...
    ],
    "strategy": str
}
```

## 7. Technical Requirements

### 7.1 Performance Requirements
- State updates: < 10ms
- Image generation: < 50ms
- Action validation: < 5ms
- Total tick processing: < 100ms

### 7.2 Implementation Requirements
- Deterministic game logic
- Reproducible random events
- Comprehensive error handling
- State validation at every step
- Complete action logging
- Replay capability

### 7.3 Testing Requirements
- Unit tests for all game logic
- State validation tests
- Performance benchmarks
- AI agent integration tests
- Replay verification

Would you like me to elaborate on any section or add more specific details to any part?
