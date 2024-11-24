# New Features Implementation Plan

Let's break this down into small, testable iterations:

Iteration 1: Basic Wood Resource System

Add wood storage to tiles in WorldState
Implement GATHER_RESOURCE: wood action
Add basic inventory to Agent class
Test wood gathering and energy costs
Iteration 2: Wood Movement & Placement

Implement 2x movement penalty when carrying wood
Add PLACE_HERE: wood action
Implement wood merging on tiles
Test movement costs and wood placement
Iteration 3: Basic Shelter Creation

Add shelter data structure to WorldState
Implement CRAFT: shelter action
Add basic shelter ownership
Test shelter creation requirements
Iteration 4: Shelter Movement & Energy

Implement owner-only movement on shelters
Add shelter energy bonuses
Update adjacent tile bonuses
Test movement restrictions and energy gains
Iteration 5: Shelter Storage

Add wood storage to shelters
Implement wood transfer to/from shelters
Test shelter storage limits
Iteration 6: Shelter Management

Implement DESTROY: shelter action
Add shelter notifications
Implement TRADE: shelter action
Test ownership transfers
Iteration 7: Turn Prompt System

Update prompt to show nearby shelters
Add available actions based on position
Show wood and shelter information
Test prompt clarity


## 1. Resource System - Wood

### Wood Collection
- New action: `GATHER_RESOURCE: wood`
  * Only available on tree tiles
  * Costs 10 energy to gather 1 wood unit
  * Each tree tile stores up to 15 wood units
  * Visual indicator shows wood amount on tile

### Agent Inventory
- Maximum capacity: 1 wood unit
- Movement penalties:
  * 2x energy cost while carrying wood
  * Example: Normal tile (2 energy), Tree tile (10 energy)
- Actions:
  * `PLACE_HERE: wood` - Drops wood on current tile
  * Wood merges with any existing wood on tile
  * Dropped wood is unclaimed (any agent can pick it up)

## 2. Shelter System

### Shelter Creation
- Requirements:
  * 5+ wood units on a single tile
  * Agent must be on the tile
  * Tile must not have shelter already
- Action: `CRAFT: shelter <name>`
- Properties:
  * 🏠 emoji marks shelter location
  * Only owner can step on shelter tile
  * Owner gets +2 energy when on their shelter
  * Owner can store/retrieve wood only when on shelter

### Turn Prompt Information
- Clear shelter information in agent prompts:
  * "Bob's House (owned by Agent 2) is north of you"
  * "Your house 'Beach House' is to your east"
  * "5 wood units are stored in your 'Beach House'"
  * Available actions shown based on position:
    - "DESTROY: Bob's House" (when adjacent)
    - "PLACE_HERE: wood" (when on own shelter)
    - "TRADE: Beach House Agent 3" (when on own shelter)

### Shelter Properties
- Storage capacity: 30 wood units
- Movement restrictions:
  * Only owner can walk on shelter tile
  * Other agents must walk around
- Energy bonuses:
  * +1 energy for same-color agents adjacent to shelter
  * +2 energy for owner when on shelter tile
  * Stacks with other bonuses (water, same-color agents)
  * Does not work through trees or water
  * Updates when owner changes color

### Shelter Management
- Ownership:
  * One shelter per tile (can be adjacent)
  * No maximum shelters per agent
  * Transferable ownership via `TRADE: <shelter> <agent>`
  * Benefits follow owner's current color
- Destruction:
  * Any agent can destroy when adjacent
  * Action: `DESTROY: <shelter name>`
  * Owner receives notification regardless of distance
  * Message includes destroyer's identity

## 3. Energy System Updates

### Energy Costs
- Base movement:
  * Normal tile: 1 energy
  * Tree tile: 5 energy
- Carrying wood:
  * Normal tile: 2 energy (2x multiplier)
  * Tree tile: 10 energy (2x multiplier)
- Resource gathering:
  * 10 energy to gather 1 wood unit
  * Can only carry 1 wood unit

### Energy Gains (Stackable)
1. Water adjacency: +1 per adjacent water tile
2. Same-color agents: +1 per nearby agent
3. Shelter bonus: +1 when adjacent to same-color shelter
4. Owner bonus: +2 when on own shelter
- No maximum energy cap
- Line of sight restrictions for shelter bonus

## 4. Implementation Phases

### Phase 1: Resource System
- [ ] Add wood resource to tiles
- [ ] Implement wood gathering
- [ ] Add inventory system
- [ ] Update movement costs

### Phase 2: Shelter System
- [ ] Add shelter creation
- [ ] Implement shelter storage
- [ ] Add ownership restrictions
- [ ] Implement energy bonuses
- [ ] Update turn prompts with shelter info

### Phase 3: Management Features
- [ ] Add shelter destruction
- [ ] Implement ownership transfer
- [ ] Add notification system
- [ ] Update prompt system

### Phase 4: Testing & Balancing
- [ ] Test energy costs
- [ ] Balance resource gathering
- [ ] Test shelter mechanics
- [ ] Verify stacking bonuses
- [ ] Test turn prompt clarity
