"""Main game implementation for Color Chat"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random
import logging
from PIL import Image, ImageDraw
import io
import base64

from games.color_chat.types import Position, Agent, WorldConfig, TerrainType, Shelter
from games.color_chat.constants import *

class ColorChatGame:
    """Main game class that interfaces with the runner"""
    
    def __init__(self, run_dir: Path, max_turns: int = 100, world_size: int = DEFAULT_WORLD_SIZE, 
                 communication_radius: int = DEFAULT_COMMUNICATION_RADIUS, 
                 fov_radius: int = DEFAULT_FOV_RADIUS):
        """Initialize the game
        
        Args:
            run_dir: Directory for game logs
            max_turns: Maximum number of turns before game ends
            world_size: Size of the grid (world_size x world_size)
            communication_radius: How far messages can be heard
            fov_radius: Field of view radius for agents (Manhattan distance)
        """
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.config = WorldConfig(size=world_size, communication_radius=communication_radius)
        self.fov_radius = fov_radius
        self.agents: Dict[int, Agent] = {}
        self.current_turn = 0
        self.game_over = False
        self.agent_messages: List[Dict[str, Any]] = []
        self.action_results: Dict[int, List[Dict[str, Any]]] = {}
        self.wood_storage: Dict[Tuple[int, int], int] = {}
        self.shelters: Dict[Tuple[int, int], Shelter] = {}  # Position -> Shelter mapping
        self.potential_tree_tiles = set()  # Track where trees can regrow
        
        # Generate water features
        self._generate_water_features()
        
        # Generate tree terrain and initialize wood storage
        self._generate_tree_terrain()
        
        # Initialize wood storage for tree tiles and track potential locations
        for pos in self.config.tree_tiles:
            self.wood_storage[pos] = MAX_WOOD_PER_TILE
            self.potential_tree_tiles.add(pos)
        
        # Setup colorful console output
        self.colors = CONSOLE_COLORS
        self.ENDC = CONSOLE_END
        self.BOLD = CONSOLE_BOLD
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Get the game's system prompt"""
        return {
            "role": "system",
            "content": """Welcome to the Color Chat Game! You are an intelligent agent in a collaborative resource management world.

CORE MECHANICS:
1. Movement & Energy
   - Move UP/DOWN/LEFT/RIGHT (multiple moves allowed)
   - Each move costs energy
   - Gain energy by:
     * Being near water (passive)
     * Coordinating with same-colored agents
     * Strategic positioning

2. Resource Management
   - Trees provide wood (🌳)
   - Water tiles restore energy (〰️)
   - Build shelters to store resources (🏠)
   - Manage your wood inventory wisely

3. Social Interaction
   - Change color to match others (R,G,B)
   - Communicate with nearby agents
   - Form alliances through color coordination
   - Share resources and strategies

AVAILABLE COMMANDS:
MOVE: UP/DOWN/LEFT/RIGHT     # Strategic positioning
SPEAK: message              # Coordinate with others
COLOR: (R,G,B)             # Match colors for energy bonus
GATHER: wood               # Collect from trees
DROP: wood                 # Create resource stockpiles
CRAFT: shelter_name        # Build permanent storage
STORE: wood               # Save in your shelter
RETRIEVE: wood            # Access stored resources

STRATEGIC TIPS:
- Balance energy consumption with resource gathering
- Coordinate colors with nearby agents for energy boosts
- Position near water for sustained energy
- Build shelters in strategic locations
- Communicate your intentions and coordinate actions

Command Format:
- One command per line
- Exact command names required
- Multiple commands allowed per turn
- No empty lines between commands

Example multi-command strategy:
MOVE: UP
MOVE: RIGHT
SPEAK: Let's coordinate our colors for energy bonus!
COLOR: (255,100,100)
GATHER: wood
STORE: wood

First, analyze the current game state and develop a strategy. Then execute your plan through commands."""
        }

    def run(self, players: List['BaseLLMPlayer']) -> Dict:
        """Main game loop"""
        # Initialize world with players
        self._initialize_agents(len(players))
        
        # Initialize players with system prompt
        system_prompt = self.get_system_prompt()
        for player in players:
            player.initialize_with_prompt(system_prompt)
            self.action_results[player.player_id] = []
        
        while not self.game_over and self.current_turn < self.max_turns:
            # Check for wood respawning at depleted tree locations
            self._handle_wood_respawn()
            
            self._print_turn_header()
            self.current_turn += 1
            
            # Process each player's turn
            for player in players:
                
                self._generate_game_image()

                # Get agent state as formatted message
                state = self._get_agent_state(player.player_id)
                
                # Generate agent-specific view
                agent_view = self._generate_agent_view(player.player_id)
                
                # Create user message
                user_message = {
                    "role": "user",
                    "content": state
                }
                
                # Clear previous turn's results and error messages
                self.action_results[player.player_id] = []
                error_messages = []
                
                # Get model's response using agent-specific view
                response = player.get_response(user_message, agent_view)
                
                # Process all actions in the response
                actions = self._parse_actions(response)
                for action in actions:
                    success, result = self._process_action(player.player_id, action)
                    result_dict = {
                        "action": action,
                        "success": success,
                        "result": result
                    }
                    self.action_results[player.player_id].append(result_dict)
                    self._print_agent_action(player.player_id, action, result_dict)
                    
                    # Collect error messages
                    if not success:
                        try:
                            error_data = json.loads(result)
                            if "error" in error_data:
                                error_messages.append(error_data["error"])
                        except json.JSONDecodeError:
                            error_messages.append(result)
                
                # If there were errors, append them as a system message
                if error_messages:
                    error_summary = "\n".join([f"❌ {error}" for error in error_messages])
                    player.add_message({
                        "role": "system",
                        "content": f"ERRORS IN YOUR TURN:\n{error_summary}"
                    })
                    player.add_message({
                        "role": "system",
                        "content": f"ERRORS IN YOUR TURN:\n{error_summary}"
                    })
                    player.add_message({
                        "role": "system",
                        "content": f"ERRORS IN YOUR TURN:\n{error_summary}"
                    })
        
        return self._generate_game_result()

    def _get_agent_state(self, agent_id: int) -> str:
        """Format the current state as a clean message for the agent"""
        agent = self.agents[agent_id]
        pos_tuple = (agent.position.x, agent.position.y)
        
        # Format current tile description
        tile_desc = self._get_tile_description(pos_tuple)
        
        # Format nearby agents
        visible_agents = self._format_visible_agents(agent)
        
        # Format messages heard this turn
        messages = self._format_messages_heard(agent)
        
        # Build the formatted message
        return f"""Turn {self.current_turn}

CURRENT STATUS
Position: ({agent.position.x}, {agent.position.y})
Energy Level: {agent.energy}/100 {'🟢 High' if agent.energy > 70 else '🟡 Medium' if agent.energy > 30 else '🔴 Low'}
Wood Inventory: {agent.wood_inventory}/1 {'📦 Full' if agent.wood_inventory >= 1 else '✨ Space Available'}
Current Color: {self._format_color(agent.color)}

ENVIRONMENT
Current Tile: {tile_desc}
Water Sources: {'Nearby! ⚡' if any((x, y) in self.config.water_tiles for x, y in [(agent.position.x + dx, agent.position.y + dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]]) else 'None in immediate vicinity'}
Trees in View: {'🌳 Available' if any((x, y) in self.config.tree_tiles and (x, y) in self.wood_storage and self.wood_storage[(x, y)] > 0 for x, y in [(agent.position.x + dx, agent.position.y + dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]]) else 'None adjacent'}

SOCIAL NETWORK
Nearby Agents:
{visible_agents}

Recent Communications:
{messages}

Strategy Reflection:
1. Analyze your current state and surroundings
2. Consider energy management and resource needs
3. Look for collaboration opportunities
4. Plan your next moves carefully
5. Avoid repeating unsuccessful patterns

After strategic analysis, provide your commands in the proper format:
COMMAND: parameter (one per line)"""

    def _get_tile_description(self, pos_tuple: Tuple[int, int]) -> str:
        """Get a clean description of a tile"""
        if pos_tuple in self.config.tree_tiles:
            wood = self.wood_storage.get(pos_tuple, 0)
            return f"Tree ({wood} wood)"
        elif pos_tuple in self.shelters:
            shelter = self.shelters[pos_tuple]
            if shelter.owner_id in self.agents:  # Only show wood storage for owned shelters
                return f"Shelter (owned by Agent {shelter.owner_id}, {shelter.wood_storage} wood stored)"
            return f"Shelter (owned by Agent {shelter.owner_id})"
        elif pos_tuple in self.wood_storage:
            return f"Wood Pile ({self.wood_storage[pos_tuple]} wood)"
        return "Empty"

    def _format_visible_agents(self, agent: Agent) -> str:
        """Format the list of visible agents"""
        visible = [
            other for other in self._get_agents_in_range(agent.position)
            if other.id != agent.id
        ]
        if not visible:
            return "None"
        
        return "\n".join(
            f"- Agent {other.id} at ({other.position.x}, {other.position.y}), "
            f"Color: {self._format_color(other.color)}, "
            f"Energy: {other.energy}, Wood: {other.wood_inventory}"
            for other in visible
        )

    def _format_messages_heard(self, agent: Agent) -> str:
        """Format messages heard this turn"""
        messages = []
        for other_id, results in self.action_results.items():
            if other_id == agent.id:
                continue
            for result in results:
                if (isinstance(result, dict) and 
                    result.get("action", "").startswith("SPEAK:") and
                    other_id in self.agents):
                    other = self.agents[other_id]
                    # Check if message was in range when spoken
                    if agent.position.distance_to(other.position) <= self.config.communication_radius:
                        message = result["action"].replace("SPEAK:", "").strip()
                        messages.append(f"- Agent {other_id}: {message}")
        
        return "\n".join(messages) if messages else "None"

    def _format_color(self, color: Tuple[int, int, int]) -> str:
        """Format an RGB color tuple"""
        return f"({color[0]}, {color[1]}, {color[2]})"

    def _initialize_agents(self, num_agents: int):
        """Initialize agents at random positions adjacent to water"""
        # Find all positions adjacent to water
        water_adjacent = set()
        for wx, wy in self.config.water_tiles:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                pos = (wx + dx, wy + dy)
                if (0 <= pos[0] < self.config.size and 
                    0 <= pos[1] < self.config.size and 
                    pos not in self.config.water_tiles and
                    pos not in self.config.tree_tiles):
                    water_adjacent.add(pos)
        
        # Convert to list of Position objects
        available_positions = [Position(x, y) for x, y in water_adjacent]
        
        if len(available_positions) < num_agents:
            raise ValueError(f"Not enough water-adjacent positions ({len(available_positions)}) for {num_agents} agents!")
        
        random.shuffle(available_positions)
        
        for i in range(num_agents):
            pos = available_positions.pop()
            self.agents[i + 1] = Agent(
                id=i + 1,
                position=pos,
                wood_inventory=0
            )

    def _get_agents_in_range(self, position: Position) -> List[Agent]:
        """Get all agents within communication radius of a position"""
        in_range = []
        for agent in self.agents.values():
            if position.distance_to(agent.position) <= self.config.communication_radius:
                in_range.append(agent)
        return in_range

    def _parse_actions(self, response: str) -> List[str]:
        """Parse multiple actions from response"""
        actions = []
        command_limits = {
            "SPEAK": 1,
            "COLOR": 1,
            "GATHER": 1,
            "DROP": 1,
            "CRAFT": 1,
            "STORE": 1,
            "RETRIEVE": 1
        }
        command_counts = {cmd: 0 for cmd in command_limits}
        
        # Split response into lines and process each line
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and non-action lines
            if not line or ":" not in line:
                continue
                
            command, content = line.split(":", 1)
            command = command.strip().upper()
            
            # Skip if not a valid command
            if command not in ["MOVE", "SPEAK", "COLOR", "GATHER", "DROP", "CRAFT", "STORE", "RETRIEVE"]:
                continue
            
            # Apply command limits (except MOVE which can be used multiple times)
            if command != "MOVE":
                if command_counts[command] >= command_limits[command]:
                    continue
                command_counts[command] += 1
            
            actions.append(line)
            
        return actions

    def _process_action(self, agent_id: int, action_str: str) -> Tuple[bool, str]:
        """Process an action from an agent"""
        try:
            # Parse the action command and content
            if ":" not in action_str:
                return False, json.dumps({"error": "Invalid action format", "details": {}})
                
            command, content = action_str.split(":", 1)
            command = command.strip().upper()
            content = content.strip()
            
            # Process based on command type
            if command == "MOVE":
                valid, msg, details = self._validate_move(content, self.agents[agent_id])
                if valid:
                    # Execute move if valid
                    return self._handle_move(agent_id, content)
                return False, json.dumps({"error": msg, "details": details})
                
            elif command == "SPEAK":
                valid, msg = self._handle_speak(agent_id, content)
                return valid, msg  # Special case - handled by _print_agent_action differently
                
            elif command == "COLOR":
                valid, msg, details = self._validate_color(content)
                if valid:
                    # Execute color change if valid
                    return self._handle_color(agent_id, content)
                return False, json.dumps({"error": msg, "details": details})
                
            elif command == "GATHER":
                valid, msg = self._handle_gather(agent_id)
                if valid:
                    return True, json.dumps({"success": msg, "details": {}})
                return False, json.dumps({"error": msg, "details": {}})
                
            elif command == "DROP":
                valid, msg = self._handle_drop(agent_id)
                if valid:
                    return True, json.dumps({"success": msg, "details": {}})
                return False, json.dumps({"error": msg, "details": {}})
                
            elif command == "CRAFT":
                valid, msg = self._handle_craft(agent_id, content)
                if valid:
                    return True, json.dumps({"success": msg, "details": {}})
                return False, json.dumps({"error": msg, "details": {}})
                
            elif command == "STORE":
                valid, msg = self._handle_store(agent_id)
                if valid:
                    return True, json.dumps({"success": msg, "details": {}})
                return False, json.dumps({"error": msg, "details": {}})
                
            elif command == "RETRIEVE":
                valid, msg = self._handle_retrieve(agent_id)
                if valid:
                    return True, json.dumps({"success": msg, "details": {}})
                return False, json.dumps({"error": msg, "details": {}})
            
            return False, json.dumps({"error": f"Unknown command: {command}", "details": {}})
            
        except Exception as e:
            return False, json.dumps({"error": str(e), "details": {"type": "exception"}})

    def _handle_speak(self, agent_id: int, message: str) -> Tuple[bool, str]:
        """Handle a speak action"""
        agent = self.agents[agent_id]
        agent.last_message = message
        
        self.agent_messages.append({
            "turn": self.current_turn,
            "agent_id": agent_id,
            "message": message,
            "position": {"x": agent.position.x, "y": agent.position.y}
        })
        
        return True, "Message broadcast"

    def _handle_move(self, agent_id: int, direction: str) -> Tuple[bool, str]:
        """Handle a move action"""
        agent = self.agents[agent_id]
        
        # Validate move
        valid, message, details = self._validate_move(direction, agent)
        if not valid:
            return False, json.dumps({"error": message, "details": details})
        
        # Execute move
        new_pos = Position(
            details["new_position"]["x"],
            details["new_position"]["y"]
        )
        agent.position = new_pos
        
        # Decrease energy for movement
        agent.energy -= details["energy_cost"]
        
        # Calculate energy gains
        energy_gain = 0
        
        # Check for adjacent water tiles
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            if (new_pos.x + dx, new_pos.y + dy) in self.config.water_tiles:
                energy_gain += WATER_ADJACENT_ENERGY_GAIN
        
        # Increase energy for nearby agents of the same color
        nearby_agents = self._get_agents_in_range(agent.position)
        same_color_agents = [
            other for other in nearby_agents 
            if other.id != agent_id and other.color == agent.color
        ]
        color_energy_gain = len(same_color_agents) * SAME_COLOR_ENERGY_GAIN
        
        if color_energy_gain > 0:
            # Moving agent gains energy for each nearby same-colored agent
            agent.energy += color_energy_gain
            # Each nearby same-colored agent gains 1 energy
            for nearby_agent in same_color_agents:
                nearby_agent.energy += SAME_COLOR_ENERGY_GAIN
        
        # Apply water energy gain
        agent.energy += energy_gain
        
        details["energy_update"] = {
            "movement_cost": -details["energy_cost"],
            "water_gain": energy_gain,
            "color_proximity_gain": color_energy_gain,
            "new_energy": agent.energy,
            "same_color_neighbors": len(same_color_agents)
        }
        
        return True, json.dumps({"success": "Moved " + direction.upper(), "details": details})

    def _handle_color(self, agent_id: int, color_str: str) -> Tuple[bool, str]:
        """Handle a color change action"""
        # Validate color
        valid, message, details = self._validate_color(color_str)
        if not valid:
            return False, json.dumps({"error": message, "details": details})
        
        # Execute color change
        self.agents[agent_id].color = details["color"]
        return True, json.dumps({"success": "Color updated", "details": details})

    def _handle_gather(self, agent_id: int) -> Tuple[bool, str]:
        """Handle a gather action"""
        agent = self.agents[agent_id]
        pos = (agent.position.x, agent.position.y)
        
        # Check if agent has energy
        if agent.energy < WOOD_GATHER_ENERGY:
            return False, f"⚡ Low energy! Need {WOOD_GATHER_ENERGY} energy to gather wood, but you have {agent.energy}. Try moving near water or coordinating colors with other agents to gain energy!"
        
        # Check if current tile has wood
        if pos not in self.config.tree_tiles:
            return False, "🌳 No trees here to gather wood from. Look for green tiles with tree symbols - they're often found in clusters away from water. Move to a tree tile first!"
            
        # Check if wood is available on this tile
        if pos not in self.wood_storage or self.wood_storage[pos] <= 0:
            # Remove depleted tree tile but keep track of location for respawning
            if pos in self.config.tree_tiles:
                self.config.tree_tiles.remove(pos)
            if pos in self.wood_storage:
                del self.wood_storage[pos]
            return False, "🌳 This tree is depleted! Trees regrow over time. Try finding another tree tile or wait for regrowth. Look for the green patches with tree symbols."
        
        # Check if agent can carry more wood
        if agent.wood_inventory >= 1:
            return False, "📦 Wood inventory full! You can: 1) DROP wood to create a stockpile, 2) STORE it in your shelter, or 3) coordinate with other agents to build something together!"
        
        # Gather wood
        agent.energy -= WOOD_GATHER_ENERGY
        agent.wood_inventory += 1
        self.wood_storage[pos] -= 1
        
        # If tile is depleted, remove it but keep track of location
        if self.wood_storage[pos] <= 0:
            self.config.tree_tiles.remove(pos)
            del self.wood_storage[pos]
        
        return True, f"Gathered wood! Energy remaining: {agent.energy}"

    def _handle_drop(self, agent_id: int) -> Tuple[bool, str]:
        """Handle a drop action"""
        if agent_id not in self.agents:
            return False
            
        agent = self.agents[agent_id]
        if agent.wood_inventory <= 0:
            return False, "📦 No wood to drop"
        
        pos_tuple = (agent.position.x, agent.position.y)
        
        # Drop wood at current position
        if pos_tuple not in self.wood_storage:
            self.wood_storage[pos_tuple] = 0
        self.wood_storage[pos_tuple] += agent.wood_inventory
        agent.wood_inventory = 0
        
        return True, "Wood dropped"

    def _handle_craft(self, agent_id: int, name: str) -> Tuple[bool, str]:
        """Handle shelter crafting action"""
        agent = self.agents[agent_id]
        pos = (agent.position.x, agent.position.y)
        
        # Check wood requirements
        if pos not in self.wood_storage or self.wood_storage[pos] < SHELTER_WOOD_REQUIRED:
            current_wood = self.wood_storage[pos] if pos in self.wood_storage else 0
            return False, f"🏠 Building a shelter requires {SHELTER_WOOD_REQUIRED} wood. You have {current_wood} wood here. To gather more:\n1) Find tree tiles (green with 🌳)\n2) Use GATHER command on tree tiles\n3) DROP wood here to accumulate enough"
            
        if pos in self.shelters:
            return False, "🏠 A shelter already exists here. Choose a new strategic location! Consider:\n- Proximity to resources\n- Access to water\n- Distance from other shelters"
        
        # Create shelter
        self.shelters[pos] = Shelter(
            name=name,
            owner_id=agent_id,
            position=Position(pos[0], pos[1])
        )
        
        # Consume wood
        self.wood_storage[pos] -= SHELTER_WOOD_REQUIRED
        if self.wood_storage[pos] <= 0:
            del self.wood_storage[pos]
        
        return True, f"Created shelter '{name}'"

    def _handle_store(self, agent_id: int) -> Tuple[bool, str]:
        """Handle storing wood in a shelter"""
        agent = self.agents[agent_id]
        pos = (agent.position.x, agent.position.y)
        
        # Check if agent has wood
        if agent.wood_inventory <= 0:
            return False, "📦 No wood to store! First gather wood from tree tiles (green with 🌳) using the GATHER command."
        
        # Check if on shelter tile
        if pos not in self.shelters:
            return False, "🏠 Must be on a shelter to store wood. Look for brown tiles with house symbols. You can:\n1) Build your own shelter\n2) Use your existing shelter\n3) Negotiate access with others"
            
        shelter = self.shelters[pos]
        
        # Check if agent owns shelter
        if shelter.owner_id != agent_id:
            return False, "🔒 This shelter belongs to another agent. You can:\n1) Use your own shelter\n2) Build a new shelter\n3) Negotiate sharing arrangements"
            
        # Check shelter storage capacity
        if shelter.wood_storage >= SHELTER_MAX_STORAGE:
            return False, f"📦 Shelter storage full (max {SHELTER_MAX_STORAGE} wood)! Consider:\n1) Building another shelter\n2) Using wood for crafting\n3) Trading with other agents"
        
        # Store wood
        shelter.wood_storage += agent.wood_inventory
        agent.wood_inventory = 0
        
        return True, f"Stored wood in shelter. Shelter now contains {shelter.wood_storage} wood"

    def _handle_retrieve(self, agent_id: int) -> Tuple[bool, str]:
        """Handle retrieving wood from a shelter"""
        agent = self.agents[agent_id]
        pos = (agent.position.x, agent.position.y)
        
        # Check if on shelter tile
        if pos not in self.shelters:
            return False, "🏠 Must be on a shelter to retrieve wood. Look for brown tiles with house symbols (🏠). Move to your shelter first!"
            
        shelter = self.shelters[pos]
        
        # Check if agent owns shelter
        if shelter.owner_id != agent_id:
            return False, "🔒 Can only retrieve wood from your own shelter. This one belongs to another agent. Find your shelter (brown tile with 🏠) or build a new one!"
            
        # Check if shelter has wood
        if shelter.wood_storage <= 0:
            return False, "📦 This shelter is empty! No wood to retrieve. Try:\n1) Gathering more wood from trees\n2) Checking other shelters you own\n3) Building up your wood storage"
            
        # Check if agent can carry wood
        if agent.wood_inventory >= MAX_WOOD_INVENTORY:
            return False, f"🎒 You're carrying maximum wood ({MAX_WOOD_INVENTORY})! Try:\n1) Dropping wood to create a stockpile\n2) Using it to build something\n3) Trading with other agents"
        
        # Retrieve wood
        agent.wood_inventory += 1
        shelter.wood_storage -= 1
        
        return True, f"Retrieved wood from shelter. Shelter has {shelter.wood_storage} wood remaining"

    def _generate_game_image(self) -> str:
        """Generate a visual representation of the game state"""
        CELL_SIZE = 50
        width = self.config.size * CELL_SIZE
        height = self.config.size * CELL_SIZE
        
        # Create image with white background
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw water tiles in blue
        for x, y in self.config.water_tiles:
            draw.rectangle(
                [x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                fill='#4169E1'  # Royal blue
            )
            draw.text(
                (x * CELL_SIZE + CELL_SIZE//10, y * CELL_SIZE + CELL_SIZE//10),
                "〰️",  # Wave symbol
                fill='white',
                font=None,
                size=int(CELL_SIZE * 1000.8)  # Larger emoji
            )
        
        # Draw tree tiles in green with tree symbol (only if they have wood)
        for x, y in self.config.tree_tiles:
            pos = (x, y)
            # Only draw tree if it has wood remaining
            if pos in self.wood_storage and self.wood_storage[pos] > 0:
                draw.rectangle(
                    [x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                    fill='#90EE90'  # Light green
                )
                draw.text(
                    (x * CELL_SIZE + CELL_SIZE//10, y * CELL_SIZE + CELL_SIZE//10),
                    "🌳",  # Tree symbol
                    fill='#006400',  # Dark green
                    font=None,
                    size=int(CELL_SIZE * 0.8)  # Larger emoji
                )
        
        # Draw wood piles that are not on tree tiles
        for pos, amount in self.wood_storage.items():
            x, y = pos
            if pos not in self.config.tree_tiles and amount > 0:
                draw.text(
                    (x * CELL_SIZE + CELL_SIZE//10, y * CELL_SIZE + CELL_SIZE//10),
                    "🪵",  # Wood symbol
                    fill='black',
                    font=None,
                    size=int(CELL_SIZE * 0.8)
                )
        
        # Draw shelters
        for pos, shelter in self.shelters.items():
            x, y = pos
            draw.rectangle(
                [x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                fill='#964B00'  # Brown
            )
            draw.text(
                (x * CELL_SIZE + CELL_SIZE//10, y * CELL_SIZE + CELL_SIZE//10),
                "🏠",  # House symbol
                fill='black',
                font=None,
                size=int(CELL_SIZE * 0.8)
            )
        
        # Draw grid
        for i in range(self.config.size + 1):
            draw.line([(i * CELL_SIZE, 0), (i * CELL_SIZE, height)], fill='#95A5A6', width=1)  # Light gray
            draw.line([(0, i * CELL_SIZE), (width, i * CELL_SIZE)], fill='#95A5A6', width=1)  # Light gray
        
        # Draw agents
        for agent in self.agents.values():
            x = agent.position.x * CELL_SIZE
            y = agent.position.y * CELL_SIZE
            
            # Draw colored square for agent
            draw.rectangle(
                [x + 5, y + 5, x + CELL_SIZE - 5, y + CELL_SIZE - 5],
                fill=agent.color,
                outline='black'
            )
            
            # Draw agent ID
            draw.text(
                (x + CELL_SIZE//3, y + CELL_SIZE//3),
                str(agent.id),
                fill='black'
            )
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img.save("board.png")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def _generate_agent_view(self, agent_id: int) -> str:
        """Generate a view centered on the agent based on AGENT_VIEW_DISTANCE"""
        CELL_SIZE = 50
        BOUNDARY_THICKNESS = 4
        
        # Calculate grid size based on view distance
        GRID_SIZE = 2 * AGENT_VIEW_DISTANCE + 1  # View distance in each direction + center tile
        
        width = GRID_SIZE * CELL_SIZE
        height = GRID_SIZE * CELL_SIZE
        
        # Create base image with dark background to represent out-of-bounds areas
        img = Image.new('RGB', (width, height), '#2C3E50')  # Dark blue-gray for out of bounds
        draw = ImageDraw.Draw(img)
        
        agent = self.agents[agent_id]
        center_x = agent.position.x
        center_y = agent.position.y
        
        # Calculate view boundaries based on AGENT_VIEW_DISTANCE
        start_x = center_x - AGENT_VIEW_DISTANCE
        start_y = center_y - AGENT_VIEW_DISTANCE
        
        # Draw tiles
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                world_x = start_x + x
                world_y = start_y + y
                
                # Only draw tiles that are within world bounds
                if 0 <= world_x < self.config.size and 0 <= world_y < self.config.size:
                    # Draw white background for in-bounds tiles
                    draw.rectangle(
                        [x * CELL_SIZE, y * CELL_SIZE, 
                         (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                        fill='white'
                    )
                    
                    # Draw water tiles
                    if (world_x, world_y) in self.config.water_tiles:
                        draw.rectangle(
                            [x * CELL_SIZE, y * CELL_SIZE, 
                             (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                            fill='#4169E1'  # Royal blue
                        )
                        draw.text(
                            (x * CELL_SIZE + CELL_SIZE//10, y * CELL_SIZE + CELL_SIZE//10),
                            "〰️",  # Wave symbol
                            fill='white',
                            font=None,
                            size=int(CELL_SIZE * 0.9)
                        )
                    
                    # Draw tree tiles (only if they exist and have wood)
                    pos = (world_x, world_y)
                    if (pos in self.config.tree_tiles and 
                        pos in self.wood_storage and 
                        self.wood_storage[pos] > 0):
                        draw.rectangle(
                            [x * CELL_SIZE, y * CELL_SIZE, 
                             (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                            fill='#90EE90'  # Light green
                        )
                        draw.text(
                            (x * CELL_SIZE + CELL_SIZE//10, y * CELL_SIZE + CELL_SIZE//10),
                            "🌳",  # Tree symbol
                            fill='#006400',
                            font=None,
                            size=int(CELL_SIZE * 0.9)
                        )
                        
                    # Draw shelters
                    if pos in self.shelters:
                        shelter = self.shelters[pos]
                        draw.rectangle(
                            [x * CELL_SIZE, y * CELL_SIZE, 
                             (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                            fill='#964B00'  # Brown
                        )
                        draw.text(
                            (x * CELL_SIZE + CELL_SIZE//10, y * CELL_SIZE + CELL_SIZE//10),
                            "🏠",  # House symbol
                            fill='black',
                            font=None,
                            size=int(CELL_SIZE * 0.9)
                        )
        
        # Draw grid only for in-bounds areas
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                world_x = start_x + x
                world_y = start_y + y
                
                if 0 <= world_x < self.config.size and 0 <= world_y < self.config.size:
                    # Draw right border if next cell is in bounds
                    if x < GRID_SIZE - 1 and 0 <= world_x + 1 < self.config.size:
                        draw.line(
                            [(x + 1) * CELL_SIZE, y * CELL_SIZE,
                             (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                            fill='#95A5A6', width=1
                        )
                    # Draw bottom border if next cell is in bounds
                    if y < GRID_SIZE - 1 and 0 <= world_y + 1 < self.config.size:
                        draw.line(
                            [x * CELL_SIZE, (y + 1) * CELL_SIZE,
                             (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE],
                            fill='#95A5A6', width=1
                        )
        
        # Draw agents
        for other_agent in self.agents.values():
            # Convert world coordinates to view coordinates
            view_x = other_agent.position.x - start_x
            view_y = other_agent.position.y - start_y
            
            # Only draw if agent is within view bounds AND within world bounds
            if (0 <= view_x < GRID_SIZE and 0 <= view_y < GRID_SIZE and
                0 <= other_agent.position.x < self.config.size and 
                0 <= other_agent.position.y < self.config.size):
                
                draw.rectangle(
                    [view_x * CELL_SIZE + 5, view_y * CELL_SIZE + 5,
                     (view_x + 1) * CELL_SIZE - 5, (view_y + 1) * CELL_SIZE - 5],
                    fill=other_agent.color,
                    outline='black'
                )
                draw.text(
                    (view_x * CELL_SIZE + CELL_SIZE//3, view_y * CELL_SIZE + CELL_SIZE//3),
                    str(other_agent.id),
                    fill='black'
                )
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img.save("agent.png")
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def _generate_game_result(self) -> Dict:
        """Generate final game results"""
        return {
            "turns_played": self.current_turn,
            "game_over": self.game_over,
            "final_state": {
                agent_id: {
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "color": agent.color,
                    "last_message": agent.last_message,
                    "energy": agent.energy,
                    "wood_inventory": agent.wood_inventory
                }
                for agent_id, agent in self.agents.items()
            }
        }

    def _print_turn_header(self):
        """Print a formatted turn header"""
        print(f"\n=== 🎮 Turn {self.current_turn} ===\n")

    def _print_agent_action(self, agent_id: int, action: str, result: Dict[str, Any]):
        """Print a formatted agent action and result"""
        agent = self.agents[agent_id]
        color = self.colors.get(agent_id, '')
        
        # Start with agent ID and position
        print(f"{color}Agent {agent_id} @ ({agent.position.x}, {agent.position.y}): ", end='')
        
        # Print action and result concisely
        if result["success"]:
            if action.startswith("SPEAK:"):
                message = action[6:].strip()
                print(f"💬 {message[:50]}..." if len(message) > 50 else f"💬 {message}")
            elif action.startswith("MOVE:"):
                print(f"🚶 {action.split(':')[1].strip()}")
            elif action.startswith("COLOR:"):
                print(f"🎨 {action.split(':')[1].strip()}")
            elif action.startswith("GATHER:"):
                print("🌳 Gathered wood")
            elif action.startswith("DROP:"):
                print("📦 Dropped wood")
            elif action.startswith("CRAFT:"):
                print("🏠 Built shelter")
            elif action.startswith("STORE:"):
                print("📥 Stored wood")
            elif action.startswith("RETRIEVE:"):
                print("📤 Retrieved wood")
        else:
            # Simplified error message
            try:
                error_data = json.loads(result['result'])
                error_msg = error_data.get('error', result['result'])
                print(f"❌ Failed: {action.split(':')[0]} - {error_msg.split('.')[0]}")
            except:
                print(f"❌ Failed: {action}")
        print(f"{self.ENDC}")

    def _get_agents_in_range(self, position: Position) -> List[Agent]:
        """Get all agents within communication radius of a position"""
        in_range = []
        for agent in self.agents.values():
            if position.distance_to(agent.position) <= self.config.communication_radius:
                in_range.append(agent)
        return in_range

    def gather_wood(self, agent_id: int) -> Tuple[bool, str]:
        """Attempt to gather wood from the current tile"""
        agent = self.agents[agent_id]
        pos = (agent.position.x, agent.position.y)
        
        # Check if agent has energy
        if agent.energy < WOOD_GATHER_ENERGY:
            return False, f"⚡ Low energy! Need {WOOD_GATHER_ENERGY} energy to gather wood, but you have {agent.energy}. Try moving near water or coordinating colors with other agents to gain energy!"
        
        # Check if current tile has wood
        if pos not in self.config.tree_tiles:
            return False, "🌳 No trees here to gather wood from. Look for green tiles with tree symbols - they're often found in clusters away from water. Move to a tree tile first!"
            
        # Check if wood is available on this tile
        if pos not in self.wood_storage or self.wood_storage[pos] <= 0:
            # Remove depleted tree tile but keep track of location for respawning
            if pos in self.config.tree_tiles:
                self.config.tree_tiles.remove(pos)
            if pos in self.wood_storage:
                del self.wood_storage[pos]
            return False, "🌳 This tree is depleted! Trees regrow over time. Try finding another tree tile or wait for regrowth. Look for the green patches with tree symbols."
        
        # Check if agent can carry more wood
        if agent.wood_inventory >= 1:
            return False, "📦 Wood inventory full! You can: 1) DROP wood to create a stockpile, 2) STORE it in your shelter, or 3) coordinate with other agents to build something together!"
        
        # Gather wood
        agent.energy -= WOOD_GATHER_ENERGY
        agent.wood_inventory += 1
        self.wood_storage[pos] -= 1
        
        # If tile is depleted, remove it but keep track of location
        if self.wood_storage[pos] <= 0:
            self.config.tree_tiles.remove(pos)
            del self.wood_storage[pos]
        
        return True, f"Gathered wood! Energy remaining: {agent.energy}"

    def drop_wood(self, agent_id: int) -> bool:
        """Drop wood at current position"""
        if agent_id not in self.agents:
            return False
            
        agent = self.agents[agent_id]
        if agent.wood_inventory <= 0:
            return False
            
        pos_tuple = (agent.position.x, agent.position.y)
        
        # Drop wood at current position
        if pos_tuple not in self.wood_storage:
            self.wood_storage[pos_tuple] = 0
        self.wood_storage[pos_tuple] += agent.wood_inventory
        agent.wood_inventory = 0
        
        return True

    def _handle_wood_respawn(self):
        """Check depleted tree locations for wood respawning"""
        for pos in self.potential_tree_tiles:
            # If this is a depleted tree location
            if pos not in self.config.tree_tiles:
                # 5% chance to respawn wood
                if random.random() < WOOD_RESPAWN_CHANCE:
                    self.config.tree_tiles.add(pos)
                    self.wood_storage[pos] = MAX_WOOD_PER_TILE

    def _validate_move(self, direction: str, agent: Agent) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate a move action before executing it"""
        direction = direction.upper()
        
        # Validate direction format
        if direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            return False, "Invalid direction format. Please use UP, DOWN, LEFT, or RIGHT in capital letters.", {}
        
        # Calculate new position
        new_x, new_y = agent.position.x, agent.position.y
        if direction == "UP":
            new_y -= 1
        elif direction == "DOWN":
            new_y += 1
        elif direction == "LEFT":
            new_x -= 1
        elif direction == "RIGHT":
            new_x += 1
        
        # Check boundaries
        if new_x < 0 or new_x >= self.config.size or new_y < 0 or new_y >= self.config.size:
            return False, f"⚠️ That move would take you outside the {self.config.size}x{self.config.size} grid. Try moving in a different direction to stay within bounds.", {}
        
        # Check water tiles
        if (new_x, new_y) in self.config.water_tiles:
            return False, "🌊 Cannot move into water tiles. While water nearby gives energy, you can't swim! Try moving along the shoreline instead.", {}
        
        # Check shelter tiles
        new_pos = (new_x, new_y)
        if new_pos in self.shelters and self.shelters[new_pos].owner_id != agent.id:
            return False, f"🏠 This shelter belongs to Agent {self.shelters[new_pos].owner_id}. You can only enter your own shelter. Try building your own or negotiating access!", {}
        
        # Check if moving into tree (allowed but costs more energy)
        energy_cost = TREE_MOVE_ENERGY_COST if (new_x, new_y) in self.config.tree_tiles else MOVE_ENERGY_COST
        
        # Check if agent has enough energy for the move
        if agent.energy < energy_cost:
            terrain = "tree (costs more energy)" if (new_x, new_y) in self.config.tree_tiles else "normal terrain"
            return False, f"⚡ Need {energy_cost} energy to move into {terrain}. You have {agent.energy}. Try moving near water or coordinating colors with other agents to gain energy!", {}
        
        # Check collisions
        if any(a.position.x == new_x and a.position.y == new_y for a in self.agents.values()):
            return False, f"👥 Position ({new_x}, {new_y}) is occupied by another agent. Try moving to an adjacent empty tile or coordinate movement with them!", {}
        
        return True, f"Moving {direction}", {
            "new_position": {"x": new_x, "y": new_y},
            "energy_cost": energy_cost
        }

    def _validate_color(self, color_str: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate a color action before executing it"""
        try:
            # Check format
            if not (color_str.startswith("(") and color_str.endswith(")")):
                return False, "🎨 Color must be in format (R,G,B).", {}
            
            # Parse values
            color_str = color_str.strip("()")
            try:
                r, g, b = map(int, color_str.split(","))
            except ValueError:
                return False, "🎨 Color needs three numbers separated by commas. Example: (R,G,B)", {}
            
            # Validate ranges
            if not all(0 <= c <= 255 for c in (r, g, b)):
                return False, "🎨 Color values must be between 0 and 255. (R,G,B) where R, G, and B are all integers between 0 and 225", {}
            
            return True, "Color changed", {"color": (r, g, b)}
            
        except Exception as e:
            return False, f"🎨 Invalid color format: {str(e)}. Use format (R,G,B) with values 0-255.", {}

    def _generate_water_features(self):
        """Generate water pools and a connecting river"""
        size = self.config.size
        max_water_tiles = (size * size) // 4  # Maximum 25% of map can be water
        
        # Calculate pool size based on map size
        pool_size = max(2, size // 8)  # Minimum 2, scales with map size
        
        # Generate 4 pools in different quadrants
        quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
        pool_centers = []
        
        for qx, qy in quadrants:
            # Calculate quadrant boundaries with padding for pool size
            x_min = qx * (size // 2) + pool_size
            x_max = (qx + 1) * (size // 2) - pool_size
            y_min = qy * (size // 2) + pool_size
            y_max = (qy + 1) * (size // 2) - pool_size
            
            if x_min >= x_max or y_min >= y_max:
                continue  # Skip if quadrant is too small
            
            # Place pool center
            center_x = random.randint(x_min, x_max)
            center_y = random.randint(y_min, y_max)
            pool_centers.append((center_x, center_y))
            
            # Create pool with size proportional to map
            pool_radius = pool_size // 2
            for dx in range(-pool_radius, pool_radius + 1):
                for dy in range(-pool_radius, pool_radius + 1):
                    # Make pools slightly irregular
                    if random.random() < 0.8:  # 80% chance to place water
                        self.config.water_tiles.add((center_x + dx, center_y + dy))
        
        # Select two random pools to connect with a river
        if len(pool_centers) >= 2:
            pool1, pool2 = random.sample(pool_centers, 2)
            self._generate_river(pool1, pool2)
        
        # If we exceeded max water tiles, remove some random water tiles
        while len(self.config.water_tiles) > max_water_tiles:
            tile_to_remove = random.choice(list(self.config.water_tiles))
            self.config.water_tiles.remove(tile_to_remove)
        
        # Generate tree terrain after water is finalized
        self._generate_tree_terrain()

    def _generate_tree_terrain(self):
        """Generate tree terrain in all positions more than 5 blocks away from water"""
        size = self.config.size
        
        # Find all positions within 5 blocks of water
        water_clearing = set()
        for wx, wy in self.config.water_tiles:
            for dx in range(-5, 6):  # -5 to +5 inclusive
                for dy in range(-5, 6):
                    # Check if total Manhattan distance is <= 5
                    if abs(dx) + abs(dy) <= 5:
                        pos = (wx + dx, wy + dy)
                        if (0 <= pos[0] < size and 0 <= pos[1] < size):
                            water_clearing.add(pos)
        
        # Add trees to all positions that are not water and not within 5 blocks of water
        for x in range(size):
            for y in range(size):
                pos = (x, y)
                if (pos not in self.config.water_tiles and 
                    pos not in water_clearing):
                    self.config.tree_tiles.add(pos)

    def _generate_river(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Generate a river path between two points using a simple pathfinding"""
        current = start
        path = set()
        
        while current != end:
            path.add(current)
            x, y = current
            end_x, end_y = end
            
            # Randomly choose between moving in x or y direction
            if random.random() < 0.5 and x != end_x:
                x += 1 if end_x > x else -1
            elif y != end_y:
                y += 1 if end_y > y else -1
            else:
                x += 1 if end_x > x else -1
            current = (x, y)
            
            # Add some randomness to river width (1-2 tiles wide)
            if random.random() < 0.3:  # 30% chance for wider river
                offset = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
                path.add((x + offset[0], y + offset[1]))
        
        path.add(end)
        self.config.water_tiles.update(path)

    def get_terrain(self, position: Position) -> TerrainType:
        """Get terrain type at a position"""
        pos_tuple = (position.x, position.y)
        if pos_tuple in self.config.water_tiles:
            return TerrainType.WATER
        elif pos_tuple in self.config.tree_tiles and pos_tuple in self.wood_storage and self.wood_storage[pos_tuple] > 0:
            return TerrainType.TREE
        elif pos_tuple in self.shelters:
            return TerrainType.SHELTER
        return TerrainType.EMPTY

    def is_valid_position(self, position: Position) -> bool:
        """Check if a position is within world bounds"""
        return (0 <= position.x < self.config.size and 
                0 <= position.y < self.config.size)

    def move_agent(self, agent_id: int, new_position: Position) -> bool:
        """Move an agent to a new position if valid"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        if not self.is_valid_position(new_position):
            return False
            
        # Check if movement is only 1 tile in any direction
        if agent.position.distance_to(new_position) != 1:
            return False
            
        # Check terrain at new position
        terrain = self.get_terrain(new_position)
        if terrain == TerrainType.WATER:
            return False
            
        # Calculate energy cost based on wood carrying status
        energy_cost = 1 * (WOOD_CARRY_MULTIPLIER if agent.wood_inventory > 0 else 1)
        
        # Check if agent has enough energy
        if agent.energy < energy_cost:
            return False
            
        # Move agent and update energy
        agent.position = new_position
        agent.energy -= energy_cost
        
        # Gain energy from nearby agents
        nearby_agents = self._get_agents_in_range(agent.position)
        if nearby_agents:
            agent.energy = min(100, agent.energy + 1)
            
        return True
