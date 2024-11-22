"""Main game implementation for Settlement"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('settlement.log'),
        logging.StreamHandler()
    ]
)

from games.settlement.world import World, WorldConfig
from games.settlement.types import (
    TerrainType, ResourceType, BuildingType, ActionType,
    Position, Resource, Building, Agent
)

class SettlementGame:
    """Main game class that interfaces with the runner"""
    
    def __init__(self, run_dir: Path, max_turns: int = 1000):
        """Initialize the game"""
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.world = World(WorldConfig())
        self.current_turn = 0
        self.game_over = False
        self.logs: List[Dict[str, Any]] = []
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Public interface for getting the game's system prompt"""
        return self._generate_system_prompt()

    def _generate_system_prompt(self) -> Dict[str, str]:
        """Internal implementation of system prompt generation"""
        return {
            "role": "system",
            "content": """You are an agent in a settlement-building simulation. Your goal is to survive and thrive by:
1. Gathering resources (wood, stone, food, water)
2. Building structures (houses, farms, wells, storage, markets)
3. Trading with other agents
4. Managing your energy levels
5. Developing relationships

The world is a 500x500 grid with:
- Terrain: WATER (impassable), MOUNTAIN (2x energy), FOREST (1.5x energy), PLAINS (1x energy), FERTILE (1x energy)
- Resources: WOOD, STONE, FOOD, WATER (each with different collection costs)
- Buildings: HOUSE (+5 energy/turn), FARM, WELL, STORAGE, MARKET
- Day/Night cycle: 24 turns per day, reduced vision and no resource collection at night

You have:
- Energy (max 100): Used for movement, building, gathering
- Inventory: Stores collected resources
- Vision: 10 tiles during day, 5 at night

VISUAL REPRESENTATION:
The game world is shown as a grid where each cell contains:
1. Terrain Types (Background Color):
   - WATER (WAT) = Blue
   - MOUNTAIN (MTN) = Gray
   - FOREST (FOR) = Green
   - PLAINS (PLN) = Khaki
   - FERTILE (FRT) = Brown

2. Resources (With Amount):
   - WOOD (WOD) = Sienna
   - STONE (STN) = Silver
   - FOOD (FOD) = Gold
   - WATER (WTR) = Deep Sky Blue

3. Buildings (With Owner ID):
   - HOUSE (HSE) = Red
   - FARM (FRM) = Yellow-Green
   - WELL (WEL) = Steel Blue
   - STORAGE (STR) = Goldenrod
   - MARKET (MKT) = Dark Orange

4. Agents:
   - Shown as: A## (where ## is agent ID)
   - Example: A01, A02, etc.

5. Grid Coordinates:
   - X coordinates shown at top
   - Y coordinates shown on left
   - Format: ###,### (x,y)

Available actions:
1. MOVE: north|south|east|west
2. GATHER: (x,y)  # Use exact coordinates from grid
3. BUILD: type (x,y)
4. TRADE: player_id offer_resource offer_amount request_resource request_amount
5. ENTER: (x,y) # the building location
6. SAY: message

You must respond with a single action.
Each individual action you would like to perform must be on its own line.
The format of each action is shown above, and must be followed strictly.

Example Response:
REASONING: I see a WOOD resource at coordinates (123,456) adjacent to my position.
ACTION: GATHER: (123,456)
"""
    }

    def run(self, players: List['BaseLLMPlayer']) -> Dict:
        """Main game loop"""
        logging.info(f"Starting game with {len(players)} players")
        logging.info(f"World size: {self.world.size}x{self.world.size}")
        
        # Initialize world with players
        for player in players:
            self.world.add_agent(f"Agent_{player.player_id}")
        
        # Initialize players with system prompt
        system_prompt = self.get_system_prompt()
        for player in players:
            player.initialize_with_prompt(system_prompt)
        
        while not self.game_over and self.current_turn < self.max_turns:
            self.current_turn += 1
            logging.info(f"\n=== Turn {self.current_turn} ===")
            
            # Process each player's turn
            for player in players:
                # Get agent state
                state = self.world.get_agent_state(player.player_id)
                agent_info = state['agent']
                world_info = state['world']['time']
                logging.info(f"\nPlayer {player.player_id}'s turn:")
                logging.info(f"Position: ({agent_info['position']['x']}, {agent_info['position']['y']})")
                logging.info(f"Energy: {agent_info['energy']}")
                logging.info(f"Inventory: {agent_info['inventory']}")
                logging.info(f"Time: Day {world_info['day']}, {'Night' if world_info['is_night'] else 'Day'}")
                
                # Generate game visualization centered on current agent
                game_image = self._generate_game_image((agent_info['position']['x'], agent_info['position']['y']))

                max_retries = 3
                retry_count = 0
                success = False
                error_msg = None
                
                while not success and retry_count < max_retries:
                    try:
                        # Create user message with game state and any previous error
                        if retry_count == 0:
                            message_content = {
                                "state": state,
                                "error": None
                            }
                        else:
                            message_content = {
                                "error": f"Error: {error_msg}"
                            }
                        
                        user_message = {
                            "role": "user",
                            "content": json.dumps(message_content)
                        }
                        
                        # Get model's response
                        response = player.get_response(user_message, game_image)
                        success, action, error_msg = self._process_action(player.player_id, response)
                        
                        if not success:
                            logging.warning(f"Failed to process action: {error_msg}")
                            retry_count += 1
                            continue
                        
                        # Execute the action
                        success, error = self._execute_action(player.player_id, action)
                        if success:
                            logging.info(f"Successfully executed action: {action['action_type']}")
                        else:
                            logging.warning(f"Failed to execute action: {error}")
                            error_msg = error
                            retry_count += 1
                            continue
                            
                    except Exception as e:
                        logging.warning(f"Error processing player {player.player_id}'s action: {str(e)}")
                        error_msg = str(e)
                        retry_count += 1
                        continue
                
                if not success:
                    logging.warning(f"Failed to process player {player.player_id}'s action after {max_retries} attempts")

                # Update world state
                self.world.update_time()
                world_time = self.world.get_agent_state(1)['world']['time']  # Get time info from any agent's state
                logging.info(f"Day/night cycle: Day {world_time['day']}, {'Night' if world_time['is_night'] else 'Day'}")
                
                # Save game state periodically
                if self.current_turn % 100 == 0:
                    self._save_game_state()
        
        logging.info("\nGame Over!")
        return self._generate_game_result()
    
    def _process_action(self, agent_id: int, action_str: str) -> Tuple[bool, Dict, str]:
        """Process an action from an agent
        
        Returns:
            Tuple[bool, Dict, str]: (success, action_dict, error_message)
        """
        try:
            # Strip any markdown code blocks
            action_str = action_str.strip()
            if action_str.startswith("```") and action_str.endswith("```"):
                action_str = "\n".join(action_str.split("\n")[1:-1])

            # Split into lines and process
            lines = action_str.strip().split("\n")
            reasoning = ""
            action = None
            
            for line in lines:
                line = line.strip()
                if line.startswith("REASONING:"):
                    reasoning = line[len("REASONING:"):].strip()
                elif line.startswith("ACTION:"):
                    action = line[len("ACTION:"):].strip()
            
            if not action:
                return False, {}, "No ACTION specified"

            # Parse the action based on format
            action_dict = {"reasoning": reasoning}
            
            # Split action into type and parameters
            action_parts = action.split(":", 1)
            if len(action_parts) != 2:
                return False, {}, "Invalid action format. Expected: ACTION_TYPE: parameters"
                
            action_type = action_parts[0].strip().lower()
            action_params = action_parts[1].strip()
            
            # MOVE: north|south|east|west
            if action_type == "move":
                direction = action_params.lower()
                if direction not in ["north", "south", "east", "west"]:
                    return False, {}, "Invalid move direction. Must be north, south, east, or west"
                action_dict.update({
                    "action_type": "move",
                    "parameters": {"direction": direction}
                })
            
            # GATHER: (x,y)
            elif action_type == "gather":
                if not action_params.startswith("(") or not action_params.endswith(")"):
                    return False, {}, "Invalid gather coordinates format. Expected: (x,y)"
                coords = action_params[1:-1].split(",")
                if len(coords) != 2:
                    return False, {}, "Invalid gather coordinates format. Expected: (x,y)"
                try:
                    x, y = map(int, [c.strip() for c in coords])
                    action_dict.update({
                        "action_type": "gather",
                        "parameters": {"resource_pos": {"x": x, "y": y}}
                    })
                except ValueError:
                    return False, {}, "Invalid gather coordinates. Must be integers"
            
            # BUILD: type (x,y)
            elif action_type == "build":
                parts = action_params.split(" ", 1)
                if len(parts) != 2:
                    return False, {}, "Invalid build format. Expected: type (x,y)"
                building_type = parts[0].lower()
                if building_type not in ["house", "farm", "well", "storage", "market"]:
                    return False, {}, "Invalid building type. Must be house, farm, well, storage, or market"
                coords_str = parts[1].strip()
                if not coords_str.startswith("(") or not coords_str.endswith(")"):
                    return False, {}, "Invalid build coordinates format. Expected: (x,y)"
                coords = coords_str[1:-1].split(",")
                if len(coords) != 2:
                    return False, {}, "Invalid build coordinates format. Expected: (x,y)"
                try:
                    x, y = map(int, [c.strip() for c in coords])
                    action_dict.update({
                        "action_type": "build",
                        "parameters": {
                            "type": building_type,
                            "position": {"x": x, "y": y}
                        }
                    })
                except ValueError:
                    return False, {}, "Invalid build coordinates. Must be integers"
            
            # TRADE: player_id offer_resource offer_amount request_resource request_amount
            elif action_type == "trade":
                parts = action_params.split(" ")
                if len(parts) != 5:
                    return False, {}, "Invalid trade format. Expected: trade player_id offer_resource offer_amount request_resource request_amount"
                try:
                    _, player_id, offer_resource, offer_amount, request_resource, request_amount = parts
                    player_id = int(player_id)
                    offer_amount = int(offer_amount)
                    request_amount = int(request_amount)
                    action_dict.update({
                        "action_type": "trade",
                        "parameters": {
                            "with_id": player_id,
                            "offer": {offer_resource: offer_amount},
                            "request": {request_resource: request_amount}
                        }
                    })
                except ValueError:
                    return False, {}, "Invalid trade parameters. Check number formats"
            
            # ENTER: (x,y)
            elif action_type == "enter":
                coords_str = action_params.strip()
                if not coords_str.startswith("(") or not coords_str.endswith(")"):
                    return False, {}, "Invalid enter coordinates format. Expected: (x,y)"
                coords = coords_str[1:-1].split(",")
                if len(coords) != 2:
                    return False, {}, "Invalid enter coordinates format. Expected: (x,y)"
                try:
                    x, y = map(int, [c.strip() for c in coords])
                    action_dict.update({
                        "action_type": "enter",
                        "parameters": {"building_pos": {"x": x, "y": y}}
                    })
                except ValueError:
                    return False, {}, "Invalid enter coordinates. Must be integers"
            
            # SAY: message
            elif action_type == "say":
                message = action_params.strip()  # Remove any leading/trailing whitespace
                action_dict.update({
                    "action_type": "say",
                    "parameters": {"message": message}
                })
            
            else:
                return False, {}, f"Invalid action format: {action}"

            return True, action_dict, ""
            
        except Exception as e:
            return False, {}, f"Error processing action: {str(e)}"

    def _execute_action(self, agent_id: int, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute an agent's action"""
        action_type = action.get("action_type")
        params = action.get("parameters", {})
        
        if action_type == "move":
            return self.world.move_agent(agent_id, params.get("direction", ""))
            
        elif action_type == "gather":
            pos = params.get("resource_pos", {})
            if not pos:
                return False, "Invalid resource position"
            return self.world.gather_resource(agent_id, Position(pos.get("x", 0), pos.get("y", 0)))
            
        elif action_type == "build":
            try:
                building_type = BuildingType[params.get("type", "")]
                pos = params.get("position", {})
                if not pos:
                    return False, "Invalid build position"
                return self.world.build(agent_id, building_type, Position(pos.get("x", 0), pos.get("y", 0)))
            except (KeyError, ValueError):
                return False, "Invalid building type"
            
        elif action_type == "trade":
            target_id = params.get("with_id")
            offer = params.get("offer", {})
            request = params.get("request", {})
            return self.world.trade(agent_id, target_id, offer, request)
            
        elif action_type == "enter":
            pos = params.get("building_pos", {})
            if not pos:
                return False, "Invalid building position"
            return self.world.enter_building(agent_id, Position(pos.get("x", 0), pos.get("y", 0)))
            
        elif action_type == "say":
            return self.world.say(agent_id, params.get("message", ""))
            
        return False, "Invalid action type"

    def _generate_game_image(self, agent_position: Tuple[int, int]) -> str:
        """Generate a visual representation of the game state centered on an agent"""
        # Constants
        CELL_SIZE = 16  
        VISION_RANGE = 16  # 16x16 grid centered on agent
        
        # Calculate view boundaries centered on agent
        center_x, center_y = agent_position
        start_x = max(0, center_x - VISION_RANGE // 2)
        start_y = max(0, center_y - VISION_RANGE // 2)
        end_x = min(self.world.size, start_x + VISION_RANGE)
        end_y = min(self.world.size, start_y + VISION_RANGE)
        
        # Create image
        width = (end_x - start_x) * CELL_SIZE
        height = (end_y - start_y) * CELL_SIZE
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
        except:
            font = ImageFont.load_default()
        
        # Draw base terrain
        terrain_colors = {
            TerrainType.WATER: 'blue',
            TerrainType.MOUNTAIN: 'gray',
            TerrainType.FOREST: 'darkgreen',
            TerrainType.PLAINS: 'khaki',
            TerrainType.FERTILE: 'forestgreen'
        }
        
        logging.info(f"Drawing terrain from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        for x in range(start_x, end_x):
            for y in range(start_y, end_y):
                terrain_type = TerrainType(self.world.terrain[y, x])
                color = terrain_colors[terrain_type]
                pixel_x = (x - start_x) * CELL_SIZE
                pixel_y = (y - start_y) * CELL_SIZE
                draw.rectangle(
                    [pixel_x, pixel_y, pixel_x + CELL_SIZE - 1, pixel_y + CELL_SIZE - 1],
                    fill=color,
                    outline='black'
                )

        # Draw resources
        resource_colors = {
            ResourceType.WOOD: 'sienna',
            ResourceType.STONE: 'silver',
            ResourceType.FOOD: 'gold',
            ResourceType.WATER: 'deepskyblue'
        }
        
        for pos, resource in self.world.resources.items():
            if start_x <= pos.x < end_x and start_y <= pos.y < end_y:
                color = resource_colors[resource.type]
                pixel_x = (pos.x - start_x) * CELL_SIZE
                pixel_y = (pos.y - start_y) * CELL_SIZE
                draw.rectangle(
                    [pixel_x, pixel_y, pixel_x + CELL_SIZE - 1, pixel_y + CELL_SIZE - 1],
                    fill=color,
                    outline='black'
                )

        # Draw buildings
        building_colors = {
            BuildingType.HOUSE: 'red',
            BuildingType.FARM: 'yellowgreen',
            BuildingType.WELL: 'steelblue',
            BuildingType.STORAGE: 'goldenrod',
            BuildingType.MARKET: 'darkorange'
        }
        
        for pos, building in self.world.buildings.items():
            if start_x <= pos.x < end_x and start_y <= pos.y < end_y:
                color = building_colors[building.type]
                pixel_x = (pos.x - start_x) * CELL_SIZE
                pixel_y = (pos.y - start_y) * CELL_SIZE
                draw.rectangle(
                    [pixel_x, pixel_y, pixel_x + CELL_SIZE - 1, pixel_y + CELL_SIZE - 1],
                    fill=color,
                    outline='black'
                )

        # Draw agents with more visibility
        for agent_id, agent in self.world.agents.items():
            if start_x <= agent.position.x < end_x and start_y <= agent.position.y < end_y:
                pixel_x = (agent.position.x - start_x) * CELL_SIZE
                pixel_y = (agent.position.y - start_y) * CELL_SIZE
                
                # Draw agent as a bright circle with black outline
                circle_bbox = [
                    pixel_x + 2, 
                    pixel_y + 2, 
                    pixel_x + CELL_SIZE - 3, 
                    pixel_y + CELL_SIZE - 3
                ]
                draw.ellipse(circle_bbox, fill='yellow', outline='black', width=2)
                
                # Add a dot in the middle
                center_x = pixel_x + CELL_SIZE // 2
                center_y = pixel_y + CELL_SIZE // 2
                dot_size = 4
                draw.ellipse(
                    [center_x - dot_size//2, center_y - dot_size//2, 
                     center_x + dot_size//2, center_y + dot_size//2],
                    fill='black'
                )
        
        # Draw grid coordinates
        for x in range(VISION_RANGE):
            draw.text(
                (x * CELL_SIZE + CELL_SIZE//3, 0),
                str((start_x + x) % 10),
                fill='black',
                font=font
            )
        for y in range(VISION_RANGE):
            draw.text(
                (0, y * CELL_SIZE + CELL_SIZE//3),
                str((start_y + y) % 10),
                fill='black',
                font=font
            )
        
        # Save both base64 and PNG file
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img.save(f"board.png")  # Save separate file for each agent
        logging.info(f"Saved game visualization to board_agent_{agent_id}.png ({width}x{height} pixels)")
        
        return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

    def _save_game_state(self):
        """Save current game state to file"""
        state_file = self.run_dir / f"game_state_turn_{self.current_turn}.json"
        
        state = {
            "turn": self.current_turn,
            "agents": {
                agent_id: {
                    "name": agent.name,
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "energy": agent.energy,
                    "inventory": {k.name: v for k, v in agent.inventory.items()}
                }
                for agent_id, agent in self.world.agents.items()
            },
            "buildings": {
                f"{pos.x},{pos.y}": {
                    "type": building.type.name,
                    "owner_id": building.owner_id
                }
                for pos, building in self.world.buildings.items()
            },
            "resources": {
                f"{pos.x},{pos.y}": {
                    "type": resource.type.name,
                    "amount": resource.amount
                }
                for pos, resource in self.world.resources.items()
            }
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _generate_game_result(self) -> Dict:
        """Generate final game results"""
        return {
            "turns_played": self.current_turn,
            "game_over": self.game_over,
            "final_state": {
                agent_id: {
                    "name": agent.name,
                    "inventory": {k.name: v for k, v in agent.inventory.items()},
                    "buildings_owned": len([b for b in self.world.buildings.values() if b.owner_id == agent_id])
                }
                for agent_id, agent in self.world.agents.items()
            },
            "logs": self.logs
        }
