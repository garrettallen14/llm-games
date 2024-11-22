"""Main game implementation for Color Chat"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random
import logging
from PIL import Image, ImageDraw
import io
import base64

from games.color_chat.types import Position, Agent, WorldConfig

class ColorChatGame:
    """Main game class that interfaces with the runner"""
    
    def __init__(self, run_dir: Path, max_turns: int = 100, world_size: int = 10, communication_radius: int = 4):
        """Initialize the game
        
        Args:
            run_dir: Directory for game logs
            max_turns: Maximum number of turns before game ends
            world_size: Size of the grid (world_size x world_size)
            communication_radius: How far messages can be heard
        """
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.config = WorldConfig(size=world_size, communication_radius=communication_radius)
        self.agents: Dict[int, Agent] = {}
        self.current_turn = 0
        self.game_over = False
        self.messages: List[Dict[str, Any]] = []
        # Track action results for feedback
        self.action_results: Dict[int, List[Dict[str, Any]]] = {}
        
        # Setup colorful console output
        self.colors = {
            1: '\033[95m',  # Magenta
            2: '\033[94m',  # Blue
            3: '\033[92m',  # Green
            4: '\033[93m',  # Yellow
            5: '\033[91m',  # Red
            6: '\033[96m',  # Cyan
            7: '\033[97m',  # White
        }
        self.ENDC = '\033[0m'
        self.BOLD = '\033[1m'
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Get the game's system prompt"""
        return {
            "role": "system",
            "content": """Commands available each turn:
MOVE: UP/DOWN/LEFT/RIGHT     # You can move multiple times
SPEAK: message              # Send one message per turn
COLOR: (R,G,B)             # Change color once per turn

IMPORTANT: Use multiple commands each turn!
Example multi-command turn:
MOVE: direction
MOVE: direction
...
SPEAK: message
COLOR: (R,G,B)

Format:
- Start each command on a new line
- Use exact command names: MOVE, SPEAK, COLOR
- No empty lines between commands"""
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
            self._print_turn_header()
            self.current_turn += 1
            
            # Process each player's turn
            for player in players:
                # Clear previous turn's results
                self.action_results[player.player_id] = []
                
                # Get agent state
                state = self._get_agent_state(player.player_id)
                
                # Generate game visualization
                game_image = self._generate_game_image()
                
                # Create user message with previous results
                user_message = {
                    "role": "user",
                    "content": json.dumps({
                        "state": state,
                        "previous_actions": self.action_results[player.player_id],
                        "message": f"You are at position ({state['position']['x']}, {state['position']['y']}). What would you like to do?"
                    })
                }
                
                # Get model's response
                response = player.get_response(user_message, game_image)
                
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
                
        return self._generate_game_result()

    def _initialize_agents(self, num_agents: int):
        """Initialize agents at random positions"""
        available_positions = [
            Position(x, y)
            for x in range(self.config.size)
            for y in range(self.config.size)
        ]
        random.shuffle(available_positions)
        
        for i in range(num_agents):
            pos = available_positions.pop()
            self.agents[i + 1] = Agent(
                id=i + 1,
                position=pos
            )
    
    def _get_agent_state(self, agent_id: int) -> Dict[str, Any]:
        """Get complete state for an agent"""
        agent = self.agents[agent_id]
        
        # Get only visible agents (within communication radius)
        visible_agents = [
            {
                "id": other.id,
                "position": {"x": other.position.x, "y": other.position.y},
                "color": other.color,
                "last_message": other.last_message
            }
            for other in self._get_agents_in_range(agent.position)
        ]
        
        # Get recent messages within radius
        recent_messages = self._get_recent_messages(agent.position)
        
        return {
            "position": {"x": agent.position.x, "y": agent.position.y},
            "color": agent.color,
            "nearby": {
                "squares": visible_agents,
                "messages": recent_messages
            }
        }
    
    def _get_recent_messages(self, position: Position) -> List[Dict[str, Any]]:
        """Get recent messages visible from position"""
        recent_messages = []
        
        for msg in reversed(self.messages[-20:]):  # Last 20 messages
            msg_pos = Position(msg["position"]["x"], msg["position"]["y"])
            if position.distance_to(msg_pos) <= self.config.communication_radius:
                recent_messages.append(msg)
        
        return recent_messages
    
    def _parse_actions(self, response: str) -> List[str]:
        """Parse multiple actions from response"""
        actions = []
        seen_speak = False
        seen_color = False
        
        # Split response into lines and process each line
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and non-action lines
            if not line or not any(cmd in line for cmd in ["MOVE:", "SPEAK:", "COLOR:"]):
                continue
                
            # Validate command format
            if ":" not in line:
                continue
                
            command, content = line.split(":", 1)
            command = command.strip().upper()
            
            # Apply command limits
            if command == "SPEAK":
                if seen_speak:
                    continue
                seen_speak = True
            elif command == "COLOR":
                if seen_color:
                    continue
                seen_color = True
            
            actions.append(line)
            
        return actions

    def _validate_move(self, direction: str, agent: Agent) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate a move action before executing it"""
        direction = direction.upper()
        
        # Validate direction format
        if direction not in ["UP", "DOWN", "LEFT", "RIGHT"]:
            return False, "Invalid direction", {
                "valid_directions": ["UP", "DOWN", "LEFT", "RIGHT"],
                "received": direction,
                "hint": "Direction must be exactly UP, DOWN, LEFT, or RIGHT (case insensitive)"
            }
        
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
            return False, "Out of bounds", {
                "current_position": {"x": agent.position.x, "y": agent.position.y},
                "attempted_position": {"x": new_x, "y": new_y},
                "grid_size": self.config.size,
                "hint": "Movement would take you outside the 10x10 grid"
            }
        
        # Check collisions
        if any(a.position.x == new_x and a.position.y == new_y for a in self.agents.values()):
            return False, "Position occupied", {
                "attempted_position": {"x": new_x, "y": new_y},
                "hint": "Another agent is already at this position"
            }
        
        return True, "Valid move", {"new_position": {"x": new_x, "y": new_y}}

    def _validate_color(self, color_str: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate a color action before executing it"""
        try:
            # Check format
            if not (color_str.startswith("(") and color_str.endswith(")")):
                return False, "Invalid format", {
                    "received": color_str,
                    "expected_format": "(R,G,B)",
                    "hint": "Color must be in (R,G,B) format with parentheses"
                }
            
            # Parse values
            color_str = color_str.strip("()")
            try:
                r, g, b = map(int, color_str.split(","))
            except ValueError:
                return False, "Invalid color values", {
                    "received": color_str,
                    "hint": "Must provide exactly three numbers separated by commas"
                }
            
            # Validate ranges
            if not all(0 <= c <= 255 for c in (r, g, b)):
                return False, "Color values out of range", {
                    "received": (r, g, b),
                    "valid_range": "0-255",
                    "hint": "Each color component must be between 0 and 255"
                }
            
            return True, "Valid color", {"color": (r, g, b)}
            
        except Exception as e:
            return False, "Invalid color format", {
                "received": color_str,
                "error": str(e),
                "hint": "Use format (R,G,B) with numbers 0-255"
            }

    def _process_action(self, agent_id: int, action_str: str) -> Tuple[bool, str]:
        """Process an action from an agent"""
        try:
            # Parse the action command and content
            if ":" not in action_str:
                return False, "Invalid action format"
                
            command, content = action_str.split(":", 1)
            command = command.strip().upper()
            content = content.strip()
            
            # Process based on command type
            if command == "MOVE":
                return self._handle_move(agent_id, content)
            elif command == "SPEAK":
                return self._handle_speak(agent_id, content)
            elif command == "COLOR":
                return self._handle_color(agent_id, content)
            
            return False, f"Unknown command: {command}"
            
        except Exception as e:
            return False, f"Error processing action: {str(e)}"

    def _handle_speak(self, agent_id: int, message: str) -> Tuple[bool, str]:
        """Handle a speak action"""
        agent = self.agents[agent_id]
        agent.last_message = message
        
        self.messages.append({
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

    def _generate_game_image(self) -> str:
        """Generate a visual representation of the game state"""
        CELL_SIZE = 50
        width = self.config.size * CELL_SIZE
        height = self.config.size * CELL_SIZE
        
        # Create image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw grid
        for i in range(self.config.size + 1):
            draw.line([(i * CELL_SIZE, 0), (i * CELL_SIZE, height)], fill='black')
            draw.line([(0, i * CELL_SIZE), (width, i * CELL_SIZE)], fill='black')
        
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
    
    def _generate_game_result(self) -> Dict:
        """Generate final game results"""
        return {
            "turns_played": self.current_turn,
            "game_over": self.game_over,
            "final_state": {
                agent_id: {
                    "position": {"x": agent.position.x, "y": agent.position.y},
                    "color": agent.color,
                    "last_message": agent.last_message
                }
                for agent_id, agent in self.agents.items()
            }
        }

    def _print_turn_header(self):
        """Print a formatted turn header"""
        print(f"\n{self.BOLD}{'='*50}{self.ENDC}")
        print(f"{self.BOLD}🎮 Turn {self.current_turn}{self.ENDC}")
        print(f"{self.BOLD}{'='*50}{self.ENDC}\n")
        
    def _print_agent_action(self, agent_id: int, action: str, result: Dict[str, Any]):
        """Print a formatted agent action and result"""
        agent = self.agents[agent_id]
        color = self.colors.get(agent_id, '')
        pos = agent.position
        
        # Print agent header
        print(f"{color}👤 Agent {agent_id} at ({pos.x}, {pos.y}){self.ENDC}")
        
        # Print action and result
        if result["success"]:
            if action.startswith("SPEAK:"):
                message = action[6:].strip()
                print(f"{color}💬 Says: {message}{self.ENDC}")
                # Print who heard the message
                hearers = self._get_agents_in_range(agent.position)
                if len(hearers) > 1:  # More than just the speaker
                    hearer_str = ", ".join([f"Agent {a.id}" for a in hearers if a.id != agent_id])
                    print(f"{color}👂 Heard by: {hearer_str}{self.ENDC}")
            elif action.startswith("MOVE:"):
                print(f"{color}🚶 {result['result']}{self.ENDC}")
            elif action.startswith("COLOR:"):
                print(f"{color}🎨 {result['result']}{self.ENDC}")
        else:
            print(f"{color}❌ Failed: {result['result']}{self.ENDC}")
        print()
        
    def _get_agents_in_range(self, position: Position) -> List[Agent]:
        """Get all agents within communication range of a position"""
        in_range = []
        for agent in self.agents.values():
            if position.distance_to(agent.position) <= self.config.communication_radius:
                in_range.append(agent)
        return in_range
