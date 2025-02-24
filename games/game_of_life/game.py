"""Conway's Game of Life implementation with LLM players"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
import numpy as np
from PIL import Image, ImageDraw
import io
import base64

@dataclass
class Position:
    """Position in the grid"""
    x: int
    y: int

@dataclass
class WorldConfig:
    """Configuration for the Game of Life world"""
    size: int
    view_radius: int = 3  # How far each agent can see

@dataclass
class Agent:
    """Agent in the Game of Life"""
    id: int
    position: Position
    life_source: int = 10  # Each agent starts with 10 life points

class GameOfLifeGame:
    """Game of Life implementation where LLM agents control cells"""
    
    def __init__(self, run_dir: Path, grid_size: int = 30, view_radius: int = 3, max_turns: int = 100):
        """Initialize the game
        
        Args:
            run_dir: Directory for game logs
            grid_size: Size of the grid (grid_size x grid_size)
            view_radius: How far each agent can see
            max_turns: Maximum number of turns before game ends
        """
        self.run_dir = run_dir
        self.max_turns = max_turns
        self.config = WorldConfig(size=grid_size, view_radius=view_radius)
        self.current_turn = 0
        self.game_over = False
        
        # Initialize grid
        self.grid = np.zeros((grid_size, grid_size), dtype=bool)
        
        # Track agents and their states
        self.agents: Dict[int, Agent] = {}  # agent_id -> Agent
        self.cell_owners: Dict[Tuple[int, int], int] = {}  # (x,y) -> agent_id
        
        # Track active agents
        self.active_agents = set()  # Set of agent IDs that are still alive
        
    def get_system_prompt(self) -> Dict[str, str]:
        """Get the game's system prompt"""
        return {
            "role": "system",
            "content": """Welcome to Conway's Game of Life! You are an intelligent agent controlling a living cell in this cellular automaton.

RULES OF THE GAME:
1. The universe is a grid where each cell is either alive (1) or dead (0)
2. Each cell interacts with its 8 neighbors (horizontal, vertical, diagonal)
3. The rules that determine the next generation are:

For living cells:
- Death by underpopulation: A live cell with <2 live neighbors dies
- Death by overcrowding: A live cell with >3 live neighbors dies
- Survival: A live cell with 2-3 live neighbors survives

For dead cells:
- Birth: A dead cell with exactly 3 live neighbors becomes alive

YOUR ROLE:
- You control a living cell in this grid
- Each turn you can see a local NxN area around your cell
- You can choose to:
  MOVE: UP/DOWN/LEFT/RIGHT to an adjacent cell
  STAY: Remain in your current position
  
STRATEGY TIPS:
1. Analyze your local neighborhood
2. Try to position yourself where you're likely to survive
3. Consider creating stable patterns with other agents
4. Avoid overcrowded or underpopulated areas

The game continues until either:
- All cells die
- A stable pattern is reached
- Maximum turns are reached

Provide your commands in the format:
COMMAND: direction

Example:
MOVE: UP
or
STAY"""
        }

    def _get_agent_state(self, agent_id: int) -> str:
        """Format the current state for the agent"""
        if agent_id not in self.agents:
            return "You are no longer alive in the grid."
            
        agent = self.agents[agent_id]
        pos = agent.position
        view_radius = self.config.view_radius
        
        # Get local grid view
        x_start = max(0, pos.x - view_radius)
        x_end = min(self.config.size, pos.x + view_radius + 1)
        y_start = max(0, pos.y - view_radius)
        y_end = min(self.config.size, pos.y + view_radius + 1)
        
        local_grid = self.grid[x_start:x_end, y_start:y_end]
        live_neighbors = np.sum(local_grid) - 1  # Subtract 1 to exclude self
        
        return f"""Turn {self.current_turn}

Your Position: ({pos.x}, {pos.y})
Life Source: {agent.life_source}/10
Live Neighbors: {live_neighbors}
Grid Size: {self.config.size}x{self.config.size}

Local Grid View:
{local_grid.astype(int)}

Status Assessment:
- Life Source: {"Critical" if agent.life_source <= 3 else "Stable" if agent.life_source >= 7 else "Moderate"}
- Underpopulation Risk (<2 neighbors): {"High" if live_neighbors < 2 else "Low"}
- Overcrowding Risk (>3 neighbors): {"High" if live_neighbors > 3 else "Low"}
- Stable Position (2-3 neighbors): {"Yes" if 2 <= live_neighbors <= 3 else "No"}

What's your next move? You can:
MOVE: UP/DOWN/LEFT/RIGHT
or
STAY"""

    def run(self, players: List['BaseLLMPlayer']) -> Dict:
        """Main game loop"""
        # Initialize grid with random positions for players
        self._initialize_players(players)
        
        # Initialize players with system prompt
        system_prompt = self.get_system_prompt()
        for player in players:
            if player.player_id in self.active_agents:
                player.initialize_with_prompt(system_prompt)
        
        # Generate initial board state
        self._generate_board_image()
        
        while not self.game_over and self.current_turn < self.max_turns:
            self.current_turn += 1
            
            # Process each player's turn immediately
            for player in players:
                if player.player_id not in self.active_agents:
                    continue
                
                # Generate agent-specific view
                agent_view = self._generate_agent_view(player.player_id)
                
                # Get agent state
                state = self._get_agent_state(player.player_id)
                
                # Create user message
                user_message = {
                    "role": "user",
                    "content": state
                }
                
                # Get model's response and process it immediately
                response = player.get_response(user_message, agent_view)
                self._process_action(player.player_id, response)

                self._generate_board_image()

            
            # Now apply life rules and update life sources
            self._apply_life_rules()
            
            # Generate final board state for this turn
            self._generate_board_image()
            
            # Check if game is over
            if len(self.active_agents) == 0:
                self.game_over = True
                break
        
        return self._generate_game_result()

    def _initialize_players(self, players: List['BaseLLMPlayer']):
        """Initialize random positions for players"""
        available_positions = [
            (x, y) for x in range(self.config.size)
            for y in range(self.config.size)
        ]
        random.shuffle(available_positions)
        
        for player in players:
            if not available_positions:
                break
                
            x, y = available_positions.pop()
            self.grid[x, y] = True
            agent = Agent(id=player.player_id, position=Position(x, y))
            self.agents[player.player_id] = agent
            self.cell_owners[(x, y)] = player.player_id
            self.active_agents.add(player.player_id)

    def _process_action(self, agent_id: int, action_str: str) -> bool:
        """Process an action from an agent"""
        if agent_id not in self.active_agents:
            return False
            
        current_pos = self.agents[agent_id].position
        
        # Parse action
        action_str = action_str.strip().upper()
        if "STAY" in action_str:
            self._generate_board_image()  # Update board even on STAY
            return True
            
        if "MOVE:" not in action_str:
            return False
            
        direction = action_str.split(":", 1)[1].strip()
        
        # Calculate new position
        new_pos = Position(current_pos.x, current_pos.y)
        if direction == "UP" and current_pos.x > 0:
            new_pos.x -= 1
        elif direction == "DOWN" and current_pos.x < self.config.size - 1:
            new_pos.x += 1
        elif direction == "LEFT" and current_pos.y > 0:
            new_pos.y -= 1
        elif direction == "RIGHT" and current_pos.y < self.config.size - 1:
            new_pos.y += 1
        else:
            return False
            
        # Always update the grid and tracking
        # First clear the old position
        self.grid[current_pos.x, current_pos.y] = False
        if (current_pos.x, current_pos.y) in self.cell_owners:
            del self.cell_owners[(current_pos.x, current_pos.y)]
        
        # Then update the new position
        self.grid[new_pos.x, new_pos.y] = True
        self.cell_owners[(new_pos.x, new_pos.y)] = agent_id
        self.agents[agent_id].position = new_pos
        
        # Update board immediately after position change
        self._generate_board_image()
            
        return True

    def _apply_life_rules(self):
        """Apply Conway's Game of Life rules and update life sources"""
        new_grid = np.zeros_like(self.grid)
        new_cell_owners = {}
        new_agents = {}
        new_active_agents = set()
        
        # Calculate next state for each cell
        for x in range(self.config.size):
            for y in range(self.config.size):
                # Count live neighbors
                x_start = max(0, x - 1)
                x_end = min(self.config.size, x + 2)
                y_start = max(0, y - 1)
                y_end = min(self.config.size, y + 2)
                
                neighbors = np.sum(self.grid[x_start:x_end, y_start:y_end]) - self.grid[x, y]
                
                # Apply rules
                if self.grid[x, y]:  # Live cell
                    agent_id = self.cell_owners.get((x, y))
                    if agent_id and agent_id in self.agents:
                        agent = self.agents[agent_id]
                        
                        # More forgiving survival conditions
                        if 2 <= neighbors <= 3:
                            # Optimal conditions - gain 2 life points
                            agent.life_source = min(10, agent.life_source + 2)
                            new_grid[x, y] = True
                            new_cell_owners[(x, y)] = agent_id
                            new_agents[agent_id] = agent
                            new_active_agents.add(agent_id)
                        elif neighbors == 1:
                            # Lonely but surviving - no life change
                            new_grid[x, y] = True
                            new_cell_owners[(x, y)] = agent_id
                            new_agents[agent_id] = agent
                            new_active_agents.add(agent_id)
                        elif neighbors == 4:
                            # Slightly crowded but surviving - lose 1 life point
                            agent.life_source = max(1, agent.life_source - 1)
                            new_grid[x, y] = True
                            new_cell_owners[(x, y)] = agent_id
                            new_agents[agent_id] = agent
                            new_active_agents.add(agent_id)
                        else:
                            # Very unfavorable conditions - lose 2 life points
                            agent.life_source -= 2
                            if agent.life_source > 0:
                                new_grid[x, y] = True
                                new_cell_owners[(x, y)] = agent_id
                                new_agents[agent_id] = agent
                                new_active_agents.add(agent_id)
                else:  # Dead cell
                    if neighbors == 3:
                        # Birth - create new cell with inherited life source
                        neighbor_owners = []
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = x + dx, y + dy
                                if (0 <= nx < self.config.size and 
                                    0 <= ny < self.config.size and 
                                    (nx, ny) in self.cell_owners):
                                    neighbor_owners.append(self.cell_owners[(nx, ny)])
                        
                        if neighbor_owners:
                            parent_id = max(set(neighbor_owners), key=neighbor_owners.count)
                            if parent_id in self.agents:
                                parent = self.agents[parent_id]
                                # New cell inherits more life source
                                inherited_life = max(3, parent.life_source // 2)  # Minimum 3 life for new cells
                                parent.life_source = max(2, parent.life_source - inherited_life // 2)  # Parent loses less
                                
                                new_grid[x, y] = True
                                new_cell_owners[(x, y)] = parent_id
                                new_agents[parent_id] = parent
                                new_active_agents.add(parent_id)
        
        # Update game state
        self.grid = new_grid
        self.cell_owners = new_cell_owners
        self.agents = new_agents
        self.active_agents = new_active_agents

    def _generate_board_image(self) -> None:
        """Generate a visual representation of the full game board"""
        CELL_SIZE = 20
        width = self.config.size * CELL_SIZE
        height = self.config.size * CELL_SIZE
        
        # Create image with white background
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw cells
        for x in range(self.config.size):
            for y in range(self.config.size):
                if self.grid[x, y]:
                    color = 'black'
                    if (x, y) in self.cell_owners:
                        # Use different colors for different agents
                        agent_id = self.cell_owners[(x, y)]
                        color = f'#{hash(str(agent_id)) % 0xFFFFFF:06x}'
                    
                    draw.rectangle(
                        [y * CELL_SIZE, x * CELL_SIZE, 
                         (y + 1) * CELL_SIZE - 1, (x + 1) * CELL_SIZE - 1],
                        fill=color
                    )
        
        # Draw grid
        for i in range(self.config.size + 1):
            draw.line([(i * CELL_SIZE, 0), (i * CELL_SIZE, height)], fill='gray')
            draw.line([(0, i * CELL_SIZE), (width, i * CELL_SIZE)], fill='gray')
        
        # Save the image
        img.save(self.run_dir / 'board.png')

    def _generate_agent_view(self, agent_id: int) -> str:
        """Generate a visual representation of the agent's local view"""
        if agent_id not in self.agents:
            return ""
            
        pos = self.agents[agent_id].position
        view_radius = self.config.view_radius
        CELL_SIZE = 30
        
        # Calculate view boundaries
        x_start = max(0, pos.x - view_radius)
        x_end = min(self.config.size, pos.x + view_radius + 1)
        y_start = max(0, pos.y - view_radius)
        y_end = min(self.config.size, pos.y + view_radius + 1)
        
        width = (x_end - x_start) * CELL_SIZE
        height = (y_end - y_start) * CELL_SIZE
        
        # Create image with white background
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw visible cells
        for x in range(x_start, x_end):
            for y in range(y_start, y_end):
                if self.grid[x, y]:
                    color = 'gray'
                    if x == pos.x and y == pos.y:
                        color = 'blue'  # Current agent
                    elif (x, y) in self.cell_owners:
                        color = 'black'  # Other agents
                    
                    draw.rectangle(
                        [(y - y_start) * CELL_SIZE, (x - x_start) * CELL_SIZE,
                         (y - y_start + 1) * CELL_SIZE - 1, (x - x_start + 1) * CELL_SIZE - 1],
                        fill=color
                    )
        
        # Draw grid
        for i in range(x_end - x_start + 1):
            draw.line([(i * CELL_SIZE, 0), (i * CELL_SIZE, height)], fill='gray')
        for i in range(y_end - y_start + 1):
            draw.line([(0, i * CELL_SIZE), (width, i * CELL_SIZE)], fill='gray')
        
        # Save and encode image
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img.save(self.run_dir / 'agent.png')
        return f"data:image/png;base64,{base64.b64encode(img_byte_arr).decode()}"

    def _generate_game_result(self) -> Dict:
        """Generate the final game result"""
        return {
            "turns_played": self.current_turn,
            "final_live_cells": int(np.sum(self.grid)),
            "surviving_agents": len(self.active_agents),
            "game_over_reason": "All cells died" if len(self.active_agents) == 0 else "Max turns reached"
        }
