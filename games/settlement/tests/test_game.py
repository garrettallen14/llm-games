"""Integration tests for the Settlement game"""

import unittest
import json
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import io
import base64

from ..game import SettlementGame
from ..world import World, WorldConfig
from ..types import (
    TerrainType, ResourceType, BuildingType, ActionType,
    Position, Resource, Building, Agent
)

class MockLLMPlayer:
    """Mock LLM player for testing"""
    def __init__(self, player_id: int, actions: list):
        self.player_id = player_id
        self.actions = actions
        self.current_action = 0
        self.system_prompt = None
    
    def initialize_with_prompt(self, prompt):
        self.system_prompt = prompt
    
    def get_response(self, message, game_image=None):
        if self.current_action >= len(self.actions):
            return json.dumps({
                "action_type": "say",
                "parameters": {"message": "No more actions"},
                "reasoning": "Test complete"
            })
        
        action = self.actions[self.current_action]
        self.current_action += 1
        return json.dumps(action)

class TestSettlementGame(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for game files
        self.test_dir = Path(tempfile.mkdtemp())
        self.game = SettlementGame(self.test_dir, max_turns=100)
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_game_initialization(self):
        self.assertEqual(self.game.current_turn, 0)
        self.assertFalse(self.game.game_over)
        self.assertEqual(len(self.game.logs), 0)
    
    def test_system_prompt(self):
        prompt = self.game.get_system_prompt()
        self.assertIsInstance(prompt, dict)
        self.assertEqual(prompt["role"], "system")
        self.assertIn("You are an agent", prompt["content"])
    
    def test_game_visualization(self):
        # Add an agent to visualize
        self.game.world.add_agent("TestAgent")
        
        # Generate visualization
        image_data = self.game._generate_game_image()
        
        # Verify it's a valid base64 PNG
        self.assertTrue(image_data.startswith("data:image/png;base64,"))
        
        # Decode and verify image
        img_data = base64.b64decode(image_data.split(",")[1])
        img = Image.open(io.BytesIO(img_data))
        self.assertEqual(img.size, (self.game.world.size * 8, self.game.world.size * 8))
    
    def test_game_state_saving(self):
        # Add agent and make some changes
        agent_id = self.game.world.add_agent("TestAgent")
        self.game.world.agents[agent_id].inventory[ResourceType.WOOD] = 50
        
        # Save state
        self.game._save_game_state()
        
        # Verify state file exists and is valid JSON
        state_file = self.test_dir / f"game_state_turn_{self.game.current_turn}.json"
        self.assertTrue(state_file.exists())
        
        with open(state_file) as f:
            state = json.load(f)
            self.assertEqual(state["turn"], self.game.current_turn)
            self.assertIn(str(agent_id), state["agents"])
    
    def test_basic_game_loop(self):
        # Create mock players with simple actions
        players = [
            MockLLMPlayer(0, [
                {
                    "action_type": "move",
                    "parameters": {"direction": "north"},
                    "reasoning": "Moving north"
                },
                {
                    "action_type": "gather",
                    "parameters": {"resource_position": [5, 5]},
                    "reasoning": "Gathering resources"
                }
            ]),
            MockLLMPlayer(1, [
                {
                    "action_type": "move",
                    "parameters": {"direction": "south"},
                    "reasoning": "Moving south"
                },
                {
                    "action_type": "say",
                    "parameters": {"message": "Hello"},
                    "reasoning": "Greeting"
                }
            ])
        ]
        
        # Run game
        result = self.game.run(players)
        
        # Verify game completion
        self.assertGreater(result["turns_played"], 0)
        self.assertTrue(len(result["logs"]) > 0)
        self.assertIn("final_state", result)
    
    def test_complex_interaction(self):
        # Test complex interaction between players
        players = [
            MockLLMPlayer(0, [
                # Player 0 gathers resources
                {
                    "action_type": "gather",
                    "parameters": {"resource_position": [5, 5]},
                    "reasoning": "Gathering wood"
                },
                # Then tries to build
                {
                    "action_type": "build",
                    "parameters": {
                        "building_type": "HOUSE",
                        "position": [6, 5]
                    },
                    "reasoning": "Building house"
                }
            ]),
            MockLLMPlayer(1, [
                # Player 1 moves closer to trade
                {
                    "action_type": "move",
                    "parameters": {"direction": "east"},
                    "reasoning": "Moving to trade"
                },
                # Attempts trade
                {
                    "action_type": "trade",
                    "parameters": {
                        "target_id": 0,
                        "offer": {"STONE": 10},
                        "request": {"WOOD": 10}
                    },
                    "reasoning": "Trading resources"
                }
            ])
        ]
        
        result = self.game.run(players)
        
        # Verify interactions were logged
        action_types = [log["action"]["action_type"] for log in result["logs"]]
        self.assertIn("gather", action_types)
        self.assertIn("build", action_types)
        self.assertIn("move", action_types)
        self.assertIn("trade", action_types)
    
    def test_error_handling(self):
        # Test with invalid actions
        players = [
            MockLLMPlayer(0, [
                {
                    "action_type": "invalid_action",
                    "parameters": {},
                    "reasoning": "This should fail"
                },
                {
                    "action_type": "move",
                    "parameters": {"direction": "invalid_direction"},
                    "reasoning": "This should fail"
                }
            ])
        ]
        
        result = self.game.run(players)
        
        # Verify errors were logged
        self.assertTrue(any(
            not log.get("success", False)
            for log in result["logs"]
            if "success" in log
        ))
    
    def test_game_termination(self):
        # Test game ends properly at max turns
        game = SettlementGame(self.test_dir, max_turns=5)
        players = [
            MockLLMPlayer(0, [
                {
                    "action_type": "move",
                    "parameters": {"direction": "north"},
                    "reasoning": "Moving"
                }
            ] * 10)  # More actions than max_turns
        ]
        
        result = game.run(players)
        
        # Verify game ended at max_turns
        self.assertEqual(result["turns_played"], 5)

if __name__ == '__main__':
    unittest.main()
