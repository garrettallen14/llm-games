"""Unit tests for the World class and its functionality"""

import unittest
import numpy as np
from ..world import World
from ..types import (
    TerrainType, ResourceType, BuildingType, ActionType,
    Position, Resource, Building, Agent, WorldConfig
)

class TestWorld(unittest.TestCase):
    def setUp(self):
        self.config = WorldConfig(size=50)  # Smaller size for testing
        self.world = World(self.config)
    
    def test_world_initialization(self):
        self.assertEqual(self.world.size, 50)
        self.assertEqual(self.world.terrain.shape, (50, 50))
        self.assertTrue(len(self.world.resources) > 0)
        self.assertEqual(len(self.world.agents), 0)
        self.assertEqual(len(self.world.buildings), 0)
    
    def test_terrain_generation(self):
        # Check if all terrain types are present
        unique_terrains = set(self.world.terrain.flatten())
        for terrain_type in TerrainType:
            self.assertIn(terrain_type.value, unique_terrains)
    
    def test_resource_generation(self):
        # Check if resources are properly placed
        for pos, resource in self.world.resources.items():
            self.assertIsInstance(pos, Position)
            self.assertIsInstance(resource, Resource)
            self.assertGreater(resource.amount, 0)
    
    def test_add_agent(self):
        agent_id = self.world.add_agent("TestAgent")
        self.assertIn(agent_id, self.world.agents)
        agent = self.world.agents[agent_id]
        self.assertEqual(agent.name, "TestAgent")
        self.assertIsInstance(agent.position, Position)
        self.assertTrue(0 <= agent.position.x < self.world.size)
        self.assertTrue(0 <= agent.position.y < self.world.size)
    
    def test_move_agent(self):
        agent_id = self.world.add_agent("TestAgent")
        original_pos = self.world.agents[agent_id].position
        
        # Test valid move
        success, _ = self.world.move_agent(agent_id, "north")
        if success:
            new_pos = self.world.agents[agent_id].position
            self.assertNotEqual(original_pos, new_pos)
        
        # Test invalid move (out of bounds)
        for _ in range(self.world.size + 1):
            self.world.move_agent(agent_id, "north")
        self.assertTrue(self.world.agents[agent_id].position.y >= 0)
    
    def test_gather_resource(self):
        # Place agent next to resource
        agent_id = self.world.add_agent("TestAgent")
        resource_pos = next(iter(self.world.resources.keys()))
        self.world.agents[agent_id].position = Position(
            resource_pos.x, resource_pos.y + 1
        )
        
        # Test gathering
        success, _ = self.world.gather_resource(agent_id, resource_pos)
        if success:
            self.assertIn(
                self.world.resources[resource_pos].type,
                self.world.agents[agent_id].inventory
            )
    
    def test_build(self):
        agent_id = self.world.add_agent("TestAgent")
        agent = self.world.agents[agent_id]
        pos = Position(agent.position.x + 1, agent.position.y)
        
        # Add required resources
        agent.inventory[ResourceType.WOOD] = 100
        agent.inventory[ResourceType.STONE] = 100
        
        success, _ = self.world.build(agent_id, BuildingType.HOUSE, pos)
        if success:
            self.assertIn(pos, self.world.buildings)
            self.assertEqual(self.world.buildings[pos].type, BuildingType.HOUSE)
            self.assertEqual(self.world.buildings[pos].owner_id, agent_id)
    
    def test_trade(self):
        agent1_id = self.world.add_agent("Agent1")
        agent2_id = self.world.add_agent("Agent2")
        
        # Setup inventories
        self.world.agents[agent1_id].inventory[ResourceType.WOOD] = 50
        self.world.agents[agent2_id].inventory[ResourceType.STONE] = 50
        
        # Place agents next to each other
        self.world.agents[agent2_id].position = Position(
            self.world.agents[agent1_id].position.x + 1,
            self.world.agents[agent1_id].position.y
        )
        
        # Test trading
        success, _ = self.world.trade(
            agent1_id, agent2_id,
            {ResourceType.WOOD: 20},
            {ResourceType.STONE: 20}
        )
        
        if success:
            self.assertEqual(self.world.agents[agent1_id].inventory[ResourceType.WOOD], 30)
            self.assertEqual(self.world.agents[agent1_id].inventory[ResourceType.STONE], 20)
            self.assertEqual(self.world.agents[agent2_id].inventory[ResourceType.WOOD], 20)
            self.assertEqual(self.world.agents[agent2_id].inventory[ResourceType.STONE], 30)
    
    def test_time_system(self):
        self.assertEqual(self.world.current_turn, 0)
        self.assertFalse(self.world.is_night)
        
        # Test day/night cycle
        for _ in range(12):
            self.world.update_time()
        self.assertTrue(self.world.is_night)
        
        for _ in range(12):
            self.world.update_time()
        self.assertFalse(self.world.is_night)
    
    def test_vision_system(self):
        agent_id = self.world.add_agent("TestAgent")
        state = self.world.get_agent_state(agent_id)
        
        # Test day vision
        self.assertIn("local_view", state["world"])
        day_view = state["world"]["local_view"]
        
        # Force night time
        for _ in range(12):
            self.world.update_time()
        
        night_state = self.world.get_agent_state(agent_id)
        night_view = night_state["world"]["local_view"]
        
        # Night view should be smaller
        self.assertTrue(len(night_view) < len(day_view))
    
    def test_energy_system(self):
        agent_id = self.world.add_agent("TestAgent")
        initial_energy = self.world.agents[agent_id].energy
        
        # Test energy consumption
        self.world.move_agent(agent_id, "north")
        self.assertTrue(self.world.agents[agent_id].energy < initial_energy)
        
        # Test night energy costs
        for _ in range(12):
            self.world.update_time()
        
        night_energy = self.world.agents[agent_id].energy
        self.world.move_agent(agent_id, "south")
        self.assertTrue(
            self.world.agents[agent_id].energy < night_energy - 1
        )  # Night movement costs more

if __name__ == '__main__':
    unittest.main()
