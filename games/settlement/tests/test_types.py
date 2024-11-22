"""Unit tests for game types and data classes"""

import unittest
from ..types import (
    TerrainType, ResourceType, BuildingType, ActionType,
    Position, Resource, Building, Agent, WorldConfig
)

class TestPosition(unittest.TestCase):
    def test_position_creation(self):
        pos = Position(10, 20)
        self.assertEqual(pos.x, 10)
        self.assertEqual(pos.y, 20)
    
    def test_position_equality(self):
        pos1 = Position(10, 20)
        pos2 = Position(10, 20)
        pos3 = Position(20, 10)
        self.assertEqual(pos1, pos2)
        self.assertNotEqual(pos1, pos3)
    
    def test_position_hash(self):
        pos_dict = {Position(1, 2): "test"}
        self.assertEqual(pos_dict[Position(1, 2)], "test")

class TestResource(unittest.TestCase):
    def test_resource_creation(self):
        resource = Resource(ResourceType.WOOD, 100)
        self.assertEqual(resource.type, ResourceType.WOOD)
        self.assertEqual(resource.amount, 100)
    
    def test_resource_depletion(self):
        resource = Resource(ResourceType.STONE, 50)
        resource.amount -= 30
        self.assertEqual(resource.amount, 20)
    
    def test_resource_validation(self):
        with self.assertRaises(ValueError):
            Resource(ResourceType.WOOD, -10)

class TestBuilding(unittest.TestCase):
    def test_building_creation(self):
        building = Building(BuildingType.HOUSE, 1)
        self.assertEqual(building.type, BuildingType.HOUSE)
        self.assertEqual(building.owner_id, 1)
    
    def test_building_occupancy(self):
        building = Building(BuildingType.HOUSE, 1)
        self.assertEqual(len(building.occupants), 0)
        building.occupants.add(2)
        self.assertEqual(len(building.occupants), 1)
        self.assertTrue(2 in building.occupants)

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent("TestAgent", Position(0, 0))
    
    def test_agent_creation(self):
        self.assertEqual(self.agent.name, "TestAgent")
        self.assertEqual(self.agent.position, Position(0, 0))
        self.assertEqual(self.agent.energy, 100)  # Default energy
    
    def test_agent_inventory(self):
        self.assertEqual(len(self.agent.inventory), 0)
        self.agent.inventory[ResourceType.WOOD] = 10
        self.assertEqual(self.agent.inventory[ResourceType.WOOD], 10)
    
    def test_agent_energy_management(self):
        initial_energy = self.agent.energy
        self.agent.energy -= 20
        self.assertEqual(self.agent.energy, initial_energy - 20)
        
        with self.assertRaises(ValueError):
            self.agent.energy = -10

class TestWorldConfig(unittest.TestCase):
    def test_default_config(self):
        config = WorldConfig()
        self.assertEqual(config.size, 500)
        self.assertEqual(config.vision_range, 10)
        self.assertEqual(config.night_vision_range, 5)
    
    def test_custom_config(self):
        config = WorldConfig(
            size=100,
            vision_range=15,
            night_vision_range=8
        )
        self.assertEqual(config.size, 100)
        self.assertEqual(config.vision_range, 15)
        self.assertEqual(config.night_vision_range, 8)
    
    def test_config_validation(self):
        with self.assertRaises(ValueError):
            WorldConfig(size=-100)
        with self.assertRaises(ValueError):
            WorldConfig(vision_range=-5)
        with self.assertRaises(ValueError):
            WorldConfig(night_vision_range=-2)

if __name__ == '__main__':
    unittest.main()
