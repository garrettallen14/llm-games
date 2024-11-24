"""Constants for the Color Chat game"""

# World Generation
WORLD_MIN_SIZE = 5
WORLD_MAX_SIZE = 100
DEFAULT_WORLD_SIZE = 20
MAX_WATER_PERCENTAGE = 0.01  # Maximum 25% of map can be water

# Movement and Energy
DEFAULT_ENERGY = 100
MOVE_ENERGY_COST = 1
TREE_MOVE_ENERGY_COST = 5
WATER_ADJACENT_ENERGY_GAIN = 1
SAME_COLOR_ENERGY_GAIN = 1

# Resource System
MAX_WOOD_PER_TILE = 5
MAX_WOOD_INVENTORY = 1
WOOD_GATHER_ENERGY = 10
WOOD_CARRY_MULTIPLIER = 2
WOOD_RESPAWN_CHANCE = 0.01  # 1% chance for wood to respawn each turn

# Shelter System
SHELTER_WOOD_REQUIRED = 1
SHELTER_MAX_STORAGE = 30
SHELTER_OWNER_ENERGY = 2
SHELTER_ADJACENT_ENERGY = 1

# Vision and Communication
DEFAULT_COMMUNICATION_RADIUS = 9
DEFAULT_FOV_RADIUS = 9
RECENT_MESSAGES_MEMORY = 20

# Terrain Generation
WATER_POOL_COUNT = 4
WATER_CLEARING_RADIUS = 3  # Blocks from water where trees won't generate
RIVER_WIDENING_CHANCE = 0.3

# Visualization
CELL_SIZE = 50
AGENT_VIEW_DISTANCE = 3
AGENT_PADDING = 5  # Padding for agent squares in visualization

# Colors
WATER_COLOR = '#4287f5'  # Light blue
TREE_COLOR = '#90EE90'  # Light green
TREE_SYMBOL_COLOR = '#006400'  # Dark green

# Console Colors
CONSOLE_COLORS = {
    1: '\033[95m',  # Magenta
    2: '\033[94m',  # Blue
    3: '\033[92m',  # Green
    4: '\033[93m',  # Yellow
    5: '\033[91m',  # Red
    6: '\033[96m',  # Cyan
    7: '\033[97m',  # White
}
CONSOLE_END = '\033[0m'
CONSOLE_BOLD = '\033[1m'

# Symbols
WATER_SYMBOL = "〰️"
TREE_SYMBOL = "🌳"
WOOD_SYMBOL = "🪵"
