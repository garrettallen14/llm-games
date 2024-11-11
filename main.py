import argparse
import json
import os
import glob
import asyncio
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

from game import GameState, GameRunner
from llm import LLMPlayer
from analysis import ChessAnalyzer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GameMetadata:
    game_id: str
    white_model: str
    black_model: str
    evaluation_list: List[float]
    winner: Optional[int] = None  # None=ongoing, 0=white, 1=black, 2=tie
    current_position: Optional[str] = None  # FEN string
    moves: List[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'GameMetadata':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

def sanitize_model_name(name: str) -> str:
    """Clean model names for file system use"""
    return name.strip('/').replace('/', '_').replace('\\', '_').replace(' ', '_')

def get_folder_name(white: str, black: str, game_id: Optional[Union[str, List[int], int]] = None) -> str:
    """Generate folder name based on models and game ID"""
    base_dir = Path("game_data")
    base_dir.mkdir(exist_ok=True)
    
    white_clean = sanitize_model_name(white)
    black_clean = sanitize_model_name(black)
    prefix = f"{white_clean}_{black_clean}"
    
    if game_id is None:
        # Find next available ID
        pattern = base_dir / f"{prefix}_*"
        existing = glob.glob(str(pattern))
        next_id = max([0] + [int(p.split('_')[-1]) for p in existing if p.split('_')[-1].isdigit()]) + 1
        return str(base_dir / f"{prefix}_{next_id}")
    elif game_id == "all":
        return str(base_dir / prefix)
    else:
        if isinstance(game_id, list):
            return [str(base_dir / f"{prefix}_{gid}") for gid in game_id]
        return str(base_dir / f"{prefix}_{game_id}")

def initialize_game_directory(directory: str, white_model: str, black_model: str) -> None:
    """Create and initialize a new game directory with required files"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    # Initialize files with empty but valid JSON structures
    files = {
        "white_messages.jsonl": [],
        "black_messages.jsonl": [],
        "game.jsonl": [GameMetadata(
            game_id=directory.name.split('_')[-1],
            white_model=white_model,
            black_model=black_model,
            evaluation_list=[],
            winner=None,
            moves=[]
        ).to_dict()]
    }
    
    for filename, initial_data in files.items():
        file_path = directory / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                if isinstance(initial_data, list):
                    for item in initial_data:
                        f.write(json.dumps(item) + '\n')

async def run_game(directory: str, white_model: str, black_model: str) -> None:
    """Run a chess game between two LLM models"""
    try:
        # Initialize game state and components
        game_state = GameState(
            directory=Path(directory),
            game_id=Path(directory).name.split('_')[-1],
            white_model=white_model,
            black_model=black_model
        )
        
        # Initialize players
        await game_state.white_player.initialize()
        await game_state.black_player.initialize()
        
        # Create game runner
        runner = GameRunner(game_state)
        
        # Start the game
        logger.info(f"Starting game in {directory}")
        runner.start()
        
        # Wait for game to complete
        while runner.is_running:
            await asyncio.sleep(1)
            
        logger.info(f"Game completed in {directory}")
        
    except Exception as e:
        logger.error(f"Error running game: {e}", exc_info=True)
        raise

def load_existing_game_state(directory: Path) -> Optional[dict]:
    """Load existing game state if available"""
    state_path = directory / 'game_state.json'
    if state_path.exists():
        try:
            with open(state_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading existing game state: {e}")
    return None

async def launch_game(white_model: str, black_model: str, load_game_id: Optional[Union[str, List[int], int]] = None) -> None:
    """Launch game with proper state preservation"""
    try:
        directories = []
        if load_game_id == "all":
            pattern = get_folder_name(white_model, black_model, "all") + "_*"
            directories = [Path(d) for d in glob.glob(pattern)]
        elif isinstance(load_game_id, (list, int)):
            game_ids = [load_game_id] if isinstance(load_game_id, int) else load_game_id
            directories = [Path(d) for d in get_folder_name(white_model, black_model, game_ids)]
        else:
            directory = Path(get_folder_name(white_model, black_model, load_game_id))
            directories = [directory]
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True)
                initialize_game_directory(directory, white_model, black_model)
            
            # Load existing state
            existing_state = load_existing_game_state(directory)
            if existing_state:
                logger.info(f"Loaded existing game state in {directory}")
            
            # Start game runner with existing state
            game_state = GameState(
                directory=directory,
                game_id=directory.name.split('_')[-1],
                white_model=white_model,
                black_model=black_model
            )
            
            if existing_state:
                game_state.load_state(existing_state)
            
            runner = GameRunner(game_state)
            runner.start()
            
            logger.info(f"Game running in {directory}")
            
    except Exception as e:
        logger.error(f"Error launching game: {e}", exc_info=True)
        raise

async def main_async():
    """Async main entry point"""
    parser = argparse.ArgumentParser(description="LLM Chess Battle System")
    parser.add_argument('--white', required=True, help="White player model")
    parser.add_argument('--black', required=True, help="Black player model")
    parser.add_argument('--load_game_id', help="Game ID to load (empty=new, 'all'=unfinished, or comma-separated IDs)")
    
    args = parser.parse_args()
    
    # Process load_game_id argument
    load_game_id = None
    if args.load_game_id:
        if args.load_game_id == "all":
            load_game_id = "all"
        else:
            try:
                load_game_id = [int(id.strip()) for id in args.load_game_id.split(',')]
                if len(load_game_id) == 1:
                    load_game_id = load_game_id[0]
            except ValueError:
                parser.error("load_game_id must be 'all' or comma-separated integers")
    
    # Launch game(s)
    await launch_game(args.white, args.black, load_game_id)

def main():
    """Main entry point"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()