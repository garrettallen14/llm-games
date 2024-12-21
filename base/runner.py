# base/runner.py

import argparse
import importlib.util
import os
import sys
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

from .llm_player import BaseLLMPlayer, PlayerConfig

# Load environment variables
load_dotenv()

@dataclass
class RunnerConfig:
    """Configuration for game runner"""
    models: List[str]
    game_id: str
    max_turns: int = 100
    games_path: Path = Path("games")
    logs_path: Path = Path("logs")
    seed: Optional[int] = None

class GameRunner:
    """Handles loading and running games with LLM players"""
    
    def __init__(self, config: RunnerConfig):
        """Initialize the game runner"""
        self.config = config
        self.validate_environment()
        self.run_dir = self.setup_run_directory()
        print(f"Initialized game run in {self.run_dir}")
        
    def validate_environment(self):
        """Ensure required environment variables are set"""
        if not os.getenv('OPENROUTER_API_KEY'):
            raise ValueError("OPENROUTER_API_KEY must be set in .env file")
            
    def get_next_run_id(self) -> int:
        """Get the next available run ID for the current game"""
        self.config.logs_path.mkdir(exist_ok=True)
        
        existing_runs = [
            d for d in self.config.logs_path.glob(f"{self.config.game_id}_run*")
            if d.is_dir() and d.name[len(self.config.game_id)+4:].isdigit()
        ]
        
        if not existing_runs:
            return 0
            
        max_run = max(
            int(d.name[len(self.config.game_id)+4:])
            for d in existing_runs
        )
        
        return max_run + 1
        
    def setup_run_directory(self) -> Path:
        """Create and return the path for the new run directory"""
        run_id = self.get_next_run_id()
        run_dir = self.config.logs_path / f"{self.config.game_id}_run{run_id}"
        run_dir.mkdir(exist_ok=True)
        return run_dir
        
    def create_players(self) -> List[BaseLLMPlayer]:
        """Create LLM players for the game"""
        return [
            BaseLLMPlayer(
                PlayerConfig(
                    model_name=model,
                    player_id=idx + 1,
                    run_dir=self.run_dir,
                )
            )
            for idx, model in enumerate(self.config.models)
        ]
        
    def load_game_module(self):
        """
        Dynamically import game module based on game_id
        Returns instance of the game class
        """
        try:
            # Construct the full path to the game module
            game_path = Path('games') / self.config.game_id / 'game.py'
            if not game_path.exists():
                raise ImportError(f"Game file not found: {game_path}")

            # Import the module using spec
            spec = importlib.util.spec_from_file_location(
                f"{self.config.game_id}_game",
                game_path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to create module spec for {game_path}")
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            
            # Get game class (assumes standard naming convention)
            game_class_name = "".join(word.capitalize() for word in self.config.game_id.split("_")) + "Game"
            game_class = getattr(module, game_class_name)
            
            return game_class
            
        except Exception as e:
            raise ImportError(f"Failed to import game module: {str(e)}")
            
    def run(self) -> Dict:
        """
        Run the game with specified models
        Returns the game result
        """
        print(f"Starting game with models: {', '.join(self.config.models)}")
        
        try:
            # Load game class and create instance
            game_class = self.load_game_module()
            
            # Pass seed if provided
            game_kwargs = {
                'run_dir': self.run_dir, 
                'max_turns': self.config.max_turns
            }
            if self.config.seed is not None:
                game_kwargs['seed'] = self.config.seed
            
            game = game_class(**game_kwargs)
            
            # Create players
            players = self.create_players()
            
            # Initialize players with game's system prompt
            for player in players:
                player.initialize_with_prompt(game.get_system_prompt())
            
            # Run the game
            result = game.run(players)
            print(f"\nGame complete! Result: {result}")
            return result
            
        except Exception as e:
            print(f"Error during game: {str(e)}")
            raise

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run an LLM game between multiple models')
    parser.add_argument(
        '--models',
        type=str,
        required=True,
        help='Comma-separated list of model names'
    )
    parser.add_argument(
        '--game_id',
        type=str,
        required=True,
        help='ID of the game to play (must match a folder in games/)'
    )
    parser.add_argument(
        '--max_turns',
        type=int,
        default=100,
        help='Maximum number of turns before game ends'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Seed for random number generation'
    )
    
    args = parser.parse_args()
    return RunnerConfig(
        models=[m.strip() for m in args.models.split(',')],
        game_id=args.game_id,
        max_turns=args.max_turns,
        seed=args.seed
    )

if __name__ == '__main__':
    config = parse_args()
    runner = GameRunner(config)
    result = runner.run()