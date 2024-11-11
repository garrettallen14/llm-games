from flask import Flask, render_template, jsonify, request
from pathlib import Path
import logging
import logging.handlers
from datetime import datetime
import time
import json
import asyncio
import fcntl
import sys
import os
from contextlib import contextmanager
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import subprocess
from typing import Dict, List

from game import GameStatus, GameMetadata, GameJSONEncoder
from model_config import ModelManager
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class SimpleGameMetadata:
    """Simplified game metadata for web display"""
    game_id: str
    white_model: str
    black_model: str
    status: str
    current_fen: str
    moves: List[str]
    evaluations: List[float]
    start_time: str
    last_update: str
    move_count: int
    winner: Optional[str] = None
    
    def to_dict(self):
        return {
            'game_id': self.game_id,
            'white_model': self.white_model,
            'black_model': self.black_model,
            'status': self.status,
            'current_fen': self.current_fen,
            'moves': self.moves,
            'evaluations': self.evaluations,
            'start_time': self.start_time,
            'last_update': self.last_update,
            'move_count': self.move_count,
            'winner': self.winner
        }

    @classmethod
    def from_game_state(cls, metadata: dict, state: dict):
        """Create from raw game data"""
        return cls(
            game_id=metadata['game_id'],
            white_model=metadata['white_model'],
            black_model=metadata['black_model'],
            status=metadata['status'],
            current_fen=state.get('current_fen', ''),
            moves=state.get('moves', []),
            evaluations=[a.get('eval_score', 0) for a in state.get('analysis', [])],
            start_time=metadata['start_time'],
            last_update=metadata['last_update'],
            move_count=len(state.get('moves', [])),
            winner=metadata.get('winner')
        )

# Configure logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            'chess_app.log',
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

class FileLock:
    """Thread and process-safe file locking mechanism"""
    def __init__(self, path: Path):
        self.lock_path = path.parent / f".{path.name}.lock"
        self.lock_fd = None
        
    def __enter__(self):
        try:
            self.lock_fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
            fcntl.flock(self.lock_fd, fcntl.LOCK_EX)
            return self
        except Exception as e:
            if self.lock_fd:
                try:
                    os.close(self.lock_fd)
                except:
                    pass
            raise e
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.lock_fd is not None:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                try:
                    os.unlink(str(self.lock_path))
                except OSError:
                    pass
        except Exception as e:
            logger.error(f"Error in FileLock exit: {e}")

class GameDirectoryMonitor(FileSystemEventHandler):
    """Monitor game directories for changes"""
    def __init__(self, app_state: 'AppState'):
        self.app_state = app_state
        self.last_update = {}
        self.debounce_interval = 0.1  # seconds
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        now = time.time()
        path = event.src_path
        
        # Debounce updates
        if path in self.last_update:
            if now - self.last_update[path] < self.debounce_interval:
                return
        
        self.last_update[path] = now
        
        if path.endswith(('metadata.json', 'game_state.json')):
            try:
                game_dir = str(Path(path).parent)
                self.app_state.reload_game(game_dir)
            except Exception as e:
                logger.error(f"Error handling file modification: {e}")

class AppState:
    """Application state manager"""
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.games: Dict[str, GameMetadata] = {}
        self.game_processes: Dict[str, subprocess.Popen] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Set up file system monitoring
        self.observer = Observer()
        self.observer.schedule(
            GameDirectoryMonitor(self),
            str(data_dir),
            recursive=True
        )
        self.observer.start()
        
        # Initial load and start all games
        self.load_games()
        self.start_all_games()
    
    def load_games(self):
        """Load all games from data directory"""
        try:
            for game_dir in self.data_dir.glob("*"):
                if game_dir.is_dir():
                    self.reload_game(str(game_dir))
        except Exception as e:
            logger.error(f"Error loading games: {e}")
    
    def reload_game(self, game_dir: str):
        """Reload a specific game's metadata"""
        try:
            metadata_file = Path(game_dir) / 'metadata.json'
            if metadata_file.exists():
                with FileLock(metadata_file):
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                        metadata = GameMetadata.from_dict(data)
                        self.games[str(metadata_file.parent)] = metadata
        except Exception as e:
            logger.error(f"Error reloading game {game_dir}: {e}")
    
    def start_all_games(self):
        """Start all loaded games"""
        for game_dir, metadata in self.games.items():
            if metadata.status != GameStatus.COMPLETED:
                try:
                    # Launch game process
                    process = subprocess.Popen([
                        sys.executable,
                        'main.py',
                        '--white', metadata.white_model,
                        '--black', metadata.black_model,
                        '--load_game_id', metadata.game_id
                    ])
                    self.game_processes[game_dir] = process
                except Exception as e:
                    logger.error(f"Error starting game {game_dir}: {e}")

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Stop file system observer
        self.observer.stop()
        self.observer.join()
        
        # Clean up processes
        for process in self.game_processes.values():
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        # Shut down executor
        self.executor.shutdown(wait=True)

# Initialize Flask app
app = Flask(__name__,
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')
app.json_encoder = GameJSONEncoder

# Initialize data directory
DATA_DIR = Path("game_data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize application state
app_state = AppState(DATA_DIR)

# Add to app initialization
model_manager = ModelManager()

def get_next_game_id() -> int:
    """Get next available sequential game ID"""
    try:
        existing_ids = []
        for game_dir in DATA_DIR.glob("*"):
            try:
                game_id = int(game_dir.name.split('_')[-1])
                existing_ids.append(game_id)
            except (ValueError, IndexError):
                continue
        return max(existing_ids, default=-1) + 1
    except Exception as e:
        logger.error(f"Error getting next game ID: {e}")
        raise

def sanitize_model_name(name: str) -> str:
    """Clean model names for file system use"""
    return name.strip('/').replace('/', '_').replace('\\', '_').replace(' ', '_')

@app.route('/')
def dashboard():
    """Render main dashboard"""
    try:
        games = {
            str(Path(game_dir).relative_to(DATA_DIR)): game.to_dict()
            for game_dir, game in app_state.games.items()
        }
        return render_template('dashboard.html', games=games)
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return "Error loading dashboard", 500

@app.route('/game/<path:game_dir>')
def game_view(game_dir):
    """Render individual game view"""
    try:
        # Normalize the path to handle both direct paths and URL-style paths
        full_path = (DATA_DIR / game_dir).resolve()
        if not str(full_path).startswith(str(DATA_DIR.resolve())):
            raise ValueError("Invalid game directory path")
            
        # Check for metadata.json to validate game exists
        metadata_path = full_path / 'metadata.json'
        if not metadata_path.exists():
            logger.warning(f"Game metadata not found at {metadata_path}")
            return "Game not found", 404
        
        # Load game metadata
        try:
            with FileLock(metadata_path):
                with open(metadata_path, 'r') as f:
                    game_data = json.load(f)
                    metadata = GameMetadata.from_dict(game_data)
        except Exception as e:
            logger.error(f"Error loading game metadata: {e}")
            return "Error loading game", 500
        
        # Initialize or validate state file
        state_path = full_path / 'game_state.json'
        if not state_path.exists():
            state_data = {
                'moves': [],
                'analysis': [],
                'fen': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
            }
            with FileLock(state_path):
                with open(state_path, 'w') as f:
                    json.dump(state_data, f, indent=2)
        
        try:
            return render_template('game.html',
                                game_dir=game_dir,
                                game=metadata.to_dict())
        except Exception as e:
            logger.error(f"Template rendering error: {str(e)}")
            raise
                             
    except ValueError as e:
        logger.warning(f"Invalid game access attempt: {e}")
        return "Invalid game path", 400
    except Exception as e:
        logger.error(f"Error rendering game view: {str(e)}")
        return f"Error loading game: {str(e)}", 500

@app.route('/api/games')
def get_games():
    """Get current games status"""
    try:
        return jsonify({
            str(Path(game_dir).relative_to(DATA_DIR)): game.to_dict()
            for game_dir, game in app_state.games.items()
        })
    except Exception as e:
        logger.error(f"Error getting games: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/game/<path:game_dir>/status')
def get_game_status(game_dir):
    """Get simplified game status for web display"""
    try:
        full_path = (DATA_DIR / game_dir).resolve()
        if not str(full_path).startswith(str(DATA_DIR.resolve())):
            raise ValueError("Invalid game directory path")
        
        state_path = full_path / 'game_state.json'
        metadata_path = full_path / 'metadata.json'
        
        if not all(p.exists() for p in [state_path, metadata_path]):
            return jsonify({'error': 'Game not found'}), 404
        
        # Load state and metadata with locks
        with FileLock(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
        
        with FileLock(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Create simplified metadata
        simple_metadata = SimpleGameMetadata.from_game_state(metadata, state)
        
        return jsonify({
            'metadata': simple_metadata.to_dict(),
            'state': {
                'current_fen': state.get('current_fen', ''),
                'moves': state.get('moves', []),
                'analysis': state.get('analysis', [])
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting game status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def get_models():
    """Get available models"""
    try:
        models = model_manager.get_all_models()
        return jsonify([{
            'display_name': m.display_name,
            'model_name': m.model_name,
            'max_tokens': m.max_tokens
        } for m in models])
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/create_game', methods=['POST'])
def create_game():
    """Create new game with model validation"""
    try:
        data = request.json
        white_model = data['white_model']
        black_model = data['black_model']
        
        # Validate models
        if not (model_manager.validate_model(white_model) and 
                model_manager.validate_model(black_model)):
            return jsonify({
                'success': False,
                'error': 'Invalid model selection'
            }), 400
        
        # Get next game ID and create directory
        game_id = get_next_game_id()
        game_dir = f"{sanitize_model_name(white_model)}_{sanitize_model_name(black_model)}_{game_id}"
        full_path = DATA_DIR / game_dir
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        metadata = GameMetadata(
            game_id=str(game_id),
            white_model=white_model,
            black_model=black_model,
            status=GameStatus.INITIALIZING,
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        # Initialize files
        init_files = {
            'metadata.json': metadata.to_dict(),
            'game_state.json': {
                'moves': [],
                'analysis': []
            }
        }
        
        for filename, content in init_files.items():
            file_path = full_path / filename
            with FileLock(file_path):
                with open(file_path, 'w') as f:
                    json.dump(content, f, indent=2)
        
        # Create empty message files
        for color in ['white', 'black']:
            (full_path / f'{color}_messages.jsonl').touch()
        
        # Start the game
        process = subprocess.Popen([
            sys.executable,
            'main.py',
            '--white', white_model,
            '--black', black_model,
            '--load_game_id', str(game_id)
        ])
        
        # Update app state
        app_state.reload_game(str(full_path))
        app_state.game_processes[str(full_path)] = process
        
        return jsonify({
            'success': True,
            'game_dir': game_dir,  # Return just the relative path
            'game_id': game_id
        })
        
    except Exception as e:
        logger.error(f"Error creating game: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def init_app():
    """Initialize the application"""
    try:
        # Ensure data directory exists and is writable
        DATA_DIR.mkdir(exist_ok=True)
        test_file = DATA_DIR / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except:
            raise RuntimeError("Data directory is not writable")
        
        # Clean up any stale lock files
        for lock_file in DATA_DIR.glob("*/.*.lock"):
            try:
                lock_file.unlink()
            except:
                pass
        
        logger.info("Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        init_app()
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)
    finally:
        app_state.cleanup()