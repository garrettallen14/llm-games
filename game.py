from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import chess
import chess.pgn
import json
import logging
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import fcntl
import time
import io
import os
from contextlib import contextmanager
from enum import Enum
from llm import LLMPlayer, MessageHistory
from analysis import ChessAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    
    def __str__(self):
        return self.value
    
    def to_json(self):
        return self.value
    
    @classmethod
    def from_json(cls, value):
        return cls(value)
    
class GameJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for game-related objects"""
    def default(self, obj):
        if isinstance(obj, GameStatus):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, GameMetadata):
            return obj.to_dict()
        return super().default(obj)

@dataclass
class GameMetadata:
    game_id: str
    white_model: str
    black_model: str
    status: GameStatus
    start_time: datetime
    last_update: datetime
    current_fen: Optional[str] = None
    winner: Optional[int] = None  # None=ongoing, 0=white, 1=black, 2=draw
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GameMetadata':
        """Create GameMetadata from dictionary"""
        # Convert string timestamps to datetime objects
        for field in ['start_time', 'last_update']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field].replace('Z', '+00:00'))
        
        # Convert status string to enum
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = GameStatus(data['status'])
            
        return cls(**data)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = {
            'game_id': self.game_id,
            'white_model': self.white_model,
            'black_model': self.black_model,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if isinstance(self.start_time, datetime) else self.start_time,
            'last_update': self.last_update.isoformat() if isinstance(self.last_update, datetime) else self.last_update,
            'current_fen': self.current_fen,
            'winner': self.winner
        }
        return data

class ChessGame:
    """Core chess game mechanics"""
    def __init__(self):
        self.board = chess.Board()
        self.game = chess.pgn.Game()
        self.node = self.game
        self.moves_history: List[str] = []
        
    def make_move(self, move_san: str) -> Tuple[bool, Optional[str]]:
        """Make a move in SAN format"""
        try:
            move = self.board.parse_san(move_san)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.node = self.node.add_variation(move)
                self.moves_history.append(move_san)
                return True, None
            return False, "Illegal move"
        except ValueError as e:
            return False, str(e)
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete game state"""
        return {
            'fen': self.board.fen(),
            'pgn': self.get_pgn(),
            'moves': self.moves_history.copy(),
            'is_game_over': self.board.is_game_over(),
            'current_player': 'white' if self.board.turn else 'black',
            'legal_moves': self._get_legal_moves()
        }
    
    def load_state(self, moves: List[str]) -> bool:
        """Load game state from move list"""
        try:
            self.__init__()  # Reset state
            for move in moves:
                success, error = self.make_move(move)
                if not success:
                    raise ValueError(f"Invalid move sequence: {error}")
            return True
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False
    
    def _get_legal_moves(self) -> Dict[str, List[str]]:
        """Get all legal moves categorized by piece"""
        legal_moves = {}
        for move in self.board.legal_moves:
            san = self.board.san(move)
            piece = self.board.piece_at(move.from_square)
            if piece:
                piece_name = piece.symbol().upper()
                if piece_name not in legal_moves:
                    legal_moves[piece_name] = []
                legal_moves[piece_name].append(san)
        return legal_moves
    
    def get_pgn(self) -> str:
        """Get game PGN"""
        output = io.StringIO()
        exporter = chess.pgn.FileExporter(output)
        self.game.accept(exporter)
        return output.getvalue().strip()

class GameState:
    """Complete game state manager with proper persistence"""
    def __init__(self, directory: Path, game_id: str, white_model: str, black_model: str):
        self.directory = Path(directory)
        self.game_id = game_id
        self.chess_game = ChessGame()
        self.white_player = LLMPlayer(white_model, "white", str(directory))
        self.black_player = LLMPlayer(black_model, "black", str(directory))
        self.analyzer = ChessAnalyzer("stockfish")
        
        # Load or initialize metadata
        self.metadata = self._initialize_metadata(white_model, black_model)
        
        # Load existing state if available
        self.load_complete_state()

    async def load_complete_state(self) -> bool:
        """Load complete game state from all available files"""
        try:
            # Load message histories first
            await self.white_player.initialize()
            await self.black_player.initialize()
            
            # Reconstruct moves from message histories
            moves = self._reconstruct_moves_from_messages()
            
            # Load saved game state
            game_state_file = self.directory / 'game_state.json'
            if game_state_file.exists():
                with open(game_state_file, 'r') as f:
                    saved_state = json.load(f)
                    
                # Validate and merge moves
                if saved_state.get('moves'):
                    # Compare saved moves with reconstructed moves
                    if len(saved_state['moves']) >= len(moves):
                        moves = saved_state['moves']
                    
                # Load analysis if available
                if saved_state.get('analysis'):
                    self.analyzer.load_evaluation_history(saved_state['analysis'])
            
            # Apply moves to chess game
            success = self.chess_game.load_state(moves)
            if not success:
                logger.error("Failed to load moves into chess game")
                return False
            
            # Update metadata
            self.metadata.current_fen = self.chess_game.board.fen()
            if self.chess_game.board.is_game_over():
                self.metadata.status = GameStatus.COMPLETED
                self.metadata.winner = self._determine_winner()
            
            self._save_metadata(self.metadata)
            return True
            
        except Exception as e:
            logger.error(f"Error loading complete state: {e}")
            return False
        
    def _reconstruct_moves_from_messages(self) -> List[str]:
        """Reconstruct game moves from message histories"""
        moves = []
        white_moves = self._extract_moves_from_history(self.white_player.message_history)
        black_moves = self._extract_moves_from_history(self.black_player.message_history)
        
        # Interleave moves in correct order
        max_moves = max(len(white_moves), len(black_moves))
        for i in range(max_moves):
            if i < len(white_moves):
                moves.append(white_moves[i])
            if i < len(black_moves):
                moves.append(black_moves[i])
        
        return moves
    
    def _extract_moves_from_history(self, history: MessageHistory) -> List[str]:
        """Extract valid moves from message history"""
        moves = []
        for msg in history.messages:
            if msg.metadata and msg.metadata.get('move') and msg.metadata.get('success', False):
                moves.append(msg.metadata['move'])
        return moves
        
    def _initialize_metadata(self, white_model: str, black_model: str) -> GameMetadata:
        """Initialize or load game metadata"""
        metadata_file = self.directory / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return GameMetadata.from_dict(data)
        else:
            metadata = GameMetadata(
                game_id=self.game_id,
                white_model=white_model,
                black_model=black_model,
                status=GameStatus.INITIALIZING,
                start_time=datetime.now(),
                last_update=datetime.now()
            )
            self._save_metadata(metadata)
            return metadata
    
    @contextmanager
    def _file_lock(self, filename: str):
        """File locking context manager with timeout"""
        lock_file = self.directory / f"{filename}.lock"
        lock_fd = None
        try:
            lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)
            
            # Try to acquire lock with timeout
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError as e:
                    if time.time() - start_time > 5:  # 5 second timeout
                        raise TimeoutError("Failed to acquire file lock")
                    time.sleep(0.1)
                    
            yield
        finally:
            if lock_fd is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                try:
                    lock_file.unlink()
                except OSError:
                    pass

    def _save_metadata(self, metadata: GameMetadata):
        """Save metadata with file locking"""
        with self._file_lock('metadata'):
            with open(self.directory / 'metadata.json', 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
    
    def save_state(self):
        """Save complete game state with atomic operations"""
        # Update metadata
        self.metadata.last_update = datetime.now()
        self.metadata.current_fen = self.chess_game.board.fen()
        if self.chess_game.board.is_game_over():
            self.metadata.status = GameStatus.COMPLETED
            self.metadata.winner = self._determine_winner()
        self._save_metadata(self.metadata)
        
        # Save game state
        with self._file_lock('game_state'):
            state = {
                'moves': self.chess_game.moves_history,
                'analysis': self.analyzer.get_evaluation_history(),
                'last_update': datetime.now().isoformat(),
                'current_fen': self.chess_game.board.fen()
            }
            
            # Atomic write using temporary file
            temp_file = self.directory / 'game_state.json.tmp'
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Atomic rename
            temp_file.replace(self.directory / 'game_state.json')
    
    def load_state(self, state_data: dict) -> bool:
        """Load game state from dictionary"""
        try:
            # Load moves
            if 'moves' in state_data:
                success = self.chess_game.load_state(state_data['moves'])
                if not success:
                    return False
            
            # Load analysis
            if 'analysis' in state_data:
                self.analyzer.load_evaluation_history(state_data['analysis'])
            
            # Update metadata
            self.metadata.current_fen = self.chess_game.board.fen()
            self.metadata.last_update = datetime.now()
            
            return True
        except Exception as e:
            logger.error(f"Error loading game state: {e}")
            return False
            
    def _determine_winner(self) -> Optional[int]:
        """Determine game winner"""
        board = self.chess_game.board
        if not board.is_game_over():
            return None
        if board.is_checkmate():
            return 0 if not board.turn else 1
        return 2  # Draw

class GameRunner:
    """Game execution manager"""
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.is_running = False
        self.current_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.last_failed_move = None  # Track the last failed move
    
    async def _make_move(self) -> bool:
        """Execute a single move with forfeit handling"""
        try:
            current_player = "white" if self.game_state.chess_game.board.turn else "black"
            player = self.game_state.white_player if current_player == "white" else self.game_state.black_player
            
            # Get move, passing the last failed move if there was one
            response = await player.get_move(
                self.game_state.chess_game.board,
                self.game_state.chess_game.get_pgn(),
                current_player,
                failed_move=self.last_failed_move  # Pass the tracked failed move
            )
            
            if not response.success:
                if "Forfeit:" in response.error:
                    logger.info(f"Game ended by forfeit: {response.error}")
                    self.game_state.metadata.status = GameStatus.COMPLETED
                    self.game_state.metadata.winner = 1 if current_player == "white" else 0
                    self.game_state.save_state()
                    self.is_running = False
                    return False
                
                # If the move failed, store it for the next attempt
                if response.failed_move:
                    self.last_failed_move = response.move
                return False
            
            # Make move
            success, error = self.game_state.chess_game.make_move(response.move)
            if not success:
                logger.error(f"Invalid move: {error}")
                self.last_failed_move = response.move  # Store failed move
                return False
            
            # Success! Clear failed move and continue
            self.last_failed_move = None
            
            # Analyze position
            await self.game_state.analyzer.analyze_position_async(self.game_state.chess_game.board)
            
            # Save state
            self.game_state.save_state()
            return True
            
        except Exception as e:
            logger.error(f"Error making move: {e}")
            return False
    
    async def _game_loop(self):
        """Main game loop"""
        while self.is_running and not self.game_state.chess_game.board.is_game_over():
            if not await self._make_move():
                await asyncio.sleep(1)  # Wait before retry
            else:
                await asyncio.sleep(0.1)  # Small delay between moves
        
        self.is_running = False
        self.game_state.save_state()
    
    def start(self):
        """Start or resume game"""
        if not self.is_running:
            self.is_running = True
            self.current_thread = threading.Thread(
                target=lambda: asyncio.run(self._game_loop())
            )
            self.current_thread.start()
    
    def stop(self):
        """Stop game execution"""
        self.is_running = False
        if self.current_thread and self.current_thread.is_alive():
            self.current_thread.join(timeout=1.0)
            self.game_state.save_state()