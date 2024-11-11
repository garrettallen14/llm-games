import chess
import chess.engine
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
import asyncio
import json
from pathlib import Path
from datetime import datetime
import aiofiles
import functools
import concurrent.futures
from contextlib import asynccontextmanager
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PositionEvaluation:
    move_number: int
    is_white_move: bool
    eval_score: float
    best_moves: List[str]
    timestamp: datetime
    depth: int
    nodes: int
    time_ms: int
    
    def to_dict(self) -> Dict:
        return {
            'move_number': self.move_number,
            'is_white_move': self.is_white_move,
            'eval_score': self.eval_score,
            'best_moves': self.best_moves,
            'timestamp': self.timestamp.isoformat(),
            'depth': self.depth,
            'nodes': self.nodes,
            'time_ms': self.time_ms
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PositionEvaluation':
        return cls(
            **{k: v if k != 'timestamp' else datetime.fromisoformat(v) 
               for k, v in data.items()}
        )

@dataclass
class MoveAnalysis:
    eval_before: float
    eval_after: float
    classification: str
    best_continuation: List[str]
    time_spent_ms: int
    depth_reached: int
    
    def to_dict(self) -> Dict:
        return asdict(self)

class AnalysisCache:
    """Cache for position evaluations"""
    def __init__(self, cache_file: Optional[Path] = None):
        self.cache: Dict[str, PositionEvaluation] = {}
        self.cache_file = cache_file
        self.modified = False
        if cache_file:
            asyncio.run(self.load_cache())
    
    async def load_cache(self):
        """Load cache from file"""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            async with aiofiles.open(self.cache_file, 'r') as f:
                data = json.loads(await f.read())
                self.cache = {
                    k: PositionEvaluation.from_dict(v) 
                    for k, v in data.items()
                }
        except Exception as e:
            logger.error(f"Error loading analysis cache: {e}")
    
    async def save_cache(self):
        """Save cache to file if modified"""
        if not self.modified or not self.cache_file:
            return
        
        try:
            async with aiofiles.open(self.cache_file, 'w') as f:
                cache_data = {
                    k: v.to_dict() 
                    for k, v in self.cache.items()
                }
                await f.write(json.dumps(cache_data, indent=2))
            self.modified = False
        except Exception as e:
            logger.error(f"Error saving analysis cache: {e}")

class ChessAnalyzer:
    def __init__(self, engine_path: str, cache_dir: Optional[Path] = None):
        """Initialize chess analyzer with engine path and optional cache directory"""
        self.engine_path = engine_path
        self.cache_dir = cache_dir
        self.cache = AnalysisCache(
            cache_dir / 'analysis_cache.json' if cache_dir else None
        )
        self.engine_semaphore = asyncio.Semaphore(1)  # Limit concurrent engine usage
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Analysis parameters
        self.default_depth = 20
        self.min_time_ms = 100
        self.max_time_ms = 5000
        self.multipv = 3
        
        # Classification thresholds
        self.thresholds = {
            'brilliant': 200,  # +2.0
            'good': 50,       # +0.5
            'inaccuracy': -50,  # -0.5
            'mistake': -100,    # -1.0
            'blunder': -200     # -2.0
        }
        
        # Evaluation history
        self.evaluation_history: List[Dict] = []
        
        logger.info(f"Initialized ChessAnalyzer with engine: {engine_path}")

    @asynccontextmanager
    async def _get_engine(self):
        """Async context manager for engine access"""
        async with self.engine_semaphore:
            transport, engine = await chess.engine.popen_uci(self.engine_path)
            try:
                yield engine
            finally:
                await engine.quit()
    
    def _normalize_score(self, score: chess.engine.Score, 
                        perspective: chess.Color) -> float:
        """Convert engine score to normalized float value"""
        try:
            if score.is_mate():
                # Convert mate scores to large values
                mate_score = score.relative.mate()
                value = 100.0 if mate_score > 0 else -100.0
                # Adjust for closer mates
                if mate_score > 0:
                    value -= 1.0 / mate_score
                else:
                    value += 1.0 / abs(mate_score)
            else:
                # Convert centipawns to pawns
                value = float(score.relative.cp or 0) / 100.0
            
            # Adjust for perspective
            return value if perspective else -value
            
        except Exception as e:
            logger.error(f"Error normalizing score: {e}")
            return 0.0
    
    def get_evaluation_history(self) -> List[Dict]:
        """Get complete evaluation history"""
        return [eval_data for eval_data in self.evaluation_history]
    
    async def analyze_position_async(self, board: chess.Board) -> Dict:
        """Analyze position and update history"""
        evaluation = await self.analyze_position(board)
        
        # Create evaluation record
        eval_data = {
            'fen': board.fen(),
            'move_number': board.fullmove_number,
            'is_white_move': board.turn,
            'timestamp': evaluation.timestamp.isoformat(),
            'eval_score': evaluation.eval_score,
            'best_moves': evaluation.best_moves,
            'depth': evaluation.depth,
            'nodes': evaluation.nodes,
            'time_ms': evaluation.time_ms,
        }
        
        # Add to history
        self.evaluation_history.append(eval_data)
        
        return eval_data
    
    async def analyze_position(self, board: chess.Board, 
                             depth: Optional[int] = None) -> PositionEvaluation:
        """Analyze current position and return evaluation"""
        fen = board.fen()
        
        # Check cache first
        cached = self.cache.cache.get(fen)
        if cached and (depth is None or cached.depth >= depth):
            return cached
        
        try:
            start_time = datetime.now()
            
            async with self._get_engine() as engine:
                # Set analysis parameters
                limit = chess.engine.Limit(
                    depth=depth or self.default_depth,
                    time=self.max_time_ms / 1000,
                    nodes=1000000
                )
                
                # Run analysis
                info = await engine.analyse(
                    board,
                    limit,
                    multipv=self.multipv
                )
                
                # Process results
                main_line = info[0]
                eval_score = self._normalize_score(
                    main_line['score'],
                    board.turn
                )
                
                # Extract best moves
                best_moves = []
                temp_board = board.copy()
                for pv in info:
                    if 'pv' in pv and pv['pv']:
                        try:
                            move = pv['pv'][0]
                            best_moves.append(temp_board.san(move))
                        except Exception:
                            continue
                
                # Create evaluation
                evaluation = PositionEvaluation(
                    move_number=board.fullmove_number,
                    is_white_move=board.turn,
                    eval_score=eval_score,
                    best_moves=best_moves[:self.multipv],
                    timestamp=datetime.now(),
                    depth=main_line.get('depth', 0),
                    nodes=main_line.get('nodes', 0),
                    time_ms=(datetime.now() - start_time).microseconds // 1000
                )
                
                # Cache result
                self.cache.cache[fen] = evaluation
                self.cache.modified = True
                await self.cache.save_cache()
                
                return evaluation
                
        except Exception as e:
            logger.error(f"Error analyzing position: {e}")
            # Return last cached evaluation if available
            return cached if cached else PositionEvaluation(
                move_number=board.fullmove_number,
                is_white_move=board.turn,
                eval_score=0.0,
                best_moves=[],
                timestamp=datetime.now(),
                depth=0,
                nodes=0,
                time_ms=0
            )
    
    def classify_move(self, eval_before: float, 
                     eval_after: float) -> Tuple[str, float]:
        """Classify move based on evaluation change"""
        eval_change = eval_after - eval_before
        
        if eval_change >= self.thresholds['brilliant']:
            return 'brilliant', eval_change
        elif eval_change >= self.thresholds['good']:
            return 'good', eval_change
        elif eval_change <= self.thresholds['blunder']:
            return 'blunder', eval_change
        elif eval_change <= self.thresholds['mistake']:
            return 'mistake', eval_change
        elif eval_change <= self.thresholds['inaccuracy']:
            return 'inaccuracy', eval_change
        return 'normal', eval_change
    
    async def analyze_game(self, game: chess.Board, 
                          moves: List[str]) -> List[MoveAnalysis]:
        """Analyze entire game"""
        analyses = []
        board = chess.Board()
        
        prev_eval = None
        for move in moves:
            # Get evaluation before move
            if prev_eval is None:
                eval_before = (await self.analyze_position(board)).eval_score
            else:
                eval_before = prev_eval
            
            # Make move
            board.push_san(move)
            
            # Get evaluation after move
            eval_after = (await self.analyze_position(board)).eval_score
            prev_eval = eval_after
            
            # Classify move
            classification, eval_change = self.classify_move(
                eval_before,
                eval_after
            )
            
            # Get best continuation
            evaluation = await self.analyze_position(board)
            
            analyses.append(MoveAnalysis(
                eval_before=eval_before,
                eval_after=eval_after,
                classification=classification,
                best_continuation=evaluation.best_moves,
                time_spent_ms=evaluation.time_ms,
                depth_reached=evaluation.depth
            ))
        
        return analyses
    
    async def get_position_summary(self, board: chess.Board) -> Dict:
        """Get comprehensive position analysis"""
        evaluation = await self.analyze_position(board)
        
        return {
            'evaluation': evaluation.eval_score,
            'best_moves': evaluation.best_moves,
            'depth': evaluation.depth,
            'nodes': evaluation.nodes,
            'time_ms': evaluation.time_ms,
            'timestamp': evaluation.timestamp.isoformat()
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)