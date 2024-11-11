import os
import aiohttp
import json
import chess
import asyncio
import aiofiles
import traceback
import logging
import base64
import io
from PIL import Image
import cairosvg
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from prompt import SYSTEM_PROMPT, format_move_request, format_error_message
from model_config import ModelManager, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MoveResponse:
    """Represents a move response with explicit failed state tracking"""
    move: str
    success: bool
    failed_move: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'move': self.move,
            'success': self.success,
            'failed_move': self.failed_move,
            'error': self.error,
        }

@dataclass
class Message:
    """Represents a conversation message with metadata"""
    role: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(
            role=data['role'],
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata')
        )

class MessageHistory:
    """Enhanced message history with move tracking"""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.messages: List[Message] = []
        self.moves_history: List[str] = []  # Track successful moves
        self.lock = asyncio.Lock()
    
    async def load(self):
        """Load messages and reconstruct moves history"""
        try:
            if self.file_path.exists():
                async with aiofiles.open(self.file_path, 'r') as f:
                    lines = await f.readlines()
                    self.messages = [
                        Message.from_dict(json.loads(line))
                        for line in lines
                        if line.strip()
                    ]
                    
                    # Reconstruct moves history
                    self.moves_history = [
                        msg.metadata['move']
                        for msg in self.messages
                        if msg.metadata and msg.metadata.get('move') 
                        and msg.metadata.get('success', False)
                    ]
        except Exception as e:
            logger.error(f"Error loading messages: {e}")
    
    async def append(self, message: Message):
        """Append message with atomic file operation"""
        async with self.lock:
            try:
                # Append to file first
                async with aiofiles.open(self.file_path, 'a') as f:
                    await f.write(json.dumps(message.to_dict()) + '\n')
                    await f.flush()
                    os.fsync(f.fileno())
                
                # Update in-memory state
                self.messages.append(message)
                if message.metadata and message.metadata.get('move') and message.metadata.get('success', False):
                    self.moves_history.append(message.metadata['move'])
                    
            except Exception as e:
                logger.error(f"Error appending message: {e}")
                raise
    
    def get_messages(self) -> List[Message]:
        """Get all messages"""
        return self.messages
    
    def get_failed_moves(self) -> List[str]:
        """Get history of failed moves"""
        return self.failed_moves_history

class BoardVisualizer:
    """Converts chess boards to various image formats using CairoSVG"""
    
    def __init__(self):
        self.size = 400  # Default board size in pixels
        
    def board_to_svg(self, board: chess.Board, flipped: bool = False) -> str:
        """Convert board to SVG string"""
        return chess.svg.board(
            board=board,
            size=self.size,
            flipped=flipped,
            coordinates=True,
            lastmove=board.peek() if board.move_stack else None
        )
    
    def svg_to_png(self, svg: str) -> bytes:
        """Convert SVG string to PNG bytes using CairoSVG"""
        try:
            return cairosvg.svg2png(
                bytestring=svg.encode('utf-8'),
                output_width=self.size,
                output_height=self.size
            )
        except Exception as e:
            logger.error(f"Error converting SVG to PNG: {e}")
            raise
    
    def optimize_png(self, png_bytes: bytes) -> bytes:
        """Optimize PNG for size while maintaining quality"""
        try:
            with io.BytesIO(png_bytes) as input_buffer:
                with Image.open(input_buffer) as img:
                    # Convert to RGB if needed
                    if img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    
                    output_buffer = io.BytesIO()
                    img.save(
                        output_buffer,
                        format='JPEG',
                        quality=85,
                        optimize=True
                    )
                    return output_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            raise
    
    def to_base64(self, image_bytes: bytes, format: str = 'jpeg') -> str:
        """Convert image bytes to base64 string with proper header"""
        try:
            b64_str = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/{format};base64,{b64_str}"
        except Exception as e:
            logger.error(f"Error converting to base64: {e}")
            raise
    
    def board_to_base64(self, board: chess.Board, flipped: bool = False) -> str:
        """Full conversion pipeline from board to base64 image"""
        try:
            svg = self.board_to_svg(board, flipped)
            png = self.svg_to_png(svg)
            optimized = self.optimize_png(png)
            
            # Debug: Save the optimized image to disk
            debug_path = Path("debug_board.jpg")
            with open(debug_path, "wb") as f:
                f.write(optimized)
            logger.info(f"Debug board image saved to {debug_path}")
            
            return self.to_base64(optimized)
        except Exception as e:
            logger.error(f"Error in board visualization pipeline: {e}")
            raise

class MessageFormatter:
    """Formats messages based on model capabilities"""
    
    def __init__(self, model_config: ModelConfig, board_visualizer: BoardVisualizer):
        self.model_config = model_config
        self.board_visualizer = board_visualizer
        
    def format_context(
        self,
        system_prompt: str,
        current_messages: List[Message],
        max_tokens: int,
        include_failures: bool = True
    ) -> List[Dict]:
        """Format complete context with proper token management"""
        formatted = [{
            "role": "system",
            "content": system_prompt
        }]
        
        # Reserve tokens for system prompt and response
        available_tokens = max_tokens * 0.8
        current_tokens = len(system_prompt) // 4
        
        # Add most recent messages that fit
        recent_messages = []
        for msg in reversed(current_messages[1:]):
            tokens = len(msg.content) // 4
            if current_tokens + tokens > available_tokens:
                break
            
            # Skip error messages unless including failures
            if msg.metadata and msg.metadata.get('error') and not include_failures:
                continue
                
            current_tokens += tokens
            recent_messages.append(msg)
        
        # Add messages in chronological order
        formatted.extend([
            {"role": msg.role, "content": msg.content}
            for msg in reversed(recent_messages)
        ])
        
        return formatted
    
    def format_move_request(
        self,
        board: chess.Board,
        prompt_text: str,
        perspective_color: str,
        system_prompt: str,
        current_messages: List[Message]
    ) -> List[Dict]:
        """Format complete move request with context"""
        try:
            messages = self.format_context(
                system_prompt,
                current_messages,
                self.model_config.max_tokens
            )
            
            # Add current move request
            if self.model_config.multimodal:
                flipped = perspective_color == 'black'
                board_image = self.board_visualizer.board_to_base64(board, flipped)
                
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": board_image}
                        }
                    ]
                })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt_text
                })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error formatting move request: {e}")
            raise

class LLMPlayer:
    """LLM-based chess player with multimodal support"""
    def __init__(self, model_name: str, color: str, game_directory: str):
        self.model = model_name
        self.color = color
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.game_directory = Path(game_directory)
        self.message_history = MessageHistory(
            self.game_directory / f"{color}_messages.jsonl"
        )

        # Get model configuration
        model_config = ModelManager().get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Invalid model name: {model_name}")
        
        self.model_config = model_config
        self.max_tokens = model_config.max_tokens
        
        # Initialize visualization and formatting services
        self.board_visualizer = BoardVisualizer()
        self.message_formatter = MessageFormatter(model_config, self.board_visualizer)

        self.consecutive_errors = 0
        self.MAX_CONSECUTIVE_ERRORS = 15
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    async def initialize(self):
        """Initialize player state"""
        await self.message_history.load()
        
        # Add system prompt if history is empty
        if not self.message_history.messages:
            await self.message_history.append(Message(
                role="system",
                content=SYSTEM_PROMPT,
                timestamp=datetime.now()
            ))
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (aiohttp.ClientError, json.JSONDecodeError, asyncio.TimeoutError)
        )
    )
    async def _get_llm_response(self, messages: List[Dict]) -> Dict:
        """Get response from LLM API with retry logic"""
        logger.info(f"Sending request to model: {self.model}")
        if self.model_config.multimodal:
            logger.info("Request includes board visualization")
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "LLM Chess Battle"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 1.0,
                        "max_tokens": 2500
                    },
                    timeout=30
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    
                    if 'choices' not in result:
                        # Log the full response for debugging
                        logger.error(f"Unexpected API response format: {json.dumps(result, indent=2)}")
                        raise aiohttp.ClientError("Missing choices in response")
                        
                    return result
                    
        except aiohttp.ClientResponseError as e:
            # Detailed HTTP error logging
            logger.error(f"""
    API Error Details:
    Status: {e.status}
    Message: {e.message}
    Headers: {json.dumps(dict(e.headers), indent=2) if e.headers else 'None'}
    Request Info: {json.dumps({
        'method': e.request_info.method,
        'url': str(e.request_info.url),
        'headers': dict(e.request_info.headers)
    }, indent=2)}
            """)
            raise
            
        except aiohttp.ClientError as e:
            # General client error logging
            logger.error(f"""
    Client Error Details:
    Error Type: {type(e).__name__}
    Error Message: {str(e)}
    Stack Trace:
    {traceback.format_exc()}
            """)
            raise
            
        except json.JSONDecodeError as e:
            # JSON parsing error logging
            logger.error(f"""
    JSON Decode Error:
    Error Message: {str(e)}
    Error Position: line {e.lineno}, column {e.colno}
    Error Line Content: {e.doc.splitlines()[e.lineno-1] if e.doc else 'N/A'}
    Full Response: {e.doc}
            """)
            raise
            
        except Exception as e:
            # Unexpected error logging
            logger.error(f"""
    Unexpected Error:
    Error Type: {type(e).__name__}
    Error Message: {str(e)}
    Stack Trace:
    {traceback.format_exc()}
    Request Messages: {json.dumps(messages, indent=2)}
            """)
            raise
            
    def _parse_move(self, content: str) -> Optional[str]:
        """Parse LLM response for chess move"""
        # Remove markdown and code blocks
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        
        # Find moves with standard or fallback patterns
        moves = re.findall(r'^MOVE:\s*(\S+)\s*$', content, re.MULTILINE | re.IGNORECASE)
        if not moves:
            moves = re.findall(r'(?:^|\n)MOVE:\s*(\S+)(?:\s|$)', content, re.IGNORECASE)
            
        if moves:
            move = moves[-1].strip()
            move = re.sub(r'["\'\.,;:\(\)\[\]\{\}]+', '', move)
            return move
            
        return None
    
    async def get_move(self, board: chess.Board, pgn: str, 
                    current_player: str, 
                    opponent_move: Optional[str] = None,
                    failed_move: Optional[str] = None) -> MoveResponse:
        """Get next move from LLM with full context and error handling"""
        try:
            # Format prompt text
            if failed_move:
                prompt_text = format_error_message(failed_move, board)
                include_failures = True
            else:
                prompt_text = format_move_request(board, pgn, current_player, opponent_move)
                include_failures = False
            
            # Get system prompt from history
            system_prompt = self.message_history.messages[0].content if self.message_history.messages else SYSTEM_PROMPT
            
            # Format complete message with context
            messages = self.message_formatter.format_move_request(
                board=board,
                prompt_text=prompt_text,
                perspective_color=self.color,
                system_prompt=system_prompt,
                current_messages=self.message_history.messages
            )
            
            # Store text version in history
            text_content = (messages[-1]['content'][0]['text'] 
                          if isinstance(messages[-1]['content'], list) 
                          else messages[-1]['content'])
            
            await self.message_history.append(Message(
                role="user",
                content=text_content,
                timestamp=datetime.now(),
                metadata={
                    "fen": board.fen(),
                    "opponent_move": opponent_move,
                    "failed_move": failed_move,
                    "consecutive_errors": self.consecutive_errors,
                    "multimodal": self.model_config.multimodal
                }
            ))
            
            # Get LLM response
            response_data = await self._get_llm_response(messages)
            
            # Parse response
            content = response_data["choices"][0]["message"]["content"]
            move = self._parse_move(content)
            
            if not move:
                self.consecutive_errors += 1
                error_msg = f"Invalid response format (Consecutive errors: {self.consecutive_errors})"
                logger.error(error_msg)
                
                await self.message_history.append(Message(
                    role="assistant",
                    content=content,
                    timestamp=datetime.now(),
                    metadata={
                        "error": error_msg,
                        "parse_failed": True,
                        "consecutive_errors": self.consecutive_errors
                    }
                ))
                return MoveResponse(
                    move="",
                    success=False,
                    failed_move=True,
                    error=error_msg
                )
            
            # Validate move
            try:
                parsed_move = board.parse_san(move)
                if parsed_move not in board.legal_moves:
                    raise chess.IllegalMoveError
                
                # Success - reset error counter
                self.consecutive_errors = 0
                
                await self.message_history.append(Message(
                    role="assistant",
                    content=content,
                    timestamp=datetime.now(),
                    metadata={
                        "move": move,
                        "success": True,
                        "consecutive_errors": 0
                    }
                ))
                
                return MoveResponse(
                    move=move,
                    success=True,
                    failed_move=False
                )
                
            except (chess.IllegalMoveError, chess.InvalidMoveError):
                self.consecutive_errors += 1
                error_msg = f"Invalid move: {move} (Consecutive errors: {self.consecutive_errors})"
                logger.error(error_msg)
                
                await self.message_history.append(Message(
                    role="assistant",
                    content=content,
                    timestamp=datetime.now(),
                    metadata={
                        "move": move,
                        "error": error_msg,
                        "failed_move": move,
                        "consecutive_errors": self.consecutive_errors
                    }
                ))
                
                return MoveResponse(
                    move=move,
                    success=False,
                    failed_move=True,
                    error=error_msg
                )
                
        except Exception as e:
            self.consecutive_errors += 1
            error_msg = f"Error getting move: {str(e)} (Consecutive errors: {self.consecutive_errors})"
            logger.error(error_msg)
            
            await self.message_history.append(Message(
                role="error",
                content=error_msg,
                timestamp=datetime.now(),
                metadata={
                    "error": str(e),
                    "consecutive_errors": self.consecutive_errors
                }
            ))
            
            return MoveResponse(
                move="",
                success=False,
                failed_move=True,
                error=error_msg
            )
    
    def get_failure_count(self) -> int:
        """Get count of failed moves"""
        return sum(
            1 for m in self.message_history.messages
            if m.metadata and m.metadata.get('failed_move')
        )