from typing import List, Dict, Optional
import chess
from model_config import ModelConfig
from board_visualizer import BoardVisualizer
import logging

logger = logging.getLogger(__name__)

class MessageFormatter:
    """Formats messages based on model capabilities"""
    
    def __init__(self, model_config: ModelConfig, board_visualizer: BoardVisualizer):
        self.model_config = model_config
        self.board_visualizer = board_visualizer
    
    def format_move_request(
        self,
        board: chess.Board,
        prompt_text: str,
        perspective_color: str
    ) -> List[Dict]:
        """Format move request based on model capabilities"""
        try:
            if not self.model_config.multimodal:
                return [{"role": "user", "content": prompt_text}]
            
            # For multimodal models, include board visualization
            flipped = perspective_color == 'black'
            board_image = self.board_visualizer.board_to_base64(board, flipped)
            
            return [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": board_image}
                    }
                ]
            }]
        except Exception as e:
            logger.error(f"Error formatting move request: {e}")
            raise
    
    def format_error_message(
        self,
        board: chess.Board,
        failed_move: str,
        prompt_text: str,
        perspective_color: str
    ) -> List[Dict]:
        """Format error message for failed moves"""
        try:
            error_text = f"Your previous move '{failed_move}' was invalid. {prompt_text}"
            
            if not self.model_config.multimodal:
                return [{"role": "user", "content": error_text}]
            
            # Include board visualization for multimodal models
            flipped = perspective_color == 'black'
            board_image = self.board_visualizer.board_to_base64(board, flipped)
            
            return [{
                "role": "user",
                "content": [
                    {"type": "text", "text": error_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": board_image}
                    }
                ]
            }]
        except Exception as e:
            logger.error(f"Error formatting error message: {e}")
            raise