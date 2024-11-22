# base/llm_player.py

from typing import List, Dict, Optional
from pathlib import Path
import requests
import json
import time
import os
from datetime import datetime
from dataclasses import dataclass

@dataclass
class PlayerConfig:
    """Configuration for LLM player"""
    model_name: str
    player_id: int
    run_dir: Path
    context_window: int = 2000000
    max_retries: int = 5
    retry_delay: int = 1
    multimodal: bool = True

class BaseLLMPlayer:
    """Base class for LLM players that can be extended by specific games"""
    
    def __init__(self, config: PlayerConfig):
        """Initialize the LLM player with configuration"""
        self.config = config
        self.messages: List[Dict[str, str]] = []
        self.model_name = config.model_name
        self.player_id = config.player_id
        
        # API configuration
        self.api_key: str = os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
            
        # Logging setup
        timestamp = datetime.now().isoformat().replace(':', '-')
        safe_model_name = self.model_name.replace('/', '_')
        self.log_file = config.run_dir / f"{safe_model_name}_p{self.player_id}_{timestamp}.jsonl"
        self.start_time = timestamp
        
    def initialize_with_prompt(self, system_prompt: Dict[str, str]) -> None:
        """Initialize the conversation with a system prompt"""
        self.messages.append(system_prompt)
        self.log_message(system_prompt)
        
    def get_response(self, message: Dict[str, str], game_image: Optional[str] = None) -> str:
        """
        Get a response from the LLM model
        
        Args:
            message: The message to send to the model
            game_image: Optional base64 encoded image for multimodal models
            
        Returns:
            str: The model's response
            
        Raises:
            Exception: If unable to get a valid response after max retries
        """
        self.messages.append(message)
        self.log_message(message)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/game-framework",
            "X-Title": "Game Framework"
        }
        
        for attempt in range(self.config.max_retries):
            try:
                request_body = {
                    "model": self.model_name,
                    "messages": self.messages
                }
                
                # Handle multimodal content if supported and provided
                if self.config.multimodal and game_image:
                    last_message = request_body["messages"][-1].copy()
                    last_message["content"] = [
                        {"type": "text", "text": last_message["content"]},
                        {
                            "type": "image_url",
                            "image_url": {"url": game_image}
                        }
                    ]
                    request_body["messages"][-1] = last_message
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=request_body,
                    timeout=30
                )
                
                response.raise_for_status()
                response_json = response.json()
                
                # Handle different API response formats
                content = None
                if "choices" in response_json:
                    # OpenAI format
                    content = response_json["choices"][0]["message"]["content"]
                elif "candidates" in response_json:
                    # Gemini format
                    content = response_json["candidates"][0]["content"]["parts"][0]["text"]
                else:
                    raise ValueError(f"Unknown API response format: {response_json.keys()}")
                
                assistant_message = {
                    "role": "assistant",
                    "content": content
                }
                
                self.messages.append(assistant_message)
                self.log_message(assistant_message)
                
                return content
                
            except requests.exceptions.RequestException as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"Failed to get response after {self.config.max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed. Retrying in {self.config.retry_delay} seconds...")
                time.sleep(self.config.retry_delay)
            
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(f"Unexpected error after {self.config.max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying in {self.config.retry_delay} seconds...")
                time.sleep(self.config.retry_delay)
    
    def log_message(self, message: Dict[str, str]) -> None:
        """Log a message to the player's JSONL file"""
        if self.log_file:
            with open(self.log_file, 'a') as f:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "player_id": self.player_id,
                    "model": self.model_name,
                    **message
                }
                json.dump(entry, f)
                f.write('\n')
    
    def get_game_specific_data(self) -> Dict:
        """
        Override this method in game-specific player classes to provide
        additional data needed for that specific game
        """
        return {}
    
    @property
    def name(self) -> str:
        """Get player name (model name + ID)"""
        return f"{self.model_name}_player_{self.player_id}"