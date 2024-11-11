from dataclasses import dataclass
from typing import Dict, List, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration with multimodal support"""
    display_name: str
    model_name: str
    max_tokens: int
    multimodal: bool = False
    
class ModelManager:
    """Manages model configurations and provides access to model data"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        if not self.initialized:
            self.models: Dict[str, ModelConfig] = {}
            self.load_models()
            self.initialized = True
    
    def load_models(self):
        """Load models from models.jsonl"""
        try:
            models_file = Path("models.jsonl")
            if not models_file.exists():
                raise FileNotFoundError("models.jsonl not found")
                
            with open(models_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        config = ModelConfig(
                            display_name=data['display_name'],
                            model_name=data['model_name'],
                            max_tokens=data['max_tokens'],
                            multimodal=data.get('multimodal', False)
                        )
                        self.models[config.model_name] = config
        except Exception as e:
            logger.error(f"Error loading model configurations: {e}")
            raise
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        return self.models.get(model_name)
    
    def get_all_models(self) -> List[ModelConfig]:
        """Get all available models"""
        return list(self.models.values())
    
    def validate_model(self, model_name: str) -> bool:
        """Check if model exists and is valid"""
        return model_name in self.models