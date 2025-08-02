"""
Model management module for Simple Decision AI.

This module handles loading, caching, and managing AI models.
"""

import torch
import warnings
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from utils.config_loader import config_loader
from utils.logger import LoggerMixin
from utils.helpers import get_device, is_valid_model_path


class ModelManager(LoggerMixin):
    """Manages AI model loading and caching."""
    
    def __init__(self, config_path: str = "model_config"):
        """
        Initialize the model manager.
        
        Args:
            config_path: Path to the model configuration file
        """
        self.config = config_loader.load_config(config_path)
        self.model_config = self.config.get('model', {})
        
        self.model_name = self.model_config.get('name', 'bert-base-uncased')
        self.cache_dir = self.model_config.get('cache_dir', './models/pretrained')
        self.num_labels = self.model_config.get('num_labels', 2)
        self.device = get_device()
        
        self.model: Optional[torch.nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._model_loaded = False
        
        self.logger.info(f"ModelManager initialized with device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None, force_reload: bool = False) -> None:
        """
        Load the AI model.
        
        Args:
            model_path: Optional path to a specific model
            force_reload: Whether to force reload even if model is already loaded
        """
        if self._model_loaded and not force_reload:
            self.logger.info("Model already loaded, skipping...")
            return
        
        try:
            model_path = model_path or self.model_name
            self.logger.info(f"Loading model: {model_path}")
            
            # Check if it's a local path or model name
            if is_valid_model_path(model_path):
                self.logger.info("Loading from local path")
                model_path = str(Path(model_path).resolve())
            else:
                self.logger.info("Loading from Hugging Face Hub")
            
            # Load model for sequence classification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=self.num_labels,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,
                device_map=None  # We'll move to device manually
            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self._model_loaded = True
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def load_tokenizer(self, tokenizer_path: Optional[str] = None) -> None:
        """
        Load the tokenizer.
        
        Args:
            tokenizer_path: Optional path to a specific tokenizer
        """
        try:
            tokenizer_path = tokenizer_path or self.model_name
            self.logger.info(f"Loading tokenizer: {tokenizer_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                cache_dir=self.cache_dir
            )
            
            self.logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def get_model(self) -> torch.nn.Module:
        """
        Get the loaded model.
        
        Returns:
            The loaded model
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        return self.model
    
    def get_tokenizer(self) -> AutoTokenizer:
        """
        Get the loaded tokenizer.
        
        Returns:
            The loaded tokenizer
            
        Raises:
            RuntimeError: If tokenizer is not loaded
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        return self.tokenizer
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded
        """
        return self._model_loaded and self.model is not None
    
    def is_tokenizer_loaded(self) -> bool:
        """
        Check if tokenizer is loaded.
        
        Returns:
            True if tokenizer is loaded
        """
        return self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._model_loaded:
            return {"status": "not_loaded"}
        
        model_info = {
            "status": "loaded",
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
        }
        
        return model_info
    
    def move_to_device(self, device: Optional[torch.device] = None) -> None:
        """
        Move model to a specific device.
        
        Args:
            device: Target device (defaults to auto-detected device)
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        if device is None:
            device = get_device()
        
        self.logger.info(f"Moving model to device: {device}")
        self.model.to(device)
        self.device = device
    
    def save_model(self, save_path: str) -> None:
        """
        Save the current model to disk.
        
        Args:
            save_path: Path to save the model
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Saving model to: {save_path}")
            self.model.save_pretrained(save_path)
            
            if self.tokenizer:
                self.tokenizer.save_pretrained(save_path)
            
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        self._model_loaded = False
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Model unloaded from memory")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage information.
        
        Returns:
            Dictionary with memory usage info
        """
        memory_info = {}
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # Add system memory info
        try:
            import psutil
            process = psutil.Process()
            memory_info['system_rss'] = process.memory_info().rss / 1024**2  # MB
            memory_info['system_percent'] = process.memory_percent()
        except ImportError:
            pass
        
        return memory_info