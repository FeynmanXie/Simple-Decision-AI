"""
Text processing module for Simple Decision AI.

This module handles text preprocessing and tokenization for the AI model.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, PreTrainedTokenizer

from utils.config_loader import config_loader
from utils.logger import LoggerMixin
from utils.validators import input_validator


class TextProcessor(LoggerMixin):
    """Handles text preprocessing and tokenization."""
    
    def __init__(self, model_name: Optional[str] = None, config_path: str = "model_config"):
        """
        Initialize the text processor.
        
        Args:
            model_name: Name of the model/tokenizer to use
            config_path: Path to the configuration file
        """
        self.config = config_loader.load_config(config_path)
        self.model_name = model_name or self.config.get('model', {}).get('name', 'bert-base-uncased')
        
        # Load tokenizer configuration
        tokenizer_config = self.config.get('tokenizer', {})
        self.max_length = tokenizer_config.get('max_length', 512)
        self.padding = tokenizer_config.get('padding', 'max_length')
        self.truncation = tokenizer_config.get('truncation', True)
        self.return_attention_mask = tokenizer_config.get('return_attention_mask', True)
        self.return_token_type_ids = tokenizer_config.get('return_token_type_ids', True)
        
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._load_tokenizer()
    
    def _load_tokenizer(self) -> None:
        """Load the tokenizer."""
        try:
            self.logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.config.get('model', {}).get('cache_dir', None)
            )
            self.logger.info("Tokenizer loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before tokenization.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        # Validate and sanitize input
        is_valid, error_msg = input_validator.validate_text_input(text)
        if not is_valid:
            raise ValueError(f"Invalid input text: {error_msg}")
        
        # Sanitize text
        text = input_validator.sanitize_text(text)
        
        # Additional preprocessing steps
        text = self._normalize_text(text)
        text = self._handle_special_cases(text)
        
        return text
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text format.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase if model expects it
        if self.config.get('tokenizer', {}).get('do_lower_case', True):
            text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _handle_special_cases(self, text: str) -> str:
        """
        Handle special cases in text preprocessing.
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        # Handle contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Handle multiple punctuation marks
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        return text
    
    def tokenize(self, text: str) -> Dict[str, Union[List[int], List[List[int]]]]:
        """
        Tokenize text for model input.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Dictionary containing tokenized inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize
        try:
            encoded = self.tokenizer(
                processed_text,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=self.return_attention_mask,
                return_token_type_ids=self.return_token_type_ids,
                return_tensors=None  # Return as lists first
            )
            
            self.logger.debug(f"Tokenized text length: {len(encoded['input_ids'])}")
            return encoded
            
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            raise
    
    def batch_tokenize(self, texts: List[str]) -> Dict[str, List[List[int]]]:
        """
        Tokenize multiple texts in batch.
        
        Args:
            texts: List of texts to tokenize
            
        Returns:
            Dictionary containing batch tokenized inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Batch tokenize
        try:
            encoded = self.tokenizer(
                processed_texts,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_attention_mask=self.return_attention_mask,
                return_token_type_ids=self.return_token_type_ids,
                return_tensors=None
            )
            
            self.logger.debug(f"Batch tokenized {len(texts)} texts")
            return encoded
            
        except Exception as e:
            self.logger.error(f"Batch tokenization failed: {e}")
            raise
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        try:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            return text
        except Exception as e:
            self.logger.error(f"Decoding failed: {e}")
            raise
    
    def get_token_info(self, text: str) -> Dict[str, Union[int, List[str]]]:
        """
        Get detailed token information for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with token information
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        processed_text = self.preprocess_text(text)
        encoded = self.tokenize(processed_text)
        
        # Get tokens as strings
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        
        return {
            'num_tokens': len(encoded['input_ids']),
            'tokens': tokens,
            'input_ids': encoded['input_ids'],
            'is_truncated': len(encoded['input_ids']) >= self.max_length
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens for text without full tokenization.
        
        Args:
            text: Input text
            
        Returns:
            Estimated number of tokens
        """
        # Simple estimation: roughly 4 characters per token for English
        return max(1, len(text) // 4)
    
    def is_text_too_long(self, text: str) -> bool:
        """
        Check if text would be too long after tokenization.
        
        Args:
            text: Input text
            
        Returns:
            True if text is too long
        """
        estimated_tokens = self.estimate_tokens(text)
        return estimated_tokens > self.max_length