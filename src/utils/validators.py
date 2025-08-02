"""
Input validation utilities for Simple Decision AI.

This module provides input validation and sanitization functions.
"""

import re
from typing import Optional, List, Tuple


class InputValidator:
    """Validates and sanitizes user input."""
    
    def __init__(self, max_length: int = 1000):
        """
        Initialize the input validator.
        
        Args:
            max_length: Maximum allowed input length
        """
        self.max_length = max_length
        
        # Regex patterns for validation
        self.allowed_chars_pattern = re.compile(r'^[a-zA-Z0-9\s\.,!?;:\-_()"\'\[\]]+$')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def validate_text_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Validate text input for decision making.
        
        Args:
            text: Input text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(text, str):
            return False, "Input must be a string"
        
        if not text.strip():
            return False, "Input cannot be empty"
        
        if len(text) > self.max_length:
            return False, f"Input too long (max {self.max_length} characters)"
        
        if not self.allowed_chars_pattern.match(text):
            return False, "Input contains invalid characters"
        
        return True, None
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text input by removing/normalizing problematic characters.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize whitespace (replace multiple spaces with single space)
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove null bytes and other control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        # Truncate to max length
        if len(text) > self.max_length:
            text = text[:self.max_length].strip()
        
        return text
    
    def is_question(self, text: str) -> bool:
        """
        Check if the input text is a question.
        
        Args:
            text: Input text to check
            
        Returns:
            True if text appears to be a question
        """
        text = text.strip().lower()
        
        # Check for question words at the beginning
        question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which', 
                         'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 
                         'could', 'would', 'should', 'will', 'shall']
        
        first_word = text.split()[0] if text.split() else ""
        
        # Check for question mark or question words
        return text.endswith('?') or first_word in question_words
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract keywords from input text.
        
        Args:
            text: Input text
            min_length: Minimum keyword length
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction (can be improved with NLP libraries)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 
                     'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 
                     'an', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 
                     'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        keywords = [word for word in words 
                   if len(word) >= min_length and word not in stop_words]
        
        return list(set(keywords))  # Remove duplicates
    
    def validate_confidence_threshold(self, threshold: float) -> Tuple[bool, Optional[str]]:
        """
        Validate confidence threshold value.
        
        Args:
            threshold: Confidence threshold to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(threshold, (int, float)):
            return False, "Confidence threshold must be a number"
        
        if not 0.0 <= threshold <= 1.0:
            return False, "Confidence threshold must be between 0.0 and 1.0"
        
        return True, None


# Global validator instance
input_validator = InputValidator()