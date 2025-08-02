"""
Helper utilities for Simple Decision AI.

This module provides various helper functions and utilities.
"""

import os
import json
import torch
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(data: Any, file_path: Union[str, Path]) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to the JSON file
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    
    Returns:
        PyTorch device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def format_decision_output(
    decision: str,
    confidence: float,
    reasoning: Optional[str] = None,
    input_text: Optional[str] = None,
    explanation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format decision output in a standardized format.
    
    Args:
        decision: The decision (Yes/No)
        confidence: Confidence score (0.0 to 1.0)
        reasoning: Optional reasoning explanation
        input_text: Optional input text that was processed
        explanation: Optional simple English explanation for the choice
        
    Returns:
        Formatted decision dictionary
    """
    output = {
        "decision": decision,
        "confidence": round(confidence, 4),
        "timestamp": get_current_timestamp()
    }
    
    if reasoning:
        output["reasoning"] = reasoning
    
    if input_text:
        output["input"] = input_text
    
    if explanation:
        output["explanation"] = explanation
    
    return output


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp string
    """
    from datetime import datetime
    return datetime.now().isoformat()


def calculate_memory_usage() -> Dict[str, float]:
    """
    Calculate current memory usage.
    
    Returns:
        Dictionary with memory usage information (in MB)
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / 1024 / 1024,  # Resident Set Size
        "vms": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        "percent": process.memory_percent()
    }


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace and converting to lowercase.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def is_valid_model_path(model_path: Union[str, Path]) -> bool:
    """
    Check if a model path is valid.
    
    Args:
        model_path: Path to check
        
    Returns:
        True if path is valid for a model
    """
    path = Path(model_path)
    
    # Check if it's a directory with model files
    if path.is_dir():
        model_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        return any((path / file).exists() for file in model_files)
    
    # Check if it's a model file
    if path.is_file():
        return path.suffix in ['.bin', '.pt', '.pth', '.safetensors']
    
    return False


# Note: Explanation generation has been moved to SmartExplanationEngine
# This maintains backward compatibility for any remaining code that might import this function