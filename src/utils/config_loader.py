"""
Configuration loader for Simple Decision AI.

This module provides functionality to load and manage configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the configuration file (without .yaml extension)
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid
        """
        if config_name in self._configs:
            return self._configs[config_name]
            
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self._configs[config_name] = config
                return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file {config_path}: {e}")
    
    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific configuration value using dot notation.
        
        Args:
            config_name: Name of the configuration file
            key_path: Dot-separated path to the configuration key (e.g., 'model.name')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        config = self.load_config(config_name)
        
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.load_config("model_config")
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.load_config("training_config")
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.load_config("app_config")
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration values.
        
        Args:
            config_name: Name of the configuration file
            updates: Dictionary of updates to apply
        """
        config = self.load_config(config_name)
        config.update(updates)
        self._configs[config_name] = config
    
    def reload_config(self, config_name: str) -> Dict[str, Any]:
        """
        Reload a configuration file from disk.
        
        Args:
            config_name: Name of the configuration file
            
        Returns:
            Reloaded configuration dictionary
        """
        if config_name in self._configs:
            del self._configs[config_name]
        return self.load_config(config_name)
    
    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._configs.clear()


# Global configuration loader instance
config_loader = ConfigLoader()