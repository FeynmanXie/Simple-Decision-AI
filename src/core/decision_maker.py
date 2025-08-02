"""
Decision maker module for Simple Decision AI.

This module provides the main interface for making binary decisions.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .inference_engine import InferenceEngine
from .rule_engine import rule_engine
from utils.config_loader import config_loader
from utils.logger import LoggerMixin
from utils.validators import input_validator
from utils.helpers import format_decision_output, save_json, load_json


class DecisionMaker(LoggerMixin):
    """Main class for making binary decisions using AI."""
    
    def __init__(self, config_path: str = "model_config"):
        """
        Initialize the decision maker.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = config_loader.load_config(config_path)
        
        # Initialize inference engine
        self.inference_engine = InferenceEngine(config_path)
        
        # Use rule engine for better decisions
        self.use_rule_engine = True
        
        # Decision history
        self.decision_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Statistics
        self.stats = {
            'total_decisions': 0,
            'yes_decisions': 0,
            'no_decisions': 0,
            'high_confidence_decisions': 0,
            'low_confidence_decisions': 0
        }
        
        self.logger.info("DecisionMaker initialized")
    
    def initialize(self) -> None:
        """Initialize the decision maker by loading models."""
        try:
            self.logger.info("Initializing DecisionMaker...")
            self.inference_engine.initialize()
            self.logger.info("DecisionMaker initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize DecisionMaker: {e}")
            raise
    
    def decide(
        self, 
        text: str, 
        include_reasoning: bool = True,
        save_to_history: bool = True
    ) -> Dict[str, Any]:
        """
        Make a binary decision based on input text.
        
        Args:
            text: Input text to make decision on
            include_reasoning: Whether to include reasoning in the output
            save_to_history: Whether to save decision to history
            
        Returns:
            Dictionary containing decision, confidence, and optional reasoning
        """
        try:
            # Validate input
            is_valid, error_msg = input_validator.validate_text_input(text)
            if not is_valid:
                raise ValueError(f"Invalid input: {error_msg}")
            
            # Try rule engine first if enabled
            if self.use_rule_engine:
                rule_result = rule_engine.evaluate(text)
                if rule_result:
                    decision, confidence, reasoning = rule_result
                    result = format_decision_output(
                        decision=decision,
                        confidence=confidence,
                        reasoning=reasoning if include_reasoning else None,
                        input_text=text
                    )
                    self.logger.info(f"Rule-based decision: {result['decision']} (confidence: {result['confidence']:.4f})")
                else:
                    # Fall back to AI model
                    result = self.inference_engine.predict(text, return_reasoning=include_reasoning)
                    result['reasoning'] = f"AI model (fallback): {result.get('reasoning', '')}" if include_reasoning else None
                    self.logger.info(f"AI model decision: {result['decision']} (confidence: {result['confidence']:.4f})")
            else:
                # Use only AI model
                result = self.inference_engine.predict(text, return_reasoning=include_reasoning)
            
            # Update statistics
            self._update_stats(result)
            
            # Save to history if requested
            if save_to_history:
                self._add_to_history(result)
            
            self.logger.info(f"Decision made: {result['decision']} (confidence: {result['confidence']:.4f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Decision making failed: {e}")
            raise
    
    def decide_batch(
        self, 
        texts: List[str], 
        include_reasoning: bool = True,
        save_to_history: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Make binary decisions for a batch of texts.
        
        Args:
            texts: List of input texts
            include_reasoning: Whether to include reasoning in outputs
            save_to_history: Whether to save decisions to history
            
        Returns:
            List of decision dictionaries
        """
        try:
            # Validate all inputs
            for i, text in enumerate(texts):
                is_valid, error_msg = input_validator.validate_text_input(text)
                if not is_valid:
                    raise ValueError(f"Invalid input at index {i}: {error_msg}")
            
            # Make individual decisions to ensure rule engine is used
            results = []
            for text in texts:
                result = self.decide(text, include_reasoning=include_reasoning, save_to_history=False)
                results.append(result)
            
            # Update statistics and history
            for result in results:
                self._update_stats(result)
                if save_to_history:
                    self._add_to_history(result)
            
            self.logger.info(f"Batch decisions completed for {len(texts)} texts")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch decision making failed: {e}")
            raise
    
    def is_confident_decision(self, result: Dict[str, Any]) -> bool:
        """
        Check if a decision result is confident.
        
        Args:
            result: Decision result dictionary
            
        Returns:
            True if decision is confident
        """
        confidence = result.get('confidence', 0.0)
        return self.inference_engine.is_confident(confidence)
    
    def get_decision_summary(self, result: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of a decision.
        
        Args:
            result: Decision result dictionary
            
        Returns:
            Summary string
        """
        decision = result.get('decision', 'Unknown')
        confidence = result.get('confidence', 0.0)
        
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        summary = f"Decision: {decision} (confidence: {confidence:.2%}, level: {confidence_level})"
        
        if 'reasoning' in result:
            summary += f"\nReasoning: {result['reasoning']}"
        
        return summary
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for decisions.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.inference_engine.set_confidence_threshold(threshold)
        self.logger.info(f"Confidence threshold updated to {threshold}")
    
    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.inference_engine.get_confidence_threshold()
    
    def set_rule_engine_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the rule engine.
        
        Args:
            enabled: Whether to use rule engine
        """
        self.use_rule_engine = enabled
        mode = "enabled" if enabled else "disabled"
        self.logger.info(f"Rule engine {mode}")
    
    def is_rule_engine_enabled(self) -> bool:
        """Check if rule engine is enabled."""
        return self.use_rule_engine
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        return rule_engine.get_rule_stats()
    
    def _update_stats(self, result: Dict[str, Any]) -> None:
        """
        Update decision statistics.
        
        Args:
            result: Decision result dictionary
        """
        self.stats['total_decisions'] += 1
        
        decision = result.get('decision', '').lower()
        if decision == 'yes':
            self.stats['yes_decisions'] += 1
        elif decision == 'no':
            self.stats['no_decisions'] += 1
        
        confidence = result.get('confidence', 0.0)
        if self.inference_engine.is_confident(confidence):
            self.stats['high_confidence_decisions'] += 1
        else:
            self.stats['low_confidence_decisions'] += 1
    
    def _add_to_history(self, result: Dict[str, Any]) -> None:
        """
        Add decision result to history.
        
        Args:
            result: Decision result dictionary
        """
        self.decision_history.append(result)
        
        # Limit history size
        if len(self.decision_history) > self.max_history_size:
            self.decision_history = self.decision_history[-self.max_history_size:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get decision statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = self.stats.copy()
        
        # Calculate percentages
        total = stats['total_decisions']
        if total > 0:
            stats['yes_percentage'] = (stats['yes_decisions'] / total) * 100
            stats['no_percentage'] = (stats['no_decisions'] / total) * 100
            stats['high_confidence_percentage'] = (stats['high_confidence_decisions'] / total) * 100
            stats['low_confidence_percentage'] = (stats['low_confidence_decisions'] / total) * 100
        else:
            stats['yes_percentage'] = 0
            stats['no_percentage'] = 0
            stats['high_confidence_percentage'] = 0
            stats['low_confidence_percentage'] = 0
        
        return stats
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get decision history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of decision history items
        """
        if limit is None:
            return self.decision_history.copy()
        else:
            return self.decision_history[-limit:] if limit > 0 else []
    
    def clear_history(self) -> None:
        """Clear decision history."""
        self.decision_history.clear()
        self.logger.info("Decision history cleared")
    
    def reset_statistics(self) -> None:
        """Reset decision statistics."""
        self.stats = {
            'total_decisions': 0,
            'yes_decisions': 0,
            'no_decisions': 0,
            'high_confidence_decisions': 0,
            'low_confidence_decisions': 0
        }
        self.logger.info("Statistics reset")
    
    def save_history(self, file_path: str) -> None:
        """
        Save decision history to a file.
        
        Args:
            file_path: Path to save the history file
        """
        try:
            history_data = {
                'history': self.decision_history,
                'statistics': self.get_statistics(),
                'config': {
                    'confidence_threshold': self.get_confidence_threshold(),
                    'max_history_size': self.max_history_size
                }
            }
            
            save_json(history_data, file_path)
            self.logger.info(f"History saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")
            raise
    
    def load_history(self, file_path: str) -> None:
        """
        Load decision history from a file.
        
        Args:
            file_path: Path to the history file
        """
        try:
            history_data = load_json(file_path)
            
            self.decision_history = history_data.get('history', [])
            
            # Restore statistics if available
            if 'statistics' in history_data:
                saved_stats = history_data['statistics']
                # Only restore the basic counters
                for key in ['total_decisions', 'yes_decisions', 'no_decisions', 
                           'high_confidence_decisions', 'low_confidence_decisions']:
                    if key in saved_stats:
                        self.stats[key] = saved_stats[key]
            
            self.logger.info(f"History loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load history: {e}")
            raise
    
    def warm_up(self) -> None:
        """Warm up the decision maker."""
        try:
            self.logger.info("Warming up DecisionMaker...")
            self.inference_engine.warm_up()
            self.logger.info("DecisionMaker warm-up completed")
        except Exception as e:
            self.logger.warning(f"DecisionMaker warm-up failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying model.
        
        Returns:
            Dictionary with model information
        """
        return self.inference_engine.get_model_info()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.inference_engine.cleanup()
        self.logger.info("DecisionMaker cleaned up")