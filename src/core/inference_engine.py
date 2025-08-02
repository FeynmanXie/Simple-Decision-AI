"""
Inference engine for Simple Decision AI.

This module handles model inference and prediction logic.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .model_manager import ModelManager
from .text_processor import TextProcessor
from utils.config_loader import config_loader
from utils.logger import LoggerMixin
from utils.helpers import format_decision_output


class InferenceEngine(LoggerMixin):
    """Handles model inference for decision making."""
    
    def __init__(self, config_path: str = "model_config"):
        """
        Initialize the inference engine.
        
        Args:
            config_path: Path to the model configuration file
        """
        self.config = config_loader.load_config(config_path)
        self.inference_config = self.config.get('inference', {})
        
        # Initialize components
        self.model_manager = ModelManager(config_path)
        self.text_processor = TextProcessor(config_path=config_path)
        
        # Configuration
        self.confidence_threshold = self.inference_config.get('confidence_threshold', 0.5)
        self.batch_size = self.inference_config.get('batch_size', 1)
        self.max_length = self.inference_config.get('max_length', 512)
        
        # Label mappings
        self.labels = self.config.get('labels', {0: "No", 1: "Yes"})
        self.label_to_id = self.config.get('label_to_id', {"No": 0, "Yes": 1})
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize the inference engine by loading models."""
        if self._initialized:
            self.logger.info("Inference engine already initialized")
            return
        
        try:
            self.logger.info("Initializing inference engine...")
            
            # Load model and tokenizer
            self.model_manager.load_model()
            self.model_manager.load_tokenizer()
            
            self._initialized = True
            self.logger.info("Inference engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize inference engine: {e}")
            raise
    
    def predict(self, text: str, return_reasoning: bool = True) -> Dict[str, Any]:
        """
        Make a prediction for a single text input.
        
        Args:
            text: Input text to make decision on
            return_reasoning: Whether to include reasoning in output
            
        Returns:
            Dictionary containing decision, confidence, and optional reasoning
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Process and tokenize input
            tokenized = self.text_processor.tokenize(text)
            
            # Convert to tensors and move to device
            input_ids = torch.tensor([tokenized['input_ids']], device=self.model_manager.device)
            attention_mask = torch.tensor([tokenized['attention_mask']], device=self.model_manager.device)
            
            # Get model prediction
            with torch.no_grad():
                model = self.model_manager.get_model()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probabilities = F.softmax(logits, dim=-1)
                confidence_scores = probabilities[0].cpu().numpy()
                
                # Get predicted class
                predicted_class = torch.argmax(logits, dim=-1).item()
                confidence = float(confidence_scores[predicted_class])
                
                # Map to decision
                decision = self.labels.get(predicted_class, "Unknown")
                
                # Generate reasoning if requested
                reasoning = None
                if return_reasoning:
                    reasoning = self._generate_reasoning(text, decision, confidence, confidence_scores)
                
                # Format output
                result = format_decision_output(
                    decision=decision,
                    confidence=confidence,
                    reasoning=reasoning,
                    input_text=text
                )
                
                self.logger.debug(f"Prediction: {decision} (confidence: {confidence:.4f})")
                return result
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, texts: List[str], return_reasoning: bool = True) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of texts.
        
        Args:
            texts: List of input texts
            return_reasoning: Whether to include reasoning in outputs
            
        Returns:
            List of prediction dictionaries
        """
        if not self._initialized:
            self.initialize()
        
        try:
            results = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize batch
                tokenized = self.text_processor.batch_tokenize(batch_texts)
                
                # Convert to tensors
                input_ids = torch.tensor(tokenized['input_ids'], device=self.model_manager.device)
                attention_mask = torch.tensor(tokenized['attention_mask'], device=self.model_manager.device)
                
                # Get model predictions
                with torch.no_grad():
                    model = self.model_manager.get_model()
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Apply softmax to get probabilities
                    probabilities = F.softmax(logits, dim=-1)
                    confidence_scores = probabilities.cpu().numpy()
                    
                    # Get predicted classes
                    predicted_classes = torch.argmax(logits, dim=-1).cpu().numpy()
                    
                    # Process each prediction in the batch
                    for j, (text, pred_class, conf_scores) in enumerate(
                        zip(batch_texts, predicted_classes, confidence_scores)
                    ):
                        decision = self.labels.get(pred_class, "Unknown")
                        confidence = float(conf_scores[pred_class])
                        
                        # Generate reasoning if requested
                        reasoning = None
                        if return_reasoning:
                            reasoning = self._generate_reasoning(text, decision, confidence, conf_scores)
                        
                        # Format output
                        result = format_decision_output(
                            decision=decision,
                            confidence=confidence,
                            reasoning=reasoning,
                            input_text=text
                        )
                        
                        results.append(result)
            
            self.logger.info(f"Batch prediction completed for {len(texts)} texts")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            raise
    
    def _generate_reasoning(
        self, 
        text: str, 
        decision: str, 
        confidence: float, 
        confidence_scores: np.ndarray
    ) -> str:
        """
        Generate reasoning explanation for the decision.
        
        Args:
            text: Input text
            decision: Predicted decision
            confidence: Confidence score
            confidence_scores: All confidence scores
            
        Returns:
            Reasoning explanation string
        """
        try:
            # Basic reasoning based on confidence and text analysis
            reasoning_parts = []
            
            # Confidence-based reasoning
            if confidence > 0.9:
                reasoning_parts.append("Very high confidence in the decision")
            elif confidence > 0.7:
                reasoning_parts.append("High confidence in the decision")
            elif confidence > 0.5:
                reasoning_parts.append("Moderate confidence in the decision")
            else:
                reasoning_parts.append("Low confidence in the decision")
            
            # Decision margin analysis
            sorted_scores = np.sort(confidence_scores)[::-1]
            if len(sorted_scores) >= 2:
                margin = sorted_scores[0] - sorted_scores[1]
                if margin > 0.5:
                    reasoning_parts.append("clear distinction between options")
                elif margin > 0.2:
                    reasoning_parts.append("moderate distinction between options")
                else:
                    reasoning_parts.append("close decision between options")
            
            # Text-based reasoning (simple heuristics)
            text_lower = text.lower()
            
            # Positive indicators
            positive_words = ['yes', 'true', 'correct', 'right', 'good', 'positive', 'agree']
            negative_words = ['no', 'false', 'wrong', 'incorrect', 'bad', 'negative', 'disagree']
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                reasoning_parts.append("positive language indicators present")
            elif neg_count > pos_count:
                reasoning_parts.append("negative language indicators present")
            
            # Question vs statement
            if text.strip().endswith('?'):
                reasoning_parts.append("input is a question")
            else:
                reasoning_parts.append("input is a statement")
            
            # Combine reasoning parts
            reasoning = f"Decision '{decision}' based on: " + ", ".join(reasoning_parts)
            
            return reasoning
            
        except Exception as e:
            self.logger.warning(f"Failed to generate reasoning: {e}")
            return f"Decision '{decision}' with {confidence:.2%} confidence"
    
    def get_confidence_threshold(self) -> float:
        """Get the current confidence threshold."""
        return self.confidence_threshold
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold.
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = threshold
        self.logger.info(f"Confidence threshold set to {threshold}")
    
    def is_confident(self, confidence: float) -> bool:
        """
        Check if a confidence score meets the threshold.
        
        Args:
            confidence: Confidence score to check
            
        Returns:
            True if confidence meets threshold
        """
        return confidence >= self.confidence_threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self._initialized:
            return {"status": "not_initialized"}
        
        model_info = self.model_manager.get_model_info()
        model_info.update({
            "confidence_threshold": self.confidence_threshold,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "labels": self.labels
        })
        
        return model_info
    
    def warm_up(self, sample_text: str = "This is a test.") -> None:
        """
        Warm up the model with a sample prediction.
        
        Args:
            sample_text: Sample text for warm-up
        """
        if not self._initialized:
            self.initialize()
        
        try:
            self.logger.info("Warming up model...")
            self.predict(sample_text, return_reasoning=False)
            self.logger.info("Model warm-up completed")
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'model_manager'):
            self.model_manager.unload_model()
        
        self._initialized = False
        self.logger.info("Inference engine cleaned up")