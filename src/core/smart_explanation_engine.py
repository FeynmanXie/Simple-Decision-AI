"""
Smart explanation engine that learns from training data.

This module provides a more intelligent explanation generation system
that can be trained on example data to produce context-aware explanations.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExplanationTemplate:
    """Template for generating explanations."""
    pattern: str
    decision_type: str  # "yes" or "no" 
    explanation: str
    confidence_threshold: float = 0.0


class SmartExplanationEngine:
    """
    Smart explanation engine that learns patterns from training data.
    
    This engine analyzes training examples to create intelligent explanation
    templates that can generate context-aware explanations.
    """
    
    def __init__(self):
        self.templates: List[ExplanationTemplate] = []
        self.fallback_explanations = {
            "yes": {
                "high": "I chose 'Yes' because the evidence strongly supports this conclusion.",
                "medium": "I chose 'Yes' because the available information indicates this is likely correct.",
                "low": "I chose 'Yes' but with low confidence due to limited evidence."
            },
            "no": {
                "high": "I chose 'No' because the evidence strongly contradicts this statement.",
                "medium": "I chose 'No' because the available information suggests this is likely incorrect.",
                "low": "I chose 'No' but with low confidence due to limited evidence."
            }
        }
        
    def train_from_data(self, training_data_path: str) -> None:
        """
        Train the explanation engine from training data.
        
        Args:
            training_data_path: Path to training data JSON file
        """
        try:
            with open(training_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Training explanation engine on {len(data)} examples")
            
            # Extract patterns from training data
            self._extract_patterns(data)
            
            logger.info(f"Extracted {len(self.templates)} explanation templates")
            
        except Exception as e:
            logger.error(f"Failed to train explanation engine: {e}")
            raise
    
    def _extract_patterns(self, data: List[Dict[str, str]]) -> None:
        """Extract explanation patterns from training data."""
        
        # Group by decision type
        yes_examples = [ex for ex in data if ex['decision'].lower() == 'yes']
        no_examples = [ex for ex in data if ex['decision'].lower() == 'no']
        
        # Extract patterns for different question types
        self._extract_patterns_for_decision_type(yes_examples, "yes")
        self._extract_patterns_for_decision_type(no_examples, "no")
    
    def _extract_patterns_for_decision_type(self, examples: List[Dict[str, str]], decision_type: str) -> None:
        """Extract patterns for a specific decision type."""
        
        # Define pattern categories and their detection rules
        pattern_categories = [
            # Personal information questions
            {
                "patterns": [
                    r'\b(do you know|tell me|what.?is).*(my|your).*(name|age|location|address)',
                    r'\b(who am i|what.?my name)',
                    r'\b(remember|know).*(about me|who i am)'
                ],
                "keywords": ["personal", "name", "know", "remember", "my", "your"]
            },
            
            # Capability questions
            {
                "patterns": [
                    r'\b(can you|are you able|do you).*(help|assist|do|make|create|write)',
                    r'\b(help me|assist me)',
                    r'\bcan.*(you|ai|assistant)'
                ],
                "keywords": ["can", "help", "assist", "able", "write", "create"]
            },
            
            # Time/prediction questions
            {
                "patterns": [
                    r'\b(will|tomorrow|next|future|predict|forecast)',
                    r'\b(what time|when)',
                    r'\b(weather|rain|snow|temperature)'
                ],
                "keywords": ["will", "tomorrow", "future", "time", "weather", "when"]
            },
            
            # Scientific facts
            {
                "patterns": [
                    r'\b(sky|blue|earth|round|sun|hot|water|wet|fire|cold|ice)',
                    r'\b(scientific|fact|physics|chemistry|biology)',
                    r'\b(gravity|energy|light|heat)'
                ],
                "keywords": ["sky", "earth", "sun", "water", "fire", "scientific", "fact"]
            },
            
            # Mathematical questions
            {
                "patterns": [
                    r'\d+\s*[\+\-\*\/]\s*\d+',
                    r'\b(plus|minus|times|divided|equals)',
                    r'\b(math|mathematics|calculation|calculate)'
                ],
                "keywords": ["plus", "minus", "equals", "math", "calculate"]
            },
            
            # AI/Technology questions
            {
                "patterns": [
                    r'\b(are you|you are).*(ai|robot|artificial|intelligence)',
                    r'\b(artificial intelligence|machine learning)',
                    r'\b(computer|technology|digital)'
                ],
                "keywords": ["ai", "artificial", "intelligence", "robot", "computer"]
            }
        ]
        
        # Process each example and try to match patterns
        for example in examples:
            input_text = example['input'].lower()
            explanation = example['explanation']
            
            # Try to match against pattern categories
            matched = False
            for category in pattern_categories:
                # Check if any pattern matches
                pattern_match = any(re.search(pattern, input_text) for pattern in category['patterns'])
                
                # Check if keywords are present
                keyword_match = any(keyword in input_text for keyword in category['keywords'])
                
                if pattern_match or keyword_match:
                    # Create a template based on the most specific pattern
                    for pattern in category['patterns']:
                        if re.search(pattern, input_text):
                            template = ExplanationTemplate(
                                pattern=pattern,
                                decision_type=decision_type,
                                explanation=explanation,
                                confidence_threshold=0.5
                            )
                            self.templates.append(template)
                            matched = True
                            break
                
                if matched:
                    break
            
            # If no specific pattern matched, create a general template based on key words
            if not matched:
                # Extract key words and create a simple pattern
                key_words = self._extract_key_words(input_text)
                if key_words:
                    pattern = r'\b(' + '|'.join(re.escape(word) for word in key_words) + r')\b'
                    template = ExplanationTemplate(
                        pattern=pattern,
                        decision_type=decision_type,
                        explanation=explanation,
                        confidence_threshold=0.3
                    )
                    self.templates.append(template)
    
    def _extract_key_words(self, text: str) -> List[str]:
        """Extract key words from text."""
        # Simple keyword extraction (remove common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'hers', 'its', 'our', 'their'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        key_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return most distinctive words (limit to 3)
        return key_words[:3]
    
    def generate_explanation(
        self, 
        input_text: str, 
        decision: str, 
        confidence: float
    ) -> str:
        """
        Generate an explanation for the given input and decision.
        
        Args:
            input_text: The input question/statement
            decision: The decision made (Yes/No)
            confidence: Confidence score (0.0 to 1.0)
            
        Returns:
            Generated explanation
        """
        decision_lower = decision.lower()
        input_lower = input_text.lower()
        
        # Try to find a matching template
        best_template = None
        best_score = 0
        
        for template in self.templates:
            if template.decision_type == decision_lower:
                # Check if pattern matches
                try:
                    if re.search(template.pattern, input_lower):
                        # Score based on pattern specificity and confidence threshold
                        pattern_specificity = len(template.pattern) / 100  # Rough measure
                        confidence_match = 1.0 if confidence >= template.confidence_threshold else 0.5
                        score = pattern_specificity * confidence_match
                        
                        if score > best_score:
                            best_score = score
                            best_template = template
                except re.error:
                    # Skip templates with invalid regex
                    continue
        
        # Use best matching template if found
        if best_template and best_score > 0.1:
            return best_template.explanation
        
        # Fall back to confidence-based explanations
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
        
        if decision_lower in self.fallback_explanations:
            return self.fallback_explanations[decision_lower][confidence_level]
        
        # Final fallback
        return f"I chose '{decision}' based on my analysis of the available information."
    
    def save_templates(self, file_path: str) -> None:
        """Save extracted templates to a file."""
        templates_data = []
        for template in self.templates:
            templates_data.append({
                'pattern': template.pattern,
                'decision_type': template.decision_type,
                'explanation': template.explanation,
                'confidence_threshold': template.confidence_threshold
            })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(templates_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(templates_data)} templates to {file_path}")
    
    def load_templates(self, file_path: str) -> None:
        """Load templates from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            
            self.templates = []
            for data in templates_data:
                template = ExplanationTemplate(
                    pattern=data['pattern'],
                    decision_type=data['decision_type'],
                    explanation=data['explanation'],
                    confidence_threshold=data.get('confidence_threshold', 0.5)
                )
                self.templates.append(template)
            
            logger.info(f"Loaded {len(self.templates)} templates from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
            # Continue with empty templates - will use fallback explanations