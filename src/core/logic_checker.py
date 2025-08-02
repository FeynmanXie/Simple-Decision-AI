"""
Logic checker for Simple Decision AI.

This module provides logical reasoning and error detection capabilities.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LogicChecker:
    """Engine for logical reasoning and error detection."""
    
    def __init__(self):
        # Common logical fallacies and errors
        self.logical_fallacies = {
            # Self-contradictions
            "self_contradiction": [
                r"(?i).*can.*not.*can.*",  # "can do X but cannot do X"
                r"(?i).*always.*never.*",  # "always does X but never does X"
                r"(?i).*all.*none.*",      # "all X are Y but none of X are Y"
            ],
            
            # Impossible statements
            "impossible": [
                r"(?i).*square.*circle.*",           # square circle
                r"(?i).*married.*bachelor.*",        # married bachelor
                r"(?i).*silent.*noise.*",            # silent noise
                r"(?i).*freezing.*hot.*",            # freezing hot
                r"(?i).*transparent.*opaque.*",      # transparent and opaque
            ],
            
            # Common misconceptions
            "misconceptions": [
                r"(?i).*heavier.*objects.*fall.*faster.*",  # Galileo's discovery
                r"(?i).*lightning.*never.*strikes.*same.*place.*twice.*",  # False
                r"(?i).*goldfish.*memory.*3.*seconds.*",     # False
                r"(?i).*great.*wall.*china.*visible.*space.*",  # False
            ],
            
            # Mathematical impossibilities
            "math_impossible": [
                r"(?i).*divide.*by.*zero.*",         # Division by zero
                r"(?i).*square.*root.*negative.*",   # Square root of negative (in real numbers)
                r"(?i).*largest.*prime.*number.*",   # No largest prime
            ]
        }
        
        # Logical reasoning patterns
        self.logic_patterns = {
            # Valid syllogisms
            "valid_syllogism": [
                r"(?i).*all\s+(\w+)\s+are\s+(\w+).*(\w+)\s+is\s+a\s+\1.*therefore.*\3\s+is\s+a\s+\2.*",
            ],
            
            # Transitive relations
            "transitive": [
                r"(?i).*(\w+)\s*(>|<|=)\s*(\w+).*\3\s*\2\s*(\w+).*therefore.*\1\s*\2\s*\4.*",
            ],
            
            # Cause and effect
            "causation": [
                r"(?i).*if\s+.*then\s+.*",
                r"(?i).*because\s+.*therefore\s+.*",
            ]
        }
    
    def check_logic(self, text: str) -> Optional[Tuple[str, float, str]]:
        """
        Check the logical validity of a statement.
        
        Args:
            text: Statement to check
            
        Returns:
            Tuple of (decision, confidence, reasoning) or None if no logical issues found
        """
        text_lower = text.lower()
        
        try:
            # Check for logical fallacies
            result = self._check_fallacies(text_lower)
            if result is not None:
                return result
            
            # Check for impossible statements
            result = self._check_impossibilities(text_lower)
            if result is not None:
                return result
            
            # Check for valid logical patterns
            result = self._check_valid_patterns(text_lower)
            if result is not None:
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in logic checking: {e}")
            return None
    
    def _check_fallacies(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Check for logical fallacies."""
        
        for fallacy_type, patterns in self.logical_fallacies.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    decision = "No"
                    confidence = 0.9
                    reasoning = f"Logical error detected: {fallacy_type.replace('_', ' ')}. This statement contains a logical contradiction."
                    return decision, confidence, reasoning
        
        return None
    
    def _check_impossibilities(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Check for impossible or contradictory statements."""
        
        # Check for mathematical impossibilities
        for pattern in self.logical_fallacies["math_impossible"]:
            if re.search(pattern, text):
                decision = "No"
                confidence = 0.95
                reasoning = "Mathematical impossibility detected. This operation is undefined or impossible."
                return decision, confidence, reasoning
        
        # Check for physical impossibilities
        impossible_physics = [
            r"(?i).*travel.*faster.*light.*",
            r"(?i).*create.*energy.*nothing.*",
            r"(?i).*perpetual.*motion.*machine.*",
        ]
        
        for pattern in impossible_physics:
            if re.search(pattern, text):
                decision = "No"
                confidence = 0.85
                reasoning = "Physical impossibility detected. This violates known laws of physics."
                return decision, confidence, reasoning
        
        return None
    
    def _check_valid_patterns(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Check for valid logical reasoning patterns."""
        
        # Check for valid syllogisms
        for pattern in self.logic_patterns["valid_syllogism"]:
            if re.search(pattern, text):
                decision = "Yes"
                confidence = 0.9
                reasoning = "Valid logical reasoning pattern detected (syllogism)."
                return decision, confidence, reasoning
        
        # Check for transitive reasoning
        transitive_patterns = [
            r"(?i).*if\s+a\s*>\s*b.*b\s*>\s*c.*then\s+a\s*>\s*c.*",
            r"(?i).*if\s+a\s*=\s*b.*b\s*=\s*c.*then\s+a\s*=\s*c.*",
        ]
        
        for pattern in transitive_patterns:
            if re.search(pattern, text):
                decision = "Yes"
                confidence = 0.9
                reasoning = "Valid transitive reasoning pattern detected."
                return decision, confidence, reasoning
        
        return None
    
    def can_check(self, text: str) -> bool:
        """Check if this logic checker can analyze the given text."""
        text_lower = text.lower()
        
        # Check if any fallacy patterns might match
        for fallacy_type, patterns in self.logical_fallacies.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return True
        
        # Logic indicators
        logic_indicators = [
            # Logical connectives
            r'\b(if|then|therefore|because|since|thus|hence)\b',
            
            # Quantifiers
            r'\b(all|some|none|every|any|always|never)\b',
            
            # Comparisons and relations
            r'\b(greater|less|equal|same|different|opposite)\b',
            
            # Contradictions
            r'\b(but|however|although|despite|contradicts?)\b',
            
            # Impossibility indicators
            r'\b(impossible|cannot|never|always|must)\b',
            
            # Possibility questions
            r'\b(possible|possibility|can.*create|can.*make)\b',
            
            # Geometric impossibilities
            r'\b(square.*circle|circle.*square)\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in logic_indicators)
    
    def detect_question_type(self, text: str) -> str:
        """Detect the type of logical question being asked."""
        text_lower = text.lower()
        
        if re.search(r'\b(if.*then|therefore|thus|hence)\b', text_lower):
            return "conditional_reasoning"
        elif re.search(r'\b(all|every.*are|none.*are)\b', text_lower):
            return "universal_statement"
        elif re.search(r'\b(some.*are|exists?)\b', text_lower):
            return "existential_statement"
        elif re.search(r'\b(possible|impossible|can|cannot)\b', text_lower):
            return "possibility_statement"
        elif re.search(r'\b(always|never|sometimes)\b', text_lower):
            return "frequency_statement"
        else:
            return "general_logic"