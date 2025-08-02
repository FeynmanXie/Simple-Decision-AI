"""
Rule-based decision engine for Simple Decision AI.

This serves as a fallback/enhancement to the AI model for common cases.
"""

import re
from typing import Dict, Any, Optional, Tuple
from utils.logger import LoggerMixin
from utils.helpers import format_decision_output
from .smart_explanation_engine import SmartExplanationEngine
from .math_reasoning_engine import MathReasoningEngine
from .logic_checker import LogicChecker


class RuleEngine(LoggerMixin):
    """Rule-based decision engine for handling common cases."""
    
    def __init__(self):
        """Initialize the rule engine."""
        self.rules = self._load_rules()
        self.explanation_engine = SmartExplanationEngine()
        self.math_engine = MathReasoningEngine()
        self.logic_checker = LogicChecker()
        self._explanations_loaded = False
        self._load_explanation_templates()
        self.logger.info("Rule engine initialized")
    
    def _load_rules(self) -> Dict[str, Any]:
        """Load decision rules."""
        return {
            # Factual knowledge rules
            "factual": [
                (r"(?i).*sky.*blue.*", "Yes", 0.95, "The sky appears blue due to light scattering"),
                (r"(?i).*water.*wet.*", "Yes", 0.95, "Water is wet by definition"),
                (r"(?i).*fire.*hot.*", "Yes", 0.95, "Fire produces heat"),
                (r"(?i).*ice.*cold.*", "Yes", 0.95, "Ice is frozen water, therefore cold"),
                (r"(?i).*sun.*hot.*", "Yes", 0.95, "The sun is a star that produces heat"),
                (r"(?i).*earth.*round.*", "Yes", 0.95, "Earth is approximately spherical"),
                (r"(?i).*earth.*flat.*", "No", 0.95, "Earth is not flat, it's spherical"),
            ],
            
            # Mathematical rules
            "math": [
                (r"(?i).*2\s*\+\s*2\s*=\s*4.*", "Yes", 0.99, "Basic arithmetic: 2+2=4"),
                (r"(?i).*2\s*\+\s*2\s*=\s*5.*", "No", 0.99, "Incorrect arithmetic: 2+2≠5"),
                (r"(?i).*10\s*>\s*5.*", "Yes", 0.99, "10 is greater than 5"),
                (r"(?i).*5\s*>\s*10.*", "No", 0.99, "5 is not greater than 10"),
                (r"(?i).*3\s*\*\s*4\s*=\s*12.*", "Yes", 0.99, "Basic multiplication: 3×4=12"),
                (r"(?i).*3\s*\*\s*4\s*=\s*11.*", "No", 0.99, "Incorrect multiplication: 3×4≠11"),
                (r"(?i).*8\s*÷\s*2\s*=\s*4.*", "Yes", 0.99, "Basic division: 8÷2=4"),
                (r"(?i).*8\s*/\s*2\s*=\s*4.*", "Yes", 0.99, "Basic division: 8/2=4"),
                (r"(?i).*100\s*-\s*25\s*=\s*75.*", "Yes", 0.99, "Basic subtraction: 100-25=75"),
                (r"(?i).*7\s*\*\s*3\s*=\s*21.*", "Yes", 0.99, "Basic multiplication: 7×3=21"),
                (r"(?i).*zero.*multiply.*anything.*zero.*", "Yes", 0.95, "Anything multiplied by zero equals zero"),
                (r"(?i).*any.*number.*divided.*by.*itself.*one.*", "Yes", 0.95, "Any number divided by itself equals one"),
            ],
            
            # Logical rules
            "logic": [
                (r"(?i).*if\s+a\s*>\s*b.*b\s*>\s*c.*then\s+a\s*>\s*c.*", "Yes", 0.95, "Transitive property of inequality"),
                (r"(?i).*all\s+cats.*animals.*fluffy.*cat.*fluffy.*animal.*", "Yes", 0.95, "Valid syllogism"),
            ],
            
            # Common sense rules
            "common_sense": [
                (r"(?i).*people.*need.*water.*survive.*", "Yes", 0.95, "Humans require water for survival"),
                (r"(?i).*cats.*fly.*naturally.*", "No", 0.95, "Cats cannot fly without assistance"),
                (r"(?i).*humans.*breathe.*underwater.*without.*equipment.*", "No", 0.95, "Humans need breathing apparatus underwater"),
                (r"(?i).*ice.*warmer.*boiling.*water.*", "No", 0.95, "Ice is much colder than boiling water"),
            ],
            
            # Language understanding rules
            "language": [
                (r"(?i).*understand.*english.*", "Yes", 0.90, "This system processes English text"),
                (r"(?i).*understand.*chinese.*", "Yes", 0.85, "This system can process Chinese characters"),
                (r"(?i).*python.*programming.*language.*", "Yes", 0.95, "Python is a programming language"),
            ],
            
            # Positive/negative indicators
            "sentiment": [
                (r"(?i).*should.*say.*yes.*", "Yes", 0.80, "Positive directive detected"),
                (r"(?i).*should.*say.*no.*", "No", 0.80, "Negative directive detected"),
            ]
        }
    
    def evaluate(self, text: str) -> Optional[Tuple[str, float, str]]:
        """
        Evaluate text against rules.
        
        Args:
            text: Input text to evaluate
            
        Returns:
            Tuple of (decision, confidence, reasoning) or None if no rule matches
        """
        # First try mathematical reasoning
        if self.math_engine.can_solve(text):
            math_result = self.math_engine.solve(text)
            if math_result is not None:
                decision, confidence, reasoning = math_result
                self.logger.debug(f"Math engine solved: {decision}")
                return decision, confidence, f"Mathematical reasoning: {reasoning}"
        
        # Try logical reasoning
        if self.logic_checker.can_check(text):
            logic_result = self.logic_checker.check_logic(text)
            if logic_result is not None:
                decision, confidence, reasoning = logic_result
                self.logger.debug(f"Logic checker result: {decision}")
                return decision, confidence, f"Logical reasoning: {reasoning}"
        
        # Clean text for better matching
        cleaned_text = self._clean_text(text)
        
        # Check all rule categories
        for category, rules in self.rules.items():
            for pattern, decision, confidence, reasoning in rules:
                if re.search(pattern, cleaned_text):
                    self.logger.debug(f"Rule matched: {category} - {pattern}")
                    return decision, confidence, f"Rule-based decision: {reasoning}"
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better rule matching."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize punctuation for better matching
        text = re.sub(r'[.!?]+$', '', text)  # Remove trailing punctuation
        
        return text
    
    def _load_explanation_templates(self) -> None:
        """Load explanation templates for smart explanations."""
        try:
            templates_path = "./models/explanation_templates.json"
            self.explanation_engine.load_templates(templates_path)
            self._explanations_loaded = True
            self.logger.info("Rule engine explanation templates loaded successfully")
        except Exception as e:
            self.logger.warning(f"Could not load explanation templates in rule engine: {e}. Using fallback explanations.")
            self._explanations_loaded = False
    
    def decide_with_rules(self, text: str, ai_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a decision using rules, with optional AI result as fallback.
        
        Args:
            text: Input text
            ai_result: Optional AI model result
            
        Returns:
            Decision result dictionary
        """
        # Try rule-based decision first
        rule_result = self.evaluate(text)
        
        if rule_result:
            decision, confidence, reasoning = rule_result
            # Generate explanation using smart explanation engine
            if self._explanations_loaded:
                explanation = self.explanation_engine.generate_explanation(text, decision, confidence)
            else:
                explanation = f"I chose '{decision}' based on a rule-based pattern match."
            
            return format_decision_output(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                input_text=text,
                explanation=explanation
            )
        
        # Fall back to AI result if available
        if ai_result:
            # Enhance AI result with rule engine info
            ai_result['reasoning'] = f"AI model decision: {ai_result.get('reasoning', 'No specific reasoning')}"
            # Add explanation if not already present (should already be there from InferenceEngine)
            if 'explanation' not in ai_result:
                if self._explanations_loaded:
                    ai_result['explanation'] = self.explanation_engine.generate_explanation(
                        text, ai_result.get('decision', 'Unknown'), ai_result.get('confidence', 0.0)
                    )
                else:
                    ai_result['explanation'] = f"I chose '{ai_result.get('decision', 'Unknown')}' based on AI analysis."
            return ai_result
        
        # Last resort: admit uncertainty
        explanation = "I could not make a clear decision due to insufficient information."
        return format_decision_output(
            decision="Unknown",
            confidence=0.0,
            reasoning="No applicable rules found and no AI model result available",
            input_text=text,
            explanation=explanation
        )
    
    def get_rule_stats(self) -> Dict[str, int]:
        """Get statistics about loaded rules."""
        stats = {}
        for category, rules in self.rules.items():
            stats[category] = len(rules)
        
        total_rules = sum(stats.values())
        stats['total'] = total_rules
        
        return stats
    
    def add_custom_rule(self, category: str, pattern: str, decision: str, 
                       confidence: float, reasoning: str) -> None:
        """
        Add a custom rule.
        
        Args:
            category: Rule category
            pattern: Regex pattern to match
            decision: Decision to make (Yes/No)
            confidence: Confidence level (0.0-1.0)
            reasoning: Reasoning explanation
        """
        if category not in self.rules:
            self.rules[category] = []
        
        self.rules[category].append((pattern, decision, confidence, reasoning))
        self.logger.info(f"Added custom rule to {category}: {pattern}")


# Global rule engine instance
rule_engine = RuleEngine()