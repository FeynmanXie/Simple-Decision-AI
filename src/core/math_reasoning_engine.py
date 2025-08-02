"""
Mathematical reasoning engine for Simple Decision AI.

This module provides mathematical reasoning capabilities including
arithmetic operations, basic logic, and problem solving.
"""

import re
import operator
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class MathReasoningEngine:
    """Engine for mathematical reasoning and problem solving."""
    
    def __init__(self):
        self.operators = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            'x': operator.mul,
            '×': operator.mul,
            '/': operator.truediv,
            '÷': operator.truediv,
            '//': operator.floordiv,
            '%': operator.mod,
            '**': operator.pow,
            '^': operator.pow
        }
        
        # Number word mappings
        self.number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
            'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14', 'fifteen': '15',
            'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100'
        }
        
        # Common math patterns
        self.math_patterns = [
            # Basic arithmetic: "5 + 3 = 8"
            r'(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)\s*(?:=|equals?)\s*(\d+(?:\.\d+)?)',
            
            # Word problems: "Tom has 15 oranges, buys 10 more, gives away 5"
            r'(?:has|have|had)\s+(\d+).*?(?:buy|buys|bought|get|gets|got)\s+(\d+).*?(?:give|gives|gave|lose|loses|lost)\s+(?:away\s+)?(\d+)',
            
            # Comparison: "is 10 greater than 5"
            r'(?:is\s+)?(\d+(?:\.\d+)?)\s+(?:greater\s+than|>\s*|more\s+than)\s+(\d+(?:\.\d+)?)',
            r'(?:is\s+)?(\d+(?:\.\d+)?)\s+(?:less\s+than|<\s*|smaller\s+than)\s+(\d+(?:\.\d+)?)',
            r'(?:is\s+)?(\d+(?:\.\d+)?)\s+(?:equal\s+to|=\s*|equals?)\s+(\d+(?:\.\d+)?)',
            
            # Simple counting: "exactly 20"
            r'exactly\s+(\d+)',
            r'(?:have|has)\s+(?:exactly\s+)?(\d+)',
        ]
        
        # Common knowledge facts
        self.math_facts = {
            # Basic arithmetic facts
            "2+2": 4, "2+2=4": True, "2+2=5": False,
            "3*4": 12, "3*4=12": True, "3*4=11": False,
            "10/2": 5, "10/2=5": True, "10/2=6": False,
            
            # Comparison facts
            "10>5": True, "5>10": False,
            "5<10": True, "10<5": False,
            "7=7": True, "7=8": False,
        }
    
    def can_solve(self, text: str) -> bool:
        """Check if this engine can solve the given problem."""
        text_lower = text.lower()
        
        # Check for mathematical indicators
        math_indicators = [
            # Numbers and operations
            r'\d+\s*[+\-*/×÷=]\s*\d+',
            
            # Word problem indicators
            r'\b(has|have|had|buy|buys|bought|give|gives|gave|get|gets|got)\b.*\d+',
            
            # Comparison words
            r'\b(greater|less|more|smaller|equal|exactly|total)\b',
            
            # Math keywords
            r'\b(plus|minus|times|divided|sum|difference|product|quotient)\b',
            
            # Question about quantities
            r'\b(how many|how much|total|altogether)\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in math_indicators)
    
    def solve(self, text: str) -> Optional[Tuple[str, float, str]]:
        """
        Solve a mathematical problem.
        
        Args:
            text: Problem text
            
        Returns:
            Tuple of (decision, confidence, reasoning) or None if can't solve
        """
        text_lower = text.lower()
        
        # Convert number words to digits
        text_lower = self._convert_number_words(text_lower)
        
        try:
            # Try different solution strategies
            
            # Strategy 1: Direct arithmetic evaluation
            result = self._solve_arithmetic(text_lower)
            if result is not None:
                return result
            
            # Strategy 2: Word problem solving
            result = self._solve_word_problem(text_lower)
            if result is not None:
                return result
            
            # Strategy 3: Comparison problems
            result = self._solve_comparison(text_lower)
            if result is not None:
                return result
            
            # Strategy 4: Check against known facts
            result = self._check_math_facts(text_lower)
            if result is not None:
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Error in math reasoning: {e}")
            return None
    
    def _solve_arithmetic(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Solve basic arithmetic expressions."""
        
        # Pattern for multi-operation expressions like "15 + 10 - 5 = 20"
        multi_op_pattern = r'(\d+(?:\.\d+)?(?:\s*[+\-*/×÷]\s*\d+(?:\.\d+)?)+)\s*(?:=|equals?|equal\s+to)\s*(\d+(?:\.\d+)?)'
        match = re.search(multi_op_pattern, text)
        if match:
            expression, expected = match.groups()
            try:
                # Safely evaluate the mathematical expression
                # Clean up the expression and make it safe
                clean_expr = expression.replace('×', '*').replace('÷', '/')
                clean_expr = re.sub(r'[^0-9+\-*/.() ]', '', clean_expr)
                
                # Evaluate using eval (safe since we cleaned the input)
                actual = eval(clean_expr)
                expected = float(expected)
                
                # Handle integer results
                if actual == int(actual):
                    actual = int(actual)
                if expected == int(expected):
                    expected = int(expected)
                
                is_correct = abs(actual - expected) < 0.0001
                decision = "Yes" if is_correct else "No"
                
                reasoning = f"Calculating: {expression} = {actual}. "
                reasoning += f"The statement claims it equals {expected}, which is {'correct' if is_correct else 'incorrect'}."
                
                return decision, 0.95, reasoning
                
            except (ValueError, ZeroDivisionError, SyntaxError, TypeError):
                pass
        
        # Pattern: "2 + 2 = 4" or "is 2 + 2 equal to 4"
        patterns = [
            r'(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)\s*(?:=|equals?|equal\s+to)\s*(\d+(?:\.\d+)?)',
            r'(?:is\s+)?(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)\s+(?:equal\s+to|equals?)\s*(\d+(?:\.\d+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                num1, op, num2, expected = match.groups()
                
                try:
                    num1 = float(num1)
                    num2 = float(num2) 
                    expected = float(expected)
                    
                    # Convert operator symbols
                    if op in ['×', 'x']:
                        op = '*'
                    elif op == '÷':
                        op = '/'
                    
                    if op in self.operators:
                        actual = self.operators[op](num1, num2)
                        
                        # Handle integer results
                        if actual == int(actual):
                            actual = int(actual)
                        if expected == int(expected):
                            expected = int(expected)
                        
                        is_correct = abs(actual - expected) < 0.0001
                        decision = "Yes" if is_correct else "No"
                        
                        reasoning = f"Calculating: {num1} {op} {num2} = {actual}. "
                        reasoning += f"The statement claims it equals {expected}, which is {'correct' if is_correct else 'incorrect'}."
                        
                        return decision, 0.95, reasoning
                        
                except (ValueError, ZeroDivisionError, TypeError):
                    pass
        
        return None
    
    def _solve_word_problem(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Solve word problems involving quantities."""
        
        # Pattern 1: "Tom has 15 oranges, buys 10 more, gives away 5, does he have exactly 20?"
        if 'exactly' in text and ('have' in text or 'has' in text):
            numbers = re.findall(r'\d+', text)
            if len(numbers) >= 3:
                try:
                    numbers = [int(n) for n in numbers]
                    
                    # Common pattern: start + add - subtract = final?
                    if len(numbers) >= 4:
                        start, add_amount, subtract_amount, expected = numbers[:4]
                        
                        # Calculate actual result
                        actual = start + add_amount - subtract_amount
                        
                        # Check if it matches expected
                        is_correct = actual == expected
                        decision = "Yes" if is_correct else "No"
                        
                        reasoning = f"Starting with {start}, adding {add_amount} gives {start + add_amount}, "
                        reasoning += f"then subtracting {subtract_amount} gives {actual}. "
                        reasoning += f"The question asks if this equals {expected}, which is {'correct' if is_correct else 'incorrect'}."
                        
                        return decision, 0.9, reasoning
                        
                except ValueError:
                    pass
        
        # Pattern 2: "If Mary has 12 apples and gives 4 to John, does she have 8 apples left?"
        give_pattern = r'(\w+)\s+has\s+(\d+)\s+\w+.*?(?:gives?|gave)\s+(\d+).*?(?:have|has)\s+(\d+).*?left'
        match = re.search(give_pattern, text, re.IGNORECASE)
        if match:
            try:
                name, start_amount, given_amount, expected_left = match.groups()
                start_amount = int(start_amount)
                given_amount = int(given_amount)
                expected_left = int(expected_left)
                
                # Calculate what should be left
                actual_left = start_amount - given_amount
                
                is_correct = actual_left == expected_left
                decision = "Yes" if is_correct else "No"
                
                reasoning = f"{name} started with {start_amount}, gave away {given_amount}, "
                reasoning += f"so has {actual_left} left. "
                reasoning += f"The question asks if this equals {expected_left}, which is {'correct' if is_correct else 'incorrect'}."
                
                return decision, 0.9, reasoning
                
            except ValueError:
                pass
        
        # Pattern 3: More general subtraction word problems
        subtraction_pattern = r'(\w+)\s+has\s+(\d+).*?(?:gives?|gave|lose|loses|lost)\s+(\d+).*?(?:have|has)\s+(\d+)'
        match = re.search(subtraction_pattern, text, re.IGNORECASE)
        if match:
            try:
                name, start_amount, subtract_amount, expected = match.groups()
                start_amount = int(start_amount)
                subtract_amount = int(subtract_amount)
                expected = int(expected)
                
                actual = start_amount - subtract_amount
                
                is_correct = actual == expected
                decision = "Yes" if is_correct else "No"
                
                reasoning = f"{name} started with {start_amount}, lost {subtract_amount}, "
                reasoning += f"so has {actual} remaining. "
                reasoning += f"The question asks if this equals {expected}, which is {'correct' if is_correct else 'incorrect'}."
                
                return decision, 0.9, reasoning
                
            except ValueError:
                pass
        
        return None
    
    def _solve_comparison(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Solve comparison problems."""
        
        comparison_patterns = [
            (r'(\d+(?:\.\d+)?)\s+(?:greater\s+than|>\s*|more\s+than)\s+(\d+(?:\.\d+)?)', 'greater'),
            (r'(\d+(?:\.\d+)?)\s+(?:less\s+than|<\s*|smaller\s+than)\s+(\d+(?:\.\d+)?)', 'less'),
            (r'(\d+(?:\.\d+)?)\s+(?:equal\s+to|=\s*|equals?)\s+(\d+(?:\.\d+)?)', 'equal')
        ]
        
        for pattern, comparison_type in comparison_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    num1 = float(match.group(1))
                    num2 = float(match.group(2))
                    
                    if comparison_type == 'greater':
                        result = num1 > num2
                        description = f"{num1} is {'greater' if result else 'not greater'} than {num2}"
                    elif comparison_type == 'less':
                        result = num1 < num2
                        description = f"{num1} is {'less' if result else 'not less'} than {num2}"
                    else:  # equal
                        result = abs(num1 - num2) < 0.0001
                        description = f"{num1} {'equals' if result else 'does not equal'} {num2}"
                    
                    decision = "Yes" if result else "No"
                    reasoning = f"Comparing numbers: {description}."
                    
                    return decision, 0.95, reasoning
                    
                except ValueError:
                    pass
        
        return None
    
    def _check_math_facts(self, text: str) -> Optional[Tuple[str, float, str]]:
        """Check against known mathematical facts."""
        
        # Normalize text for fact checking
        normalized = re.sub(r'\s+', '', text)
        normalized = normalized.replace('equals', '=').replace('equal', '=')
        
        for fact, value in self.math_facts.items():
            if fact in normalized:
                if isinstance(value, bool):
                    decision = "Yes" if value else "No"
                    reasoning = f"This is a known mathematical fact: {fact} is {'true' if value else 'false'}."
                    return decision, 0.95, reasoning
                else:
                    # It's a calculation fact
                    if str(value) in text:
                        decision = "Yes"
                        reasoning = f"Mathematical calculation: {fact} = {value}, which is correct."
                        return decision, 0.95, reasoning
        
        return None
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract all numbers from text."""
        pattern = r'-?\d+(?:\.\d+)?'
        numbers = re.findall(pattern, text)
        return [float(n) for n in numbers]
    
    def has_math_keywords(self, text: str) -> bool:
        """Check if text contains mathematical keywords."""
        keywords = [
            'plus', 'minus', 'times', 'divided', 'equals', 'equal',
            'greater', 'less', 'more', 'fewer', 'total', 'sum',
            'difference', 'product', 'quotient', 'exactly', 'approximately'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)
    
    def _convert_number_words(self, text: str) -> str:
        """Convert number words to digits in text."""
        result = text
        
        # Convert operation words to symbols
        operation_words = {
            'plus': '+', 'add': '+', 'added to': '+', 'and': '+',
            'minus': '-', 'subtract': '-', 'take away': '-', 'less': '-',
            'times': '*', 'multiply': '*', 'multiplied by': '*', 'of': '*',
            'divided by': '/', 'divide': '/', 'split by': '/',
            'equals': '=', 'equal to': '=', 'is': '=', 'makes': '=',
            'percent': '%', 'percentage': '%'
        }
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_ops = sorted(operation_words.items(), key=lambda x: len(x[0]), reverse=True)
        for word, symbol in sorted_ops:
            pattern = r'\b' + re.escape(word) + r'\b'
            result = re.sub(pattern, symbol, result, flags=re.IGNORECASE)
        
        # Convert number words to digits
        sorted_words = sorted(self.number_words.items(), key=lambda x: len(x[0]), reverse=True)
        for word, digit in sorted_words:
            # Use word boundary to avoid partial matches
            pattern = r'\b' + re.escape(word) + r'\b'
            result = re.sub(pattern, digit, result, flags=re.IGNORECASE)
        
        return result