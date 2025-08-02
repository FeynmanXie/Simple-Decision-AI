"""
Simple Decision AI - A binary decision making AI system.

This package provides a simple AI system that can make binary (Yes/No) decisions
based on natural language input.
"""

__version__ = "0.1.0"
__author__ = "AI Developer"
__email__ = "developer@example.com"

from .core.decision_maker import DecisionMaker
from .core.inference_engine import InferenceEngine

__all__ = ["DecisionMaker", "InferenceEngine"]