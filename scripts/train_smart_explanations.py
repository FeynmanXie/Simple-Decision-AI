"""
Train the smart explanation engine from training data.

This script trains the explanation engine to generate context-aware explanations
by learning patterns from example data.
"""

import sys
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from core.smart_explanation_engine import SmartExplanationEngine
from utils.logger import setup_logger
import logging

logger = logging.getLogger(__name__)


def main():
    """Train the smart explanation engine."""
    setup_logger("train_smart_explanations", "INFO")
    
    logger.info("Starting smart explanation engine training...")
    
    # Create and train the explanation engine
    engine = SmartExplanationEngine()
    
    # Train from data
    training_data_path = "./data/training/enhanced_math_training.json"
    engine.train_from_data(training_data_path)
    
    # Save the trained templates
    templates_path = "./models/explanation_templates.json"
    engine.save_templates(templates_path)
    
    logger.info("Training completed!")
    
    # Test the engine with some examples
    test_examples = [
        ("Do you know my name?", "No", 0.6),
        ("Can you help me write code?", "Yes", 0.8),
        ("Is the sky blue?", "Yes", 0.95),
        ("Will it rain tomorrow?", "No", 0.5),
        ("Are you an AI?", "Yes", 0.9)
    ]
    
    logger.info("\nTesting the trained engine:")
    for input_text, decision, confidence in test_examples:
        explanation = engine.generate_explanation(input_text, decision, confidence)
        logger.info(f"Q: {input_text}")
        logger.info(f"A: {decision} - {explanation}")
        logger.info("-" * 50)


if __name__ == "__main__":
    main()