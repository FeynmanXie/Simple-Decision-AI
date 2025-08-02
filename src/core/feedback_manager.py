"""
Feedback Manager for Simple Decision AI.

This module is responsible for handling and storing user feedback
to improve the model over time.
"""
import json
import os
from datetime import datetime
from pathlib import Path

from utils.helpers import ensure_dir

class FeedbackManager:
    """Handles the collection and storage of user feedback."""

    def __init__(self, feedback_dir: str = 'data/feedback'):
        """
        Initializes the FeedbackManager.

        Args:
            feedback_dir (str): The directory to store feedback files.
        """
        self.feedback_path = Path(feedback_dir) / 'feedback.jsonl'
        ensure_dir(self.feedback_path.parent)

    def save_feedback(self, original_question: str, ai_response: dict, expected_answer: str, user_reason: str) -> None:
        """
        Saves a piece of feedback to the feedback file.

        The feedback is stored in JSON Lines format, where each line is a
        JSON object representing a single feedback item.

        Args:
            original_question (str): The question the user asked.
            ai_response (dict): The AI's original response.
            expected_answer (str): The correct answer (e.g., 'Yes', 'No').
            user_reason (str): The user's explanation for why the answer is correct.
        """
        feedback_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": original_question,
            "ai_decision": ai_response.get('decision', 'N/A'),
            "ai_confidence": ai_response.get('confidence', 0.0),
            "ai_reasoning": ai_response.get('reasoning', ''),
            "expected_decision": expected_answer,
            "user_reasoning": user_reason,
            "source": "debug_mode_feedback"
        }

        try:
            with open(self.feedback_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')
        except IOError as e:
            print(f"Error: Could not write to feedback file at {self.feedback_path}. Reason: {e}")
            # Optionally, re-raise or handle more gracefully
            raise

    def get_all_feedback(self) -> list:
        """
        Retrieves all feedback entries from the storage file.

        Returns:
            list: A list of feedback dictionaries. Returns an empty list if the
                  file doesn't exist or is empty.
        """
        if not self.feedback_path.exists():
            return []

        feedback_list = []
        with open(self.feedback_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    feedback_list.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip corrupted lines
                    continue
        return feedback_list
