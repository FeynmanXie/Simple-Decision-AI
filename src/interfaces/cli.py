"""
Command Line Interface for Simple Decision AI.

This module provides a CLI for interacting with the decision AI system.
"""

import sys
import json
import click
import warnings
import logging
from typing import Optional
from pathlib import Path

# Suppress warnings for a cleaner user experience
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress noisy library logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from core.decision_maker import DecisionMaker
from core.feedback_manager import FeedbackManager
from utils.config_loader import config_loader
from utils.logger import setup_logger
from utils.helpers import ensure_dir


class CLI:
    """A wrapper class for CLI functionalities."""
    
    def __init__(self):
        self.decision_maker: Optional[DecisionMaker] = None
        self.logger = setup_logger("CLI", log_level="ERROR", console_output=False)
        
    def initialize_decision_maker(self) -> None:
        """Initializes the core DecisionMaker component."""
        if self.decision_maker is None:
            try:
                self.decision_maker = DecisionMaker()
                self.decision_maker.initialize()
                self.logger.info("Decision maker initialized successfully.")
            except Exception as e:
                self.logger.error(f"Fatal: Failed to initialize decision maker: {e}")
                raise

def _format_decision_output(ctx, result: dict, no_reasoning: bool, format: str):
    """Helper function to format and print the decision result."""
    if format == 'json':
        click.echo(json.dumps(result, indent=2))
    else:
        decision = result['decision']
        confidence = result['confidence']
        
        # Color coding for decision
        decision_color = 'green' if decision.lower() == 'yes' else 'red'
        decision_colored = click.style(decision, fg=decision_color, bold=True)
        
        # Color coding for confidence
        if confidence > 0.8: conf_color = 'green'
        elif confidence > 0.6: conf_color = 'yellow'
        else: conf_color = 'red'
        confidence_colored = click.style(f"{confidence:.2%}", fg=conf_color)
        
        click.echo(f"Decision: {decision_colored}")
        click.echo(f"Confidence: {confidence_colored}")
        
        # Show explanation first (simple English explanation)
        if 'explanation' in result:
            click.echo(f"Explanation: {click.style(result['explanation'], fg='cyan')}")
        
        if not no_reasoning and 'reasoning' in result:
            click.echo(f"Reasoning: {result['reasoning']}")
        
        if ctx.obj.get('verbose'):
            click.echo(f"Timestamp: {result.get('timestamp', 'N/A')}")

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output.')
@click.option('--config', '-c', default='config', help='Path to configuration directory.')
@click.pass_context
def main(ctx, verbose: bool, config: str):
    """Simple Decision AI - A binary decision-making system."""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config_path'] = config
    log_level = "DEBUG" if verbose else "INFO"
    setup_logger("CLI", log_level=log_level)

@main.command()
@click.argument('text', required=True)
@click.option('--no-reasoning', is_flag=True, help='Disable reasoning output.')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='text', help='Output format.')
@click.option('--confidence-threshold', '-t', type=float, help='Confidence threshold (0.0-1.0).')
@click.pass_context
def decide(ctx, text: str, no_reasoning: bool, format: str, confidence_threshold: Optional[float]):
    """Make a binary decision for a given text."""
    try:
        cli = CLI()
        cli.initialize_decision_maker()
        
        if confidence_threshold is not None:
            cli.decision_maker.set_confidence_threshold(confidence_threshold)
        
        result = cli.decision_maker.decide(text, include_reasoning=not no_reasoning, save_to_history=True)
        
        _format_decision_output(ctx, result, no_reasoning, format)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

@main.command()
@click.argument('question')
@click.argument('expected_answer', type=click.Choice(['Yes', 'No'], case_sensitive=False))
@click.argument('reason')
@click.pass_context
def feedback(ctx, question: str, expected_answer: str, reason: str):
    """Provide feedback to improve the AI's future decisions."""
    try:
        cli = CLI()
        cli.initialize_decision_maker()
        
        # Get the AI's original response first
        ai_response = cli.decision_maker.decide(question, include_reasoning=True, save_to_history=False)
        
        click.echo("\n--- AI's Original Answer ---")
        _format_decision_output(ctx, ai_response, no_reasoning=False, format='text')
        
        # Save the feedback
        feedback_manager = FeedbackManager()
        feedback_manager.save_feedback(
            original_question=question,
            ai_response=ai_response,
            expected_answer=expected_answer.capitalize(),
            user_reason=reason
        )
        
        click.secho("\nThank you! Your feedback has been recorded.", fg="cyan", bold=True)
        click.echo("This will be used to improve the model in future training cycles.")
        
    except Exception as e:
        click.echo(f"Error recording feedback: {e}", err=True)
        sys.exit(1)

# ... (other commands like batch, stats, history remain the same) ...
@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--no-reasoning', is_flag=True, help='Disable reasoning output')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json', 
              help='Output format')
@click.pass_context
def batch(ctx, input_file: str, output: Optional[str], no_reasoning: bool, format: str):
    """Process a batch of texts from a file."""
    try:
        cli = CLI()
        cli.initialize_decision_maker()
        
        input_path = Path(input_file)
        
        if input_path.suffix.lower() == '.json':
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = data
                elif isinstance(data, dict) and 'texts' in data:
                    texts = data['texts']
                else:
                    raise ValueError("JSON file must contain a list of texts or a dict with 'texts' key")
        else:
            with open(input_path, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        click.echo(f"Processing {len(texts)} texts...")
        
        results = cli.decision_maker.decide_batch(
            texts,
            include_reasoning=not no_reasoning,
            save_to_history=True
        )
        
        if output:
            output_path = Path(output)
            ensure_dir(output_path.parent)
            
            if format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            elif format == 'csv':
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Input', 'Decision', 'Confidence', 'Explanation', 'Reasoning', 'Timestamp'])
                    for result in results:
                        writer.writerow([
                            result.get('input', ''),
                            result.get('decision', ''),
                            result.get('confidence', 0),
                            result.get('explanation', ''),
                            result.get('reasoning', ''),
                            result.get('timestamp', '')
                        ])
            
            click.echo(f"Results saved to {output_path}")
        else:
            for i, result in enumerate(results, 1):
                click.echo(f"\n--- Result {i} ---")
                click.echo(f"Input: {result.get('input', '')}")
                click.echo(f"Decision: {result['decision']}")
                click.echo(f"Confidence: {result['confidence']:.2%}")
                if 'explanation' in result:
                    click.echo(f"Explanation: {result['explanation']}")
                if not no_reasoning and 'reasoning' in result:
                    click.echo(f"Reasoning: {result['reasoning']}")
        
        yes_count = sum(1 for r in results if r['decision'].lower() == 'yes')
        no_count = len(results) - yes_count
        avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
        
        click.echo(f"\n--- Summary ---")
        click.echo(f"Total: {len(results)}")
        click.echo(f"Yes: {yes_count} ({yes_count/len(results):.1%})")
        click.echo(f"No: {no_count} ({no_count/len(results):.1%})")
        click.echo(f"Average confidence: {avg_confidence:.2%}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def stats(ctx):
    """Show decision statistics."""
    try:
        cli = CLI()
        cli.initialize_decision_maker()
        
        statistics = cli.decision_maker.get_statistics()
        
        click.echo("=== Decision Statistics ===")
        click.echo(f"Total decisions: {statistics['total_decisions']}")
        
        if statistics['total_decisions'] > 0:
            click.echo(f"Yes decisions: {statistics['yes_decisions']} ({statistics['yes_percentage']:.1f}%)")
            click.echo(f"No decisions: {statistics['no_decisions']} ({statistics['no_percentage']:.1f}%)")
            click.echo(f"High confidence: {statistics['high_confidence_decisions']} ({statistics['high_confidence_percentage']:.1f}%)")
            click.echo(f"Low confidence: {statistics['low_confidence_decisions']} ({statistics['low_confidence_percentage']:.1f}%)")
        
        model_info = cli.decision_maker.get_model_info()
        click.echo(f"\n=== Model Information ===")
        click.echo(f"Status: {model_info.get('status', 'Unknown')}")
        click.echo(f"Model: {model_info.get('model_name', 'Unknown')}")
        click.echo(f"Device: {model_info.get('device', 'Unknown')}")
        click.echo(f"Confidence threshold: {model_info.get('confidence_threshold', 0):.2f}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--limit', '-l', type=int, default=10, help='Number of recent decisions to show')
@click.option('--format', '-f', type=click.Choice(['json', 'text']), default='text', 
              help='Output format')
@click.pass_context
def history(ctx, limit: int, format: str):
    """Show recent decision history."""
    try:
        cli = CLI()
        cli.initialize_decision_maker()
        
        history_items = cli.decision_maker.get_history(limit)
        
        if not history_items:
            click.echo("No decision history available.")
            return
        
        if format == 'json':
            click.echo(json.dumps(history_items, indent=2))
        else:
            click.echo(f"=== Recent {len(history_items)} Decisions ===")
            for i, item in enumerate(reversed(history_items), 1):
                click.echo(f"\n{i}. Input: {item.get('input', 'N/A')}")
                click.echo(f"   Decision: {item['decision']}")
                click.echo(f"   Confidence: {item['confidence']:.2%}")
                click.echo(f"   Time: {item.get('timestamp', 'N/A')}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.confirmation_option(prompt='Are you sure you want to clear all history?')
@click.pass_context
def clear_history(ctx):
    """Clear decision history."""
    try:
        cli = CLI()
        cli.initialize_decision_maker()
        
        cli.decision_maker.clear_history()
        cli.decision_maker.reset_statistics()
        
        click.echo("History and statistics cleared.")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--threshold', '-t', type=float, required=True, 
              help='New confidence threshold (0.0-1.0)')
@click.pass_context
def set_threshold(ctx, threshold: float):
    """Set confidence threshold."""
    try:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        cli = CLI()
        cli.initialize_decision_maker()
        
        cli.decision_maker.set_confidence_threshold(threshold)
        
        click.echo(f"Confidence threshold set to {threshold:.2f}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def interactive(ctx):
    """Start interactive mode."""
    try:
        cli = CLI()
        cli.initialize_decision_maker()
        
        click.echo("=== Simple Decision AI - Interactive Mode ===")
        click.echo("Type 'quit' or 'exit' to stop, 'help' for commands")
        click.echo()
        
        while True:
            try:
                text = click.prompt("Enter text for decision", type=str)
                
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                elif text.lower() == 'help':
                    click.echo("Commands: help, stats, threshold <value>, quit/exit/q")
                    continue
                elif text.lower() == 'stats':
                    stats_info = cli.decision_maker.get_statistics()
                    click.echo(f"Total: {stats_info['total_decisions']}, Yes: {stats_info['yes_decisions']}, No: {stats_info['no_decisions']}")
                    continue
                elif text.lower().startswith('threshold '):
                    try:
                        new_threshold = float(text.split()[1])
                        cli.decision_maker.set_confidence_threshold(new_threshold)
                        click.echo(f"Threshold set to {new_threshold:.2f}")
                    except (IndexError, ValueError):
                        click.echo("Usage: threshold <value> (0.0-1.0)")
                    continue
                
                result = cli.decision_maker.decide(text, include_reasoning=True)
                
                _format_decision_output(ctx, result, no_reasoning=False, format='text')
                click.echo()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                click.echo(f"Error: {e}")
        
        click.echo("Goodbye!")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
