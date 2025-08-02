#!/usr/bin/env python3
"""
CLI launcher script for Simple Decision AI.

This script properly sets up the Python path and runs the CLI interface.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Now we can import and run the CLI
if __name__ == "__main__":
    from interfaces.cli import main
    main()