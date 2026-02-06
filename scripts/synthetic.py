"""
Main entry point: comprehensive neuron identification test
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import run_comprehensive_test

if __name__ == "__main__":
    run_comprehensive_test()
