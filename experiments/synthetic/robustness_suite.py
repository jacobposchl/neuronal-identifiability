"""
Main entry point: comprehensive neuron identification test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.evaluation import run_comprehensive_test

if __name__ == "__main__":
    run_comprehensive_test()
