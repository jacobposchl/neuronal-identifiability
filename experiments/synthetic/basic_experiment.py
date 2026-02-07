"""
Basic synthetic neuron identification experiment.

Main entry point for running the comprehensive neuron identification test
on synthetic neural populations with known ground truth labels.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.evaluation import run_comprehensive_test

if __name__ == "__main__":
    print("\n" + "="*70)
    print("SYNTHETIC NEURON IDENTIFICATION EXPERIMENT")
    print("="*70)
    print("\nTesting deformation-based classification on synthetic populations")
    print("with known ground truth functional roles (Rotation, Contraction,")
    print("Expansion, State-encoding).\n")
    
    run_comprehensive_test()
