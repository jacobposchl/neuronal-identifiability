"""
Cognitive task definitions and generators.

This module contains implementations of various cognitive tasks
for training and testing RNNs.

Intended contents:
- Task base classes
- Specific task implementations (FlipFlop, Memory, Context, etc.)
- Task factory functions
"""

from .tasks import (
    TaskBase,
    FlipFlopTask,
    CyclingMemoryTask,
    ContextIntegrationTask,
    SequentialMNISTTask,
    ParametricWorkingMemoryTask,
    DelayedMatchToSampleTask,
    GoNoGoTask,
    FiniteStateMachineTask,
    get_task
)

__all__ = [
    'TaskBase',
    'FlipFlopTask',
    'CyclingMemoryTask',
    'ContextIntegrationTask',
    'SequentialMNISTTask',
    'ParametricWorkingMemoryTask',
    'DelayedMatchToSampleTask',
    'GoNoGoTask',
    'FiniteStateMachineTask',
    'get_task',
]
