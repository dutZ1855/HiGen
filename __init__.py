"""Prototype system for RL-driven compiler fuzzing.

This package implements a reinforcement-learning based compiler fuzzing
system that learns to choose effective test configurations to improve
compiler bug discovery efficiency.

Main components:
- `CompilerFuzzConfig`: system configuration management
- `CompilerFuzzEnv`: reinforcement learning environment interface

The system uses a hierarchical RL strategy:
1. High-level agent selects test dimensions (operator types, dtypes, etc.)
2. Low-level agent optimizes concrete parameters under fixed dimensions
3. Rewards guide agents toward finding more compiler bugs
"""

from .config import CompilerFuzzConfig
from .env import CompilerFuzzEnv

__all__ = ["CompilerFuzzConfig", "CompilerFuzzEnv"]

