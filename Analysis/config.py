"""
Configuration for analysis and optimization tasks.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class AnalysisConfig:
    # Optimizer (optional manual seed)
    optimizer_manual_seed: list | None = None
