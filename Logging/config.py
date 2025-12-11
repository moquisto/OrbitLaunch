"""
Configuration for logging outputs and post-simulation analysis.
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class LoggingConfig:
    # Logging
    impact_altitude_buffer_m: float = -100.0
    escape_radius_factor: float = 1.05
    log_filename: str = "simulation_log.txt"
    plot_trajectory: bool = True
    animate_trajectory: bool = False
