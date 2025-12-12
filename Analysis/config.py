"""
Configuration for analysis and optimization tasks.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class OptimizationParams:
    """A structured container for the 35 parameters being optimized."""
    meco_mach: float
    booster_pitch_time_0: float
    booster_pitch_angle_0: float
    booster_pitch_time_1: float
    booster_pitch_angle_1: float
    booster_pitch_time_2: float
    booster_pitch_angle_2: float
    booster_pitch_time_3: float
    booster_pitch_angle_3: float
    booster_pitch_time_4: float
    booster_pitch_angle_4: float
    coast_s: float
    upper_burn_s: float
    upper_ignition_delay_s: float
    azimuth_deg: float
    upper_pitch_time_0: float
    upper_pitch_angle_0: float
    upper_pitch_time_1: float
    upper_pitch_angle_1: float
    upper_pitch_time_2: float
    upper_pitch_angle_2: float
    upper_throttle_level_0: float
    upper_throttle_level_1: float
    upper_throttle_level_2: float
    upper_throttle_level_3: float
    upper_throttle_switch_ratio_0: float
    upper_throttle_switch_ratio_1: float
    upper_throttle_switch_ratio_2: float
    booster_throttle_level_0: float
    booster_throttle_level_1: float
    booster_throttle_level_2: float
    booster_throttle_level_3: float
    booster_throttle_switch_ratio_0: float
    booster_throttle_switch_ratio_1: float
    booster_throttle_switch_ratio_2: float


@dataclass
class AnalysisConfig:
    # Optimizer (optional manual seed)
    optimizer_manual_seed: list | None = None
