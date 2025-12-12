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

@dataclass
class OptimizationBounds:
    """
    Defines the bounds for the optimization parameters.
    """
    @staticmethod
    def get_bounds() -> List[Tuple[float, float]]:
        """
        Returns the hardcoded bounds for the 35 optimization parameters.
        """
        return [
            (4.5, 6.5),      # 0: MECO Mach
            # Pitch profile (5 points, time in seconds, angle in degrees relative to horizontal)
            (0.0, 20.0),     # 1: pitch_time_0 (s) - Initial liftoff phase
            (80.0, 90.0),     # 2: pitch_angle_0 (deg, 90=vertical)
            (20.0, 60.0),     # 3: pitch_time_1 (s) - Start of gravity turn
            (60.0, 85.0),     # 4: pitch_angle_1 (deg)
            (50.0, 90.0),     # 5: pitch_time_2 (s) - Approaching max Q
            (40.0, 70.0),     # 6: pitch_angle_2 (deg)
            (80.0, 120.0),    # 7: pitch_time_3 (s) - Mid-atmosphere flight
            (20.0, 50.0),     # 8: pitch_angle_3 (deg)
            (110.0, 150.0),   # 9: pitch_time_4 (s) - Final moments before MECO
            (0.0, 20.0),      # 10: pitch_angle_4 (deg)
            # Staging and upper stage burn
            (5.0, 200.0),    # 11: Coast duration after MECO (s)
            (100.0, 300.0),  # 12: Upper stage burn duration (s)
            (0.0, 60.0),     # 13: Upper stage ignition delay after separation (s)
            (-15.0, 15.0),   # 14: Azimuth heading (deg from east toward north)
            # Upper-stage pitch profile (time from upper ignition, deg from horizontal)
            (0.0, 60.0),     # 15: upper_pitch_time_0 (s)
            (5.0, 45.0),     # 16: upper_pitch_angle_0 (deg)
            (40.0, 180.0),   # 17: upper_pitch_time_1 (s)
            (0.0, 30.0),     # 18: upper_pitch_angle_1 (deg)
            (150.0, 300.0),  # 19: upper_pitch_time_2 (s)
            (0.0, 15.0),     # 20: upper_pitch_angle_2 (deg)
            # Upper stage throttle profile (4 levels, 3 switch ratios)
            (0.3, 1.0),      # 21: upper_throttle_level_0 (0-1)
            (0.3, 1.0),      # 22: upper_throttle_level_1 (0-1)
            (0.3, 1.0),      # 23: upper_throttle_level_2 (0-1)
            (0.3, 1.0),      # 24: upper_throttle_level_3 (0-1)
            (0.05, 0.4),     # 25: upper_throttle_switch_ratio_0 (0-1, fraction of burn duration)
            (0.25, 0.8),     # 26: upper_throttle_switch_ratio_1 (0-1, fraction of burn duration)
            (0.6, 0.95),     # 27: upper_throttle_switch_ratio_2 (0-1, fraction of burn duration)
            # Booster throttle profile (4 levels, 3 switch ratios)
            (0.3, 1.0),      # 28: booster_throttle_level_0 (0-1)
            (0.3, 1.0),      # 29: booster_throttle_level_1 (0-1)
            (0.3, 1.0),      # 30: booster_throttle_level_2 (0-1)
            (0.3, 1.0),      # 31: booster_throttle_level_3 (0-1)
            (0.05, 0.4),     # 32: booster_throttle_switch_ratio_0 (0-1, fraction of booster burn duration)
            (0.25, 0.8),     # 33: booster_throttle_switch_ratio_1 (0-1, fraction of booster burn duration)
            (0.6, 0.95),     # 34: booster_throttle_switch_ratio_2 (0-1, fraction of booster burn duration)
        ]
