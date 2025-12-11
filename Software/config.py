"""
Configuration for the software components (guidance, mission profile, events).
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SoftwareConfig:
    # --- Guidance Programs ---

    # Pitch Program (Gravity Turn) — time-based, separate booster/upper schedules
    pitch_guidance_mode: str = "parameterized"
    pitch_guidance_function: str = "custom_guidance.simple_pitch_program"
    # Booster pitch schedule (time from liftoff, deg from horizontal)
    # Realistic duration is ~145s, so points beyond that are not useful.
    pitch_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 89.8],    # Vertical clear of tower
            [10.0, 86.0],   # Initiate kick early
            [40.0, 72.0],   # Aggressive turn through lower atmosphere
            [80.0, 55.0],   # Punch through max Q
            [120.0, 40.0],  # Stratosphere transition
            [140.0, 25.0],  # Final moments before MECO
        ]
    )
    # Upper-stage pitch schedule (time from upper ignition, deg from horizontal)
    upper_pitch_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 10.0],   # Gentle initial pitch up to preserve altitude
            [60.0, 0.0],   # Transition to prograde during upper burn
            [180.0, 0.0],  # Hold prograde through main upper-stage burn
        ]
    )
    pitch_prograde_speed_threshold: float = 100.0

    # Throttle Program (Max-Q Bucket)
    throttle_guidance_mode: str = "parameterized"
    throttle_guidance_function_class: str = "custom_guidance.TwoPhaseUpperThrottle"
    
    # Booster: Liftoff 100%, Dip for Max Q, Back to 100%
    # Realistic duration is ~145s.
    booster_throttle_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 1.0],    # ALL ENGINES GO
            [50.0, 1.0],   # Approaching Max Q speed
            [60.0, 0.65],  # THROTTLE BUCKET: Reduce to 65% to save structure
            [75.0, 0.65],  # Hold bucket
            [85.0, 1.0],   # Max Q passed, power up
            [150.0, 1.0],  # Burn to depletion/MECO
        ]
    )
    
    # Upper Stage: Simple burn
    upper_stage_throttle_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 1.0],    # Full power
            [350.0, 1.0],  # Long burn to orbit
            [351.0, 0.0],  # SECO 1 (Main orbital insertion)
            [2500.0, 0.0], # Coast phase
            [2501.0, 1.0], # Circularization burn (if needed)
            [2520.0, 1.0], 
            [2521.0, 0.0], # Done
        ]
    )
    
    upper_throttle_vr_tolerance: float = 2.0
    upper_throttle_alt_tolerance: float = 1000.0
    base_throttle_cmd: float = 1.0