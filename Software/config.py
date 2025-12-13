"""
Configuration for the software components (guidance, mission profile, events).
"""

import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional
import importlib
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

# from Software.guidance import Guidance # For type hinting
from Environment.config import EnvironmentConfig


if TYPE_CHECKING:
    from Software.guidance import Guidance

@dataclass
class SoftwareConfig:
    # --- Guidance Programs ---

    # Pitch Program (Gravity Turn) — time-based, separate booster/upper schedules
    pitch_guidance_mode: str = "parameterized"
    pitch_guidance_function: str = "custom_guidance.simple_pitch_program"
    # Launch azimuth in degrees from East toward North. Used as the horizontal
    # reference direction for pitch programs (az=0 is due East).
    launch_azimuth_deg: float = 0.0
    # Booster pitch schedule (time from liftoff, deg from horizontal)
    # Realistic duration is ~145s, so points beyond that are not useful.
    pitch_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 89.8],    # Vertical clear of tower
            [40.0, 72.0],   # Aggressive turn through lower atmosphere
            [80.0, 55.0],   # Punch through max Q
            [120.0, 40.0],  # Stratosphere transition
            [140.0, 25.0],  # Final moments before MECO
        ]
    )
    # Upper-stage pitch schedule (time from upper ignition, deg from horizontal)
    upper_pitch_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 60.0],    # Start with a higher pitch
            [120.0, 10.0],  # Pitch down
            [240.0, 0.0],   # Delay full prograde
        ]
    )
    pitch_prograde_speed_threshold: float = 1800.0

    # Throttle Program (Max-Q Bucket)
    throttle_guidance_mode: str = "parameterized"
    throttle_guidance_function_class: str = "custom_guidance.TwoPhaseUpperThrottle"
    
    # Booster: Liftoff 100%, Dip for Max Q, Back to 100%
    # Realistic duration is ~145s.
    booster_throttle_program: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 1.0],    # ALL ENGINES GO
            [150.0, 1.0],  # Burn to depletion/MECO - always full throttle
        ]
    )
    
    # Upper Stage: Simple burn
    upper_throttle_program_schedule: list = dataclasses.field(
        default_factory=lambda: [
            [0.0, 1.0],    # Full power
            [900.0, 1.0],  # Long burn to orbit
            [901.0, 0.0],  # SECO 1 (Main orbital insertion)
            [2500.0, 0.0], # Coast phase
            [2501.0, 1.0], # Circularization burn (if needed)
            [2520.0, 1.0], 
            [2521.0, 0.0], # Done
        ]
    )
    
    upper_throttle_vr_tolerance: float = 2.0
    upper_throttle_alt_tolerance: float = 1000.0
    base_throttle_cmd: float = 1.0

    # Engine and Staging Parameters (moved from HardwareConfig / Rocket)
    main_engine_ramp_time: float = 3.0           # [s] (time for engines to reach full thrust from ignition)
    upper_engine_ramp_time: float = 1.0          # [s]
    meco_mach: float = 6.0                       # [Mach] (Mach number at which booster MECO occurs)
    separation_delay_s: float = 2.0              # [s] (time between booster MECO and stage separation)
    upper_ignition_delay_s: float = 2.0          # [s] (time between stage separation and upper stage ignition)
    engine_min_throttle: float = 0.4             # (Raptor deep throttle limit)
    engine_shutdown_ramp_s: float = 0.5          # [s] (time for engines to ramp down to zero thrust)
    throttle_full_shape_threshold: float = 0.99  # (shape value considered "full" for min throttle enforcement)

    def create_guidance(self, rocket_stages_info, env_config: EnvironmentConfig) -> "Guidance": # Add env_config
        from Software.guidance import Guidance, StageAwarePitchProgram, ParameterizedThrottleProgram # Local import
        pitch_program_instance = StageAwarePitchProgram(sw_config=self, env_config=env_config)
        upper_throttle_program_instance = ParameterizedThrottleProgram(schedule=self.upper_throttle_program_schedule)
        booster_throttle_program_instance = ParameterizedThrottleProgram(schedule=self.booster_throttle_program)

        return Guidance(
            sw_config=self,
            env_config=env_config, # Use the passed env_config instance
            pitch_program=pitch_program_instance,
            upper_throttle_program=upper_throttle_program_instance, # This is the upper stage throttle program
            booster_throttle_program=booster_throttle_program_instance,
            rocket_stages_info=rocket_stages_info
        )
