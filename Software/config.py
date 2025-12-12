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

    # Engine and Staging Parameters (moved from HardwareConfig / Rocket)
    main_engine_ramp_time: float = 3.0           # [s] (time for engines to reach full thrust from ignition)
    upper_engine_ramp_time: float = 1.0          # [s]
    meco_mach: float = 4.0                       # [Mach] (Mach number at which booster MECO occurs)
    separation_delay_s: float = 2.0              # [s] (time between booster MECO and stage separation)
    upper_ignition_delay_s: float = 2.0          # [s] (time between stage separation and upper stage ignition)
    engine_min_throttle: float = 0.4             # (Raptor deep throttle limit)
    engine_shutdown_ramp_s: float = 0.5          # [s] (time for engines to ramp down to zero thrust)
    throttle_full_shape_threshold: float = 0.99  # (shape value considered "full" for min throttle enforcement)

    def create_pitch_program(self, env_config: EnvironmentConfig):
        from Software.guidance import StageAwarePitchProgram # Local import
        if self.pitch_guidance_mode == 'parameterized':
            return StageAwarePitchProgram(
                sw_config=self,
                env_config=env_config
            )
        elif self.pitch_guidance_mode == 'function':
            try:
                module_name, func_name = self.pitch_guidance_function.rsplit('.', 1)
                module = importlib.import_module(module_name)
                return getattr(module, func_name)
            except (ImportError, AttributeError, ValueError) as e:
                raise ImportError(f"Could not load pitch guidance function '{self.pitch_guidance_function}': {e}")
        else:
            raise ValueError(f"Unknown pitch_guidance_mode: '{self.pitch_guidance_mode}'")

    def create_throttle_program(self, target_radius: float, mu: float):
        from Software.guidance import ParameterizedThrottleProgram # Local import
        if self.throttle_guidance_mode == 'parameterized':
            # This returns a ParameterizedThrottleProgram for the upper stage.
            # The booster throttle program needs to be handled separately where the Rocket is built.
            return ParameterizedThrottleProgram(schedule=self.upper_stage_throttle_program)
        elif self.throttle_guidance_mode == 'function':
            try:
                module_name, class_name = self.throttle_guidance_function_class.rsplit('.', 1)
                module = importlib.import_module(module_name)
                ControllerClass = getattr(module, class_name)
                return ControllerClass(target_radius=target_radius, mu=mu)
            except (ImportError, AttributeError, ValueError) as e:
                raise ImportError(f"Could not load throttle guidance class '{self.throttle_guidance_function_class}': {e}")
        else:
            raise ValueError(f"Unknown throttle_guidance_mode: '{self.throttle_guidance_mode}'")

    def create_guidance(self, pitch_program, throttle_schedule, booster_throttle_schedule, rocket_stages_info, env_config: EnvironmentConfig) -> "Guidance": # Add env_config
        from Software.guidance import Guidance # Local import
        return Guidance(
            sw_config=self,
            env_config=env_config, # Use the passed env_config instance
            pitch_program=pitch_program,
            upper_throttle_program=throttle_schedule, # This is the upper stage throttle program
            booster_throttle_schedule=booster_throttle_schedule,
            rocket_stages_info=rocket_stages_info
        )