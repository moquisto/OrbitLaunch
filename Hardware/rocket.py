"""
Vehicle model: outlines for engines, stages, and rocket behavior.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np
from ..config import Config  # Keep for now until main config is fully dismantled
from .stage import Stage     # New import for Stage
from .config import HardwareConfig
from Environment.config import EnvironmentConfig


class Rocket:
    """
    Simple two-stage rocket model.

    Assumptions
    -----------
    * stages[0] is the booster (first stage).
    * stages[1] is the upper stage / payload stage.
    * The overall vehicle mass is carried in State.m and is updated by the
      integrator using the returned mass-flow rate dm_dt.
    * This class does NOT mutate the State object; it only reads from it.
      Stage index changes (state.stage_index) and mass drops at separation
      should be handled in the outer simulation code.

    Throttle schedule
    -----------------
    * Booster (stage_index == 0):
        - Thrust ramps linearly from 0 to full over `main_engine_ramp_time`
          seconds from liftoff (t=0).
        - After ramp-up, thrust is held constant until either the vehicle reaches
          `meco_mach` (approximate Mach number based on inertial speed) or booster
          propellant depletion is detected.
        - At MECO or depletion, thrust is cut to zero permanently, the internal
          `meco_time` is recorded (for Mach-driven cutoff), and separation/upper
          ignition times are scheduled.

    * Upper stage (stage_index == 1):
        - No thrust until `meco_time` has been recorded and a jettison delay
          of `separation_delay` seconds has elapsed.
        - Then the upper-stage engine ramps linearly from 0 to full thrust
          over `upper_engine_ramp_time` seconds.
        - After ramp-up, thrust is kept constant.

    The user still decides when to actually change `state.stage_index`
    (and adjust State.m for the dropped booster). A convenient strategy is:
        - Use rocket.meco_time to detect MECO.
        - At t = meco_time + separation_delay, drop the booster mass and set
          state.stage_index = 1.
    """

    def __init__(
        self,
        stages: List[Stage],
        hw_config: HardwareConfig,
        env_config: EnvironmentConfig,
        booster_throttle_program: Optional[Any] = None,
    ):
        if len(stages) < 2:
            raise ValueError("Rocket expects at least two stages (booster + upper stage).")

        self.stages = stages
        self.hw_config = hw_config
        self.env_config = env_config
        self.main_engine_ramp_time = float(hw_config.main_engine_ramp_time)
        self.upper_engine_ramp_time = float(hw_config.upper_engine_ramp_time)
        self.meco_mach = float(hw_config.meco_mach)
        self.separation_delay = float(hw_config.separation_delay_s)
        self.upper_ignition_delay = float(hw_config.upper_ignition_delay_s)
        self.separation_altitude_m = hw_config.separation_altitude_m
        self.earth_radius = float(env_config.earth_radius_m)
        self.min_throttle = float(np.clip(hw_config.engine_min_throttle, 0.0, 1.0))
        self.shutdown_ramp_time = float(max(hw_config.engine_shutdown_ramp_s, 0.0))
        self.throttle_shape_full_threshold = float(np.clip(hw_config.throttle_full_shape_threshold, 0.0, 1.0))

        self.booster_throttle_program = booster_throttle_program

        # Internal state for event timing
        self.meco_time: float | None = None
        self._last_time: float = 0.0

        # Per-stage propellant bookkeeping for scheduling (independent of State.m)
        self.stage_prop_remaining = [s.prop_mass for s in stages]
        self.stage_fuel_empty_time: list[float | None] = [None] * len(stages)
        self.stage_engine_off_complete_time: list[float | None] = [None] * len(stages)

        # Planned timeline events driven by stage 1 fuel depletion unless overridden by altitude:
        #  - separation_time_planned: booster jettison time (if no altitude specified)
        #  - upper_ignition_start_time: upper-stage ramp-up start time
        self.separation_time_planned: float | None = None
        self.upper_ignition_start_time: float | None = None
        self.reset()


    def reset(self):
        """Reset all internal time-varying state for a new simulation run."""
        self.meco_time = None
        self._last_time = 0.0
        self.stage_prop_remaining = [s.prop_mass for s in self.stages]
        self.stage_fuel_empty_time = [None] * len(self.stages)
        self.stage_engine_off_complete_time = [None] * len(self.stages)
        self.separation_time_planned = None
        self.upper_ignition_start_time = None

    # ------------------------------------------------------------------
    # Helper methods for stage selection and geometry
    # ------------------------------------------------------------------
    def current_stage_index(self, state) -> int:
        """
        Return the clamped stage index from the simulation State.

        The State is expected to have an integer attribute `stage_index`
        (see integrators.State). Values outside [0, len(stages)-1] are
        clamped.
        """
        idx = getattr(state, "stage_index", 0)
        idx = int(np.clip(idx, 0, len(self.stages) - 1))
        return idx

    def current_stage(self, state) -> Stage:
        """Return the currently active stage object."""
        return self.stages[self.current_stage_index(state)]

    def reference_area(self, state) -> float:
        """Return the reference area [m^2] to be used for drag."""
        return self.current_stage(state).ref_area

    # ------------------------------------------------------------------
    # Thrust and mass flow
    # ------------------------------------------------------------------
    def _booster_thrust(self, t: float, state: Any, throttle_cmd: float, mach: float) -> float:
        """Calculates the throttle shape for the booster stage."""
        fuel_empty_time = self.stage_fuel_empty_time[0]
        off_time = self.stage_engine_off_complete_time[0]

        if self.meco_time is not None and t >= self.meco_time:
            # Post-MECO: engine stays off.
            shape = 0.0
        elif fuel_empty_time is None:
            # 1. Booster has fuel. Normal operation.
            if t < self.main_engine_ramp_time:
                # Ramp-up phase from t=0.
                shape = max(0.0, t / self.main_engine_ramp_time)
            else:
                # Use booster throttle program if defined, otherwise full thrust.
                if self.booster_throttle_program:
                    # Throttle based on time for the booster
                    # The program is a callable, so we call it with time.
                    shape = self.booster_throttle_program(t, state)
                else:
                    shape = 1.0
        else:
            # 2. Booster has run out of fuel.
            # The `stage_fuel_empty_time` is set when propellant tracking shows empty.
            ramp_end = fuel_empty_time + self.shutdown_ramp_time
            if t < ramp_end:
                # Ramp-down phase.
                shape = max(0.0, 1.0 - (t - fuel_empty_time) / max(self.shutdown_ramp_time, 1e-9))
            else:
                # Engine is fully off.
                shape = 0.0
                # If this is the first time we've noted the engine is off,
                # schedule the future events: stage separation and upper stage ignition.
                if off_time is None:
                    off_time = ramp_end
                    self.stage_engine_off_complete_time[0] = off_time
                    self.separation_time_planned = off_time + self.separation_delay
                    self.upper_ignition_start_time = self.separation_time_planned + self.upper_ignition_delay
        
        # Detect MECO event by Mach threshold (booster only) and schedule timeline.
        if (
            self.meco_time is None
            and mach >= self.meco_mach
        ):
            self.meco_time = t
            # Mark booster engine fully off at MECO and schedule downstream events.
            if self.stage_engine_off_complete_time[0] is None:
                self.stage_engine_off_complete_time[0] = t
            if self.separation_time_planned is None:
                self.separation_time_planned = t + self.separation_delay
            if self.upper_ignition_start_time is None:
                self.upper_ignition_start_time = self.separation_time_planned + self.upper_ignition_delay
            shape = 0.0 # cut off thrust immediately
            
        return shape

    def _upper_stage_thrust(self, t: float, state: Any, throttle_cmd: float) -> float:
        """Calculates the throttle shape for the upper stage."""
        shape = 0.0
        fuel_empty_time = self.stage_fuel_empty_time[1]
        off_time = self.stage_engine_off_complete_time[1]
        ignition_start = self.upper_ignition_start_time

        if ignition_start is not None and t >= ignition_start:
            # 1. It is time to ignite the upper stage.
            t_rel = t - ignition_start
            if fuel_empty_time is None:
                # 1a. Upper stage has fuel.
                if t_rel < self.upper_engine_ramp_time:
                    # Ramp-up phase.
                    shape = max(0.0, t_rel / self.upper_engine_ramp_time)
                else:
                    # Full thrust phase.
                    shape = 1.0
            else:
                # 1b. Upper stage has run out of fuel.
                ramp_end = fuel_empty_time + self.shutdown_ramp_time
                if t < ramp_end:
                    # Ramp-down phase.
                    shape = max(0.0, 1.0 - (t - fuel_empty_time) / max(self.shutdown_ramp_time, 1e-9))
                else:
                    # Engine is fully off.
                    shape = 0.0
                    if off_time is None:
                        self.stage_engine_off_complete_time[1] = ramp_end
        return shape

    def thrust_and_mass_flow(self, control, state, p_amb: float) -> Tuple[np.ndarray, float]:
        """
        Compute thrust vector and mass flow rate for the current state.
        ...
        """
        # ... (control extraction and time logic remains the same)
        if isinstance(control, dict):
            t = float(control.get("t", 0.0))
            throttle_cmd = float(control.get("throttle", 1.0))
            thrust_dir = control.get("thrust_dir_eci", None)
            dir_is_unit = bool(control.get("dir_is_unit", False))
            speed_override = control.get("speed", None)
            v_override = control.get("velocity_vec", None)
            air_speed_override = control.get("air_speed", None)
            air_v_override = control.get("air_velocity_vec", None)
        else:
            t = float(getattr(control, "t", 0.0))
            throttle_cmd = float(getattr(control, "throttle", 1.0))
            thrust_dir = getattr(control, "thrust_dir_eci", None)
            dir_is_unit = bool(getattr(control, "dir_is_unit", False))
            speed_override = getattr(control, "speed", None)
            v_override = getattr(control, "velocity_vec", None)
            air_speed_override = getattr(control, "air_speed", None)
            air_v_override = getattr(control, "air_velocity_vec", None)

        prev_t = self._last_time
        dt_internal = t - prev_t if t >= prev_t else 0.0
        self._last_time = t

        v_vec = np.asarray(v_override, dtype=float) if v_override is not None else np.asarray(state.v_eci, dtype=float)
        speed = float(speed_override) if speed_override is not None else float(np.linalg.norm(v_vec))

        altitude = np.linalg.norm(state.r_eci) - self.earth_radius
        # TODO: This line needs to be refactored to get speed of sound from an AtmosphereModel instance
        # local_speed_of_sound = self.config.get_speed_of_sound(altitude)
        local_speed_of_sound = 340.29 # Placeholder for now

        mach_vec = np.asarray(air_v_override, dtype=float) if air_v_override is not None else v_vec
        mach_speed = float(air_speed_override) if air_speed_override is not None else float(np.linalg.norm(mach_vec))
        mach = mach_speed / local_speed_of_sound if local_speed_of_sound > 0.0 else 0.0

        stage_idx = self.current_stage_index(state)
        stage = self.stages[stage_idx]

        if self.stage_prop_remaining[stage_idx] <= 0.0 and self.stage_fuel_empty_time[stage_idx] is None:
            self.stage_fuel_empty_time[stage_idx] = t

        shape = 0.0
        if stage_idx == 0:
            shape = self._booster_thrust(t, state, throttle_cmd, mach)
        else:
            shape = self._upper_stage_thrust(t, state, throttle_cmd)
        
        effective_throttle = np.clip(throttle_cmd * shape, 0.0, 1.0)
        if (
            shape >= self.throttle_shape_full_threshold
            and effective_throttle > 0.0
            and self.min_throttle > 0.0
        ):
            effective_throttle = max(effective_throttle, self.min_throttle)

        thrust_mag, isp = stage.engine.thrust_and_isp(effective_throttle, p_amb)

        if self.stage_prop_remaining[stage_idx] <= 0.0:
            thrust_mag = 0.0
            if self.stage_fuel_empty_time[stage_idx] is None:
                self.stage_fuel_empty_time[stage_idx] = t

        if thrust_dir is None:
            if speed > 0.0:
                dir_vec = v_vec / speed
            else:
                dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            dir_vec = np.asarray(thrust_dir, dtype=float)
            if not dir_is_unit:
                n = np.linalg.norm(dir_vec)
                if n == 0.0:
                    dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)
                else:
                    dir_vec = dir_vec / n

        thrust_vec = thrust_mag * dir_vec

        if thrust_mag > 0.0 and isp > 0.0:
            dm_dt = -thrust_mag / (isp * self.env_config.G0)
        else:
            dm_dt = 0.0

        if dt_internal > 0.0 and dm_dt < 0.0:
            burned = -dm_dt * dt_internal
            self.stage_prop_remaining[stage_idx] = max(
                0.0, self.stage_prop_remaining[stage_idx] - burned
            )
            if (
                self.stage_prop_remaining[stage_idx] <= 0.0
                and self.stage_fuel_empty_time[stage_idx] is None
            ):
                self.stage_fuel_empty_time[stage_idx] = t

        if self.stage_prop_remaining[stage_idx] <= 0.0:
            thrust_vec = np.zeros_like(thrust_vec)
            thrust_mag = 0.0
            dm_dt = 0.0

        return thrust_vec, dm_dt
