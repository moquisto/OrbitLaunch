
"""
Vehicle model: outlines for engines, stages, and rocket behavior.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from gravity import R_EARTH
from config import CFG



@dataclass
class Engine:
    thrust_vac: float
    thrust_sl: float
    isp_vac: float
    isp_sl: float

    def thrust_and_isp(self, throttle: float, p_amb: float) -> Tuple[float, float]:
        """
        Compute engine thrust and Isp for a given throttle and ambient pressure.

        Parameters
        ----------
        throttle : float
            Commanded throttle in [0, 1].
        p_amb : float
            Ambient static pressure [Pa]. Typically provided by the atmosphere
            model using the current altitude.

        Returns
        -------
        thrust : float
            Thrust magnitude [N] at the requested operating point.
        isp : float
            Specific impulse [s] at the requested operating point.

        Notes
        -----
        We interpolate linearly between sea-level (p = P_SL) and vacuum
        (p = 0) performance based on p_amb, then scale by throttle.
        """
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # Clamp ambient pressure to [0, P_SL] for interpolation
        p = float(np.clip(p_amb, 0.0, CFG.P_SL))
        # Fraction of "vacuum-ness": 0 at sea level, 1 in vacuum
        f_vac = 1.0 - p / CFG.P_SL

        thrust_nominal = self.thrust_sl + f_vac * (self.thrust_vac - self.thrust_sl)
        isp_nominal = self.isp_sl + f_vac * (self.isp_vac - self.isp_sl)

        thrust = throttle * thrust_nominal
        return thrust, isp_nominal


@dataclass
class Stage:
    dry_mass: float
    prop_mass: float
    engine: Engine
    ref_area: float

    def total_mass(self) -> float:
        """
        Return total stage mass (dry + propellant).

        The time evolution of the vehicle's total mass is handled via the
        State.m variable in the integrator; this value is mainly useful
        for constructing the initial mass budget or for stage-drop deltas.
        """
        return self.dry_mass + self.prop_mass



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
        main_engine_ramp_time: float = 1.0,
        upper_engine_ramp_time: float = 1.0,
        meco_mach: float = 6.0,
        separation_delay: float = 30.0,
        upper_ignition_delay: float = 30.0,
        separation_altitude_m: Optional[float] = None,
        earth_radius: float = R_EARTH,
        min_throttle: float = 0.0,
        shutdown_ramp_time: float = 1.0,
        throttle_shape_full_threshold: float = 0.99,
        mach_ref_speed: float | None = None,
        booster_throttle_program: Optional[Any] = None, # Changed type hint to Any for now
    ):
        if len(stages) < 2:
            raise ValueError("Rocket expects at least two stages (booster + upper stage).")

        self.stages = stages
        self.main_engine_ramp_time = float(main_engine_ramp_time)
        self.upper_engine_ramp_time = float(upper_engine_ramp_time)
        self.meco_mach = float(meco_mach)
        self.separation_delay = float(separation_delay)
        self.upper_ignition_delay = float(upper_ignition_delay)
        self.separation_altitude_m = separation_altitude_m
        self.earth_radius = float(earth_radius)
        self.min_throttle = float(np.clip(min_throttle, 0.0, 1.0))
        self.shutdown_ramp_time = float(max(shutdown_ramp_time, 0.0))
        self.throttle_shape_full_threshold = float(np.clip(throttle_shape_full_threshold, 0.0, 1.0))
        self.mach_ref_speed = float(mach_ref_speed) if mach_ref_speed is not None else CFG.mach_reference_speed
        self.booster_throttle_program = booster_throttle_program # Store the program

        # Internal state for event timing
        self.meco_time: float | None = None  # time when Mach first exceeds meco_mach
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
    def thrust_and_mass_flow(self, control, state, p_amb: float) -> Tuple[np.ndarray, float]:
        """
        Compute thrust vector and mass flow rate for the current state.

        Parameters
        ----------
        control :
            An object or dict providing at least:
                * t : float
                    Current simulation time [s].
                * throttle : float in [0, 1] (optional, default=1)
                    High-level throttle scaling (e.g. for guidance).
                * thrust_dir_eci : np.ndarray shape (3,) (optional)
                    Desired thrust direction in ECI frame. If omitted,
                    thrust is aligned with the velocity vector, or +z if
                    the vehicle is stationary.
        state :
            The current State instance (r_eci, v_eci, m, stage_index).
        p_amb : float
            Ambient static pressure [Pa], typically from the atmosphere
            model at the vehicle altitude.

        Returns
        -------
        thrust_vec : np.ndarray
            Thrust vector in ECI coordinates [N].
        dm_dt : float
            Mass flow rate [kg/s]. This value is negative when thrust
            is positive (propellant is consumed).
        """
        # Extract control fields in a robust way (support dicts and simple objects)
        if isinstance(control, dict):
            t = float(control.get("t", 0.0))
            throttle_cmd = float(control.get("throttle", 1.0))
            thrust_dir = control.get("thrust_dir_eci", None)
            dir_is_unit = bool(control.get("dir_is_unit", False))
            speed_override = control.get("speed", None)
            v_override = control.get("velocity_vec", None)
        else:
            t = float(getattr(control, "t", 0.0))
            throttle_cmd = float(getattr(control, "throttle", 1.0))
            thrust_dir = getattr(control, "thrust_dir_eci", None)
            dir_is_unit = bool(getattr(control, "dir_is_unit", False))
            speed_override = getattr(control, "speed", None)
            v_override = getattr(control, "velocity_vec", None)

        # Estimate local time step based on the last call (for propellant bookkeeping)
        prev_t = self._last_time
        dt_internal = t - prev_t if t >= prev_t else 0.0
        self._last_time = t

        # Determine an approximate Mach number based on inertial speed.
        v_vec = np.asarray(v_override, dtype=float) if v_override is not None else np.asarray(state.v_eci, dtype=float)
        speed = float(speed_override) if speed_override is not None else float(np.linalg.norm(v_vec))
        mach = speed / self.mach_ref_speed if self.mach_ref_speed > 0.0 else 0.0

        stage_idx = self.current_stage_index(state)
        stage = self.stages[stage_idx]

        # Detect MECO event by Mach threshold (booster only) and schedule timeline.
        if (
            stage_idx == 0
            and self.meco_time is None
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

        # This section implements a state machine for the throttle profile of each stage.
        # The 'shape' variable is a multiplier (0.0 to 1.0) that defines the throttle
        # based on pre-defined events like ramp-up, fuel exhaustion, and ignition delays.
        # This is separate from the 'throttle_cmd' which comes from the guidance system.
        shape = 0.0

        if stage_idx == 0:
            # --- Booster Stage Logic ---
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
        else:
            # --- Upper Stage Logic ---
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
            # Note: If ignition_start is None or t < ignition_start, shape remains 0.0.

        # Combine stage shape with commanded throttle
        effective_throttle = np.clip(throttle_cmd * shape, 0.0, 1.0)
        # Enforce minimum throttle when the engine is up to speed (shape ~1) and commanded on.
        if (
            shape >= self.throttle_shape_full_threshold
            and effective_throttle > 0.0
            and self.min_throttle > 0.0
        ):
            effective_throttle = max(effective_throttle, self.min_throttle)

        # Engine performance (uses ambient pressure from the atmosphere model)
        thrust_mag, isp = stage.engine.thrust_and_isp(effective_throttle, p_amb)

        # If propellant for the active stage is gone, force thrust to zero and
        # ensure we mark fuel depletion (prevents burning past empty).
        if self.stage_prop_remaining[stage_idx] <= 0.0:
            thrust_mag = 0.0
            if self.stage_fuel_empty_time[stage_idx] is None:
                self.stage_fuel_empty_time[stage_idx] = t

        # Direction: use control thrust_dir_eci if provided, otherwise align with velocity
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
                    # Fallback if a zero vector is passed
                    dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)
                else:
                    dir_vec = dir_vec / n

        thrust_vec = thrust_mag * dir_vec

        # Mass flow: thrust = Isp * g0 * |dm_dt|
        if thrust_mag > 0.0 and isp > 0.0:
            dm_dt = -thrust_mag / (isp * CFG.G0)
        else:
            dm_dt = 0.0

        # ---------------------------------
        # Update internal propellant tracking for the active stage
        # ---------------------------------
        if dt_internal > 0.0 and dm_dt < 0.0:
            burned = -dm_dt * dt_internal  # positive propellant consumed
            self.stage_prop_remaining[stage_idx] = max(
                0.0, self.stage_prop_remaining[stage_idx] - burned
            )
            if (
                self.stage_prop_remaining[stage_idx] <= 0.0
                and self.stage_fuel_empty_time[stage_idx] is None
            ):
                # Record the first time we detect that this stage's fuel is empty
                self.stage_fuel_empty_time[stage_idx] = t

        return thrust_vec, dm_dt
