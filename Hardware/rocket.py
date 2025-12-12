"""
Vehicle model: outlines for engines, stages, and rocket behavior.
Implementations are placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import numpy as np
from .stage import Stage
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
    ):
        if len(stages) < 2:
            raise ValueError("Rocket expects at least two stages (booster + upper stage).")

        self.stages = stages
        self.hw_config = hw_config
        self.env_config = env_config
        self.earth_radius = float(env_config.earth_radius_m)
        self.reset()


    def reset(self):
        """Reset all internal time-varying state for a new simulation run."""
        self._last_time = 0.0

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
    def thrust_and_mass_flow(
        self,
        t: float, # Current simulation time, needed for _last_time
        throttle_command: float,
        thrust_direction_eci: np.ndarray,
        state,
        p_amb: float,
        current_prop_mass: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute thrust vector and mass flow rate for the current state
        given a throttle command and thrust direction.

        Parameters
        ----------
        t : float
            Current simulation time.
        throttle_command : float
            Desired throttle setting (0.0 to 1.0).
        thrust_direction_eci : np.ndarray
            Unit vector for thrust direction in ECI frame.
        state : State
            Current simulation state.
        p_amb : float
            Ambient pressure at current altitude.
        current_prop_mass : float
            Current propellant mass of the active stage.

        Returns
        -------
        Tuple[np.ndarray, float]
            Thrust vector [N] and mass flow rate [kg/s].
        """
        
        prev_t = self._last_time
        self._last_time = t

        stage_idx = self.current_stage_index(state)
        stage = self.stages[stage_idx]

        # Apply throttle command directly
        effective_throttle = np.clip(throttle_command, 0.0, 1.0)

        thrust_mag, isp = stage.engine.thrust_and_isp(effective_throttle, p_amb)

        # If no propellant, no thrust and no mass flow
        if current_prop_mass <= 0.0:
            thrust_mag = 0.0
            dm_dt = 0.0
        elif thrust_mag > 0.0 and isp > 0.0:
            dm_dt = -thrust_mag / (isp * self.env_config.G0)
        else:
            dm_dt = 0.0

        thrust_vec = thrust_mag * thrust_direction_eci

        return thrust_vec, dm_dt
