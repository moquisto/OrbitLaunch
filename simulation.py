"""
Simulation glue code: guidance, events, logging, and integration loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from aerodynamics import Aerodynamics
from atmosphere import CombinedAtmosphere
from gravity import EarthModel
from integrators import Integrator, RK4, State
from rocket import Rocket


@dataclass
class ControlCommand:
    throttle: float
    thrust_direction: np.ndarray  # unit vector in ECI


class Guidance:
    """
    Deterministic guidance stub: provides throttle and thrust direction.
    """

    def __init__(
        self,
        pitch_program: Optional[Callable[[float, State], np.ndarray]] = None,
        throttle_schedule: Optional[Callable[[float, State], float]] = None,
    ):
        self.pitch_program = pitch_program or (lambda t, state: np.array([1.0, 0.0, 0.0]))
        self.throttle_schedule = throttle_schedule or (lambda t, state: 1.0)

    def compute_command(self, t: float, state: State) -> ControlCommand:
        direction = np.array(self.pitch_program(t, state), dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        direction_unit = direction / norm
        throttle = float(self.throttle_schedule(t, state))
        return ControlCommand(throttle=max(0.0, min(1.0, throttle)), thrust_direction=direction_unit)


class Logger:
    """
    Minimal in-memory logger for trajectories.
    """

    def __init__(self):
        self.t = []
        self.r = []
        self.v = []
        self.m = []
        self.stage = []

    def record(self, t: float, state: State):
        self.t.append(float(t))
        self.r.append(state.r_eci.copy())
        self.v.append(state.v_eci.copy())
        self.m.append(float(state.m))
        self.stage.append(int(state.stage_index))


class Simulation:
    def __init__(
        self,
        earth: EarthModel,
        atmosphere: CombinedAtmosphere,
        aerodynamics: Aerodynamics,
        rocket: Rocket,
        integrator: Optional[Integrator] = None,
        guidance: Optional[Guidance] = None,
    ):
        self.earth = earth
        self.atmosphere = atmosphere
        self.aero = aerodynamics
        self.rocket = rocket
        self.integrator = integrator or RK4()
        self.guidance = guidance or Guidance()

    def _derivatives(self, t: float, state: State, control: ControlCommand):
        r = state.r_eci
        v = state.v_eci
        mass = max(state.m, 1e-6)  # avoid divide-by-zero

        a_grav = self.earth.gravity_accel(r)

        altitude = np.linalg.norm(r) - self.earth.radius
        props = self.atmosphere.properties(altitude, t)
        p_amb = props.get("p", 0.0)

        F_drag = self.aero.drag_force(state, self.rocket, self.earth, t)
        F_thrust, mdot = self.rocket.thrust_and_mass_flow(control, state, p_amb)

        a_drag = F_drag / mass
        a_thrust = F_thrust / mass

        drdt = v
        dvdt = a_grav + a_drag + a_thrust
        dmdt = mdot

        return drdt, dvdt, dmdt

    def run(self, t0: float, tf: float, dt: float, state0: State) -> Logger:
        """
        Execute the time-marching simulation and return a trajectory log.
        Stage separation and other events can be inserted in the loop later.
        """
        logger = Logger()
        t = t0
        state = state0

        while t <= tf:
            logger.record(t, state)

            control = self.guidance.compute_command(t, state)
            deriv_fn = lambda tau, s: self._derivatives(tau, s, control)
            state = self.integrator.step(deriv_fn, state, t, dt)

            t += dt

        return logger
