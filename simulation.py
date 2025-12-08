"""
Simulation glue outline: guidance, events, logging, and integration loop.
Behavior is left unimplemented to serve as a structural template.
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
        self.pitch_program = pitch_program
        self.throttle_schedule = throttle_schedule

    def compute_command(self, t: float, state: State) -> ControlCommand:
        raise NotImplementedError("Implement guidance law")


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
        raise NotImplementedError("Implement logging behavior")


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
        raise NotImplementedError("Implement equations of motion")

    def run(self, t0: float, tf: float, dt: float, state0: State) -> Logger:
        raise NotImplementedError("Implement main simulation loop")
