
"""
Vehicle model: stages, engines, thrust, and mass flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Engine:
    thrust_vac: float
    thrust_sl: float
    isp_vac: float
    isp_sl: float

    def thrust_and_isp(self, throttle: float, p_amb: float) -> Tuple[float, float]:
        """
        Return thrust [N] and specific impulse [s] for a given throttle and
        ambient pressure. Uses linear interpolation between sea level and vacuum.
        """
        throttle_clamped = float(np.clip(throttle, 0.0, 1.0))

        thrust_interp = np.interp(
            p_amb,
            [0.0, 101_325.0],
            [self.thrust_vac * throttle_clamped, self.thrust_sl * throttle_clamped],
        )
        isp_interp = np.interp(p_amb, [0.0, 101_325.0], [self.isp_vac, self.isp_sl])
        return thrust_interp, isp_interp


@dataclass
class Stage:
    dry_mass: float
    prop_mass: float
    engine: Engine
    ref_area: float

    def total_mass(self) -> float:
        return self.dry_mass + self.prop_mass


class Rocket:
    def __init__(self, stages: List[Stage]):
        if not stages:
            raise ValueError("At least one stage is required.")
        self.stages = stages

    def current_stage_index(self, state) -> int:
        return min(state.stage_index, len(self.stages) - 1)

    def current_stage(self, state) -> Stage:
        return self.stages[self.current_stage_index(state)]

    def reference_area(self, state) -> float:
        return self.current_stage(state).ref_area

    def thrust_and_mass_flow(self, control, state, p_amb: float) -> Tuple[np.ndarray, float]:
        """
        Compute thrust vector and mass flow (negative) for the active stage.
        """
        stage = self.current_stage(state)
        thrust, isp = stage.engine.thrust_and_isp(control.throttle, p_amb)
        g0 = 9.80665
        mdot = -thrust / (isp * g0)

        direction = np.array(control.thrust_direction, dtype=float)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        thrust_vec = (thrust / norm) * direction
        return thrust_vec, mdot

    def stage_separation(self, state):
        """
        Advance to the next stage and drop the current dry mass.
        The caller is responsible for ensuring propellant is depleted.
        """
        if state.stage_index >= len(self.stages) - 1:
            return state  # no more stages

        current_stage = self.current_stage(state)
        state.m = max(0.0, state.m - current_stage.dry_mass)
        state.stage_index += 1
        return state
