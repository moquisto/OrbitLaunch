"""
Numerical integrators for the translational equations of motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class State:
    r_eci: np.ndarray
    v_eci: np.ndarray
    m: float
    stage_index: int = 0

    def copy(self) -> "State":
        return State(self.r_eci.copy(), self.v_eci.copy(), float(self.m), int(self.stage_index))


class Integrator:
    def step(self, deriv_fn: Callable[[float, State], Tuple[np.ndarray, np.ndarray, float]], state: State, t: float, dt: float) -> State:
        raise NotImplementedError


class RK4(Integrator):
    def step(self, deriv_fn: Callable[[float, State], Tuple[np.ndarray, np.ndarray, float]], state: State, t: float, dt: float) -> State:
        k1_r, k1_v, k1_m = deriv_fn(t, state)

        s2 = _state_increment(state, k1_r, k1_v, k1_m, dt * 0.5)
        k2_r, k2_v, k2_m = deriv_fn(t + 0.5 * dt, s2)

        s3 = _state_increment(state, k2_r, k2_v, k2_m, dt * 0.5)
        k3_r, k3_v, k3_m = deriv_fn(t + 0.5 * dt, s3)

        s4 = _state_increment(state, k3_r, k3_v, k3_m, dt)
        k4_r, k4_v, k4_m = deriv_fn(t + dt, s4)

        dr = (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        dv = (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        dm = (dt / 6.0) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m)

        return _state_increment(state, dr, dv, dm, scale=1.0)


def _state_increment(state: State, dr: np.ndarray, dv: np.ndarray, dm: float, scale: float) -> State:
    return State(
        r_eci=state.r_eci + dr * scale,
        v_eci=state.v_eci + dv * scale,
        m=state.m + dm * scale,
        stage_index=state.stage_index,
    )
