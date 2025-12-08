"""
Integrator interfaces and State container for translational dynamics.
Implementations are placeholders.
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
        raise NotImplementedError("Implement state copy if needed")


class Integrator:
    def step(self, deriv_fn: Callable[[float, State], Tuple[np.ndarray, np.ndarray, float]], state: State, t: float, dt: float) -> State:
        raise NotImplementedError("Implement integrator step")


class RK4(Integrator):
    def step(self, deriv_fn: Callable[[float, State], Tuple[np.ndarray, np.ndarray, float]], state: State, t: float, dt: float) -> State:
        raise NotImplementedError("Implement RK4 step")
